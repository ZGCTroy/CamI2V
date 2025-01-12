import logging
from math import sqrt

import torch
from einops import rearrange
from torch import Tensor, nn

from CameraControl.base.base import CameraControlLVDM, custom_meshgrid
from CameraControl.CamI2V.cami2v_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
)
from CameraControl.CamI2V.epipolar import Epipolar, pix2coord

mainlogger = logging.getLogger('mainlogger')


class CamI2V(CameraControlLVDM):
    def __init__(self, add_type="add_into_temporal_attn", epipolar_config=None, *args, **kwargs):
        super(CamI2V, self).__init__(*args, **kwargs)
        self.add_type = add_type

        self.epipolar_config = epipolar_config
        if self.epipolar_config is not None:
            if not hasattr(self.epipolar_config, "is_3d_full_attn"):
                self.epipolar_config.is_3d_full_attn = False
            if not hasattr(self.epipolar_config, "attention_resolution"):
                self.epipolar_config.attention_resolution = [8, 4, 2, 1]
            if not hasattr(self.epipolar_config, "apply_epipolar_soft_mask"):
                self.epipolar_config.apply_epipolar_soft_mask = False
            if not hasattr(self.epipolar_config, "soft_mask_temperature"):
                self.epipolar_config.soft_mask_temperature = 1.0
            if not hasattr(self.epipolar_config, "epipolar_hybrid_attention"):
                self.epipolar_config.epipolar_hybrid_attention = False
            if not hasattr(self.epipolar_config, "epipolar_hybrid_attention_v2"):
                self.epipolar_config.epipolar_hybrid_attention_v2 = False
            if not hasattr(self.epipolar_config, "only_self_pixel_on_current_frame"):
                self.epipolar_config.only_self_pixel_on_current_frame = False
            if not hasattr(self.epipolar_config, "current_frame_as_register_token"):
                self.epipolar_config.current_frame_as_register_token = False
            if not hasattr(self.epipolar_config, "pluker_add_type"):
                self.epipolar_config.pluker_add_type = "add_to_pre_x_only"
            if not hasattr(self.epipolar_config, "add_small_perturbation_on_zero_T"):
                self.epipolar_config.add_small_perturbation_on_zero_T = False

        bound_method = new_forward_for_unet.__get__(
            self.model.diffusion_model,
            self.model.diffusion_model.__class__
        )
        setattr(self.model.diffusion_model, 'forward', bound_method)

        for _name, _module in self.model.diffusion_model.named_modules():
            if _module.__class__.__name__ == 'TemporalTransformer':
                bound_method = new_forward_for_TemporalTransformer.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'TimestepEmbedSequential':
                bound_method = new_forward_for_TimestepEmbedSequential.__get__(_module, _module.__class__)
                setattr(_module, 'forward', bound_method)
            elif _module.__class__.__name__ == 'BasicTransformerBlock':
                # SpatialTransformer only
                if _module.context_dim is None and _module.attn1.to_k.in_features != self.model.diffusion_model.init_attn[0].proj_in.out_channels:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

                    bound_method = new_forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, 'forward', bound_method)

                    bound_method = new__forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    if self.pose_encoder is not None:
                        pluker_projection = nn.Linear(_module.attn1.to_k.in_features, _module.attn1.to_k.in_features)
                        nn.init.zeros_(list(pluker_projection.parameters())[0])
                        nn.init.zeros_(list(pluker_projection.parameters())[1])
                        pluker_projection.requires_grad_(True)
                        _module.add_module('pluker_projection', pluker_projection)
                        # _module.add_module('norm_pluker1', nn.LayerNorm(_module.attn1.to_k.in_features))
                        # _module.add_module('norm_pluker2', nn.LayerNorm(_module.attn1.to_k.in_features))

                    if self.epipolar_config is not None:
                        epipolar = Epipolar(
                            query_dim=_module.attn1.to_k.in_features,
                            context_dim=_module.attn1.to_k.in_features,
                            heads=_module.attn1.heads,
                            **self.epipolar_config
                        )
                        _module.add_module('epipolar', epipolar)
                        # _module.add_module('norm_epipolar', nn.LayerNorm(_module.attn1.to_k.in_features))

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_relative_c2w_RT_pairs(self, RT: Tensor):
        '''
        :param RT: B, T, 4 4   c2w relative RT
        :return: relative RT pairs, c2w, (B, T, T, 4, 4)
        given c2w RT, camera system transform from T1 to T2: inverse(RT_2) @ (RT_1)
        '''

        RT_inv = rearrange(RT.inverse(), "b t ... -> b 1 t ...")
        relative_RT_pairs = RT_inv @ rearrange(RT, "b t ... -> b t 1 ...")  # B, T, T, 4, 4

        return relative_RT_pairs  # B,T,T,4,4

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_fundamental_matrix(self, K: Tensor, R: Tensor, t: Tensor) -> Tensor:
        '''
        :param   K: B, 3, 3
        :param   R: B, 3, 3
        :param   t: B, 3, 1
        :return: F: B, 3, 3
        '''
        E = torch.cross(t, R, dim=-2)
        K_inv = torch.inverse(K)
        F = K_inv.transpose(-1, -2) @ E @ K_inv
        return F

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def get_epipolar_mask(self, F: Tensor, T: int, H: int, W: int, downsample: int):
        """
        modified to take in batch inputs

        Args:
            grid: (H*W, 3)
            F: camera fundamental matrix (B, T1, T2, 3, 3)
            resolution: feature map resolution H * W
            downsample: downsample scale

        return: weight matrix M(HW * HW)
        """
        # B = F.shape[0]
        device = F.device

        y = torch.arange(0, H, dtype=torch.float, device=device)  # 0 .. 128
        x = torch.arange(0, W, dtype=torch.float, device=device)  # 0 .. 84

        y = pix2coord(y, downsample)  # H
        x = pix2coord(x, downsample)  # W

        grid_y, grid_x = custom_meshgrid(y, x)  # H * W
        # grid_y: 84x128
        # 3 x HW·
        # TODO check whether yx or xy
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=2).view(-1, 3).float()  # H*W, 3

        lines = F @ grid.transpose(-1, -2)  # [B, T1, T2, 3, H*W]
        norm = torch.norm(lines[..., :2, :], dim=-2, keepdim=True)  # [B, T1, T2, 1, H*W]
        # norm = torch.where(
        #     norm == 0.0,
        #     torch.ones_like(norm),
        #     norm
        # )
        lines = lines / norm  # [B, T1, T2, 3, H*W]

        dist = (lines.transpose(-1, -2) @ grid.transpose(-1, -2)).abs()  # [B, T1, T2, H*W, H*W]
        mask = dist < (downsample * sqrt(2) / 2)  # [B, T1, T2, H*W, H*W]
        # switch to 3d full attention if epipolar mask is empty
        if self.epipolar_config.apply_epipolar_soft_mask:
            raise NotImplementedError
            mask = -dist * self.epipolar_config.soft_mask_temperature  # 高斯分布形式的权重

        if self.epipolar_config.epipolar_hybrid_attention:    # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=-1, keepdim=True), mask, torch.ones_like(mask))

        if self.epipolar_config.epipolar_hybrid_attention_v2:  # Handling Empty Epipolar Masks
            mask = torch.where(mask.any(dim=[2,4], keepdim=True).repeat(1,1,T,1,H*W), mask, torch.ones_like(mask))

        if self.epipolar_config.only_self_pixel_on_current_frame:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1)
            mask = mask * (~same_frame)  # Zero out same frame interactions

            # Step 2: Create identity mask for same pixel in the same frame
            identity_hw = torch.eye(T * H * W, device=device, dtype=mask.dtype).reshape(T, H, W, T, H, W)
            identity_hw = rearrange(
                identity_hw,
                'T1 H1 W1 T2 H2 W2 -> 1 T1 T2 (H1 W1) (H2 W2)'
            ).repeat(mask.shape[0], 1, 1, 1, 1)
            mask = torch.where(identity_hw, identity_hw, mask)

        if self.epipolar_config.current_frame_as_register_token:
            # Step 1: Zero out masks for same frame interactions
            same_frame = torch.eye(T, device=device, dtype=mask.dtype).view(1, T, T, 1, 1).repeat(mask.shape[0], 1, 1, H * W, H * W)
            mask = torch.where(same_frame, same_frame, mask)

        return rearrange(mask, "B T1 T2 HW1 HW2 -> B (T1 HW1) (T2 HW2)")

    def add_small_perturbation(self, t, epsilon=1e-6):
        zero_mask = (t.abs() < epsilon).all(dim=-2, keepdim=True)  # 检查 T 的 x, y, z 是否都接近 0
        perturbation = torch.randn_like(t) * epsilon  # 生成微小扰动
        t = torch.where(zero_mask, perturbation, t)  # 如果 T 为零，替换为扰动，否则保持原值

        return t

    def get_batch_input_camera_condition_process(self, batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame, *args, **kwargs):
        return_log = {}
        return_kwargs = {}

        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

            relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
            relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

            if self.epipolar_config is not None and not self.epipolar_config.is_3d_full_attn:
                relative_c2w_RT_4x4_pairs = self.get_relative_c2w_RT_pairs(relative_c2w_RT_4x4)  # b,t,t,4,4
                R = relative_c2w_RT_4x4_pairs[..., :3, :3]  # b,t,t,3,3
                t = relative_c2w_RT_4x4_pairs[..., :3, 3:4]  # b,t,t,3,1

                if self.epipolar_config.add_small_perturbation_on_zero_T:
                    t = self.add_small_perturbation(t, epsilon=1e-6)

                K = camera_intrinsics_3x3.unsqueeze(1)
                F = self.get_fundamental_matrix(K, R, t)
                sample_locs_dict = {d: self.get_epipolar_mask(F, T, H // d, W // d, d) for d in [int(8 * ds) for ds in self.epipolar_config.attention_resolution]}
            else:
                sample_locs_dict = None

        if self.pose_encoder is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                pluker_embedding = self.ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # b, 6, t, H, W

            pluker_embedding_features = self.pose_encoder(pluker_embedding)  # bf c h w
            pluker_embedding_features = [rearrange(_, '(b f) c h w -> b c f h w', b=batch_size) for _ in pluker_embedding_features]
        else:
            pluker_embedding_features = None

        return_kwargs["camera_condition"] = {
            "pluker_embedding_features": pluker_embedding_features,
            "sample_locs_dict": sample_locs_dict,
            "cond_frame_index": cond_frame_index,
            "add_type": self.add_type,
        }

        return return_log, return_kwargs
