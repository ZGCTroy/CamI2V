import logging
import os
import pdb
import random
from math import ceil, floor, sqrt
from uuid import uuid4

import imageio
import numpy as np
import open3d as o3d
import torch
from decord import VideoReader, cpu
from einops import rearrange, repeat
from packaging import version as pver
from torch import Tensor, nn
from torchvision import transforms

from CameraControl.CamI2V.cami2v_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
)
from CameraControl.CamI2V.epipolar import Epipolar, coord2pix, normalize, pix2coord
from CameraControl.data.realestate10k import make_spatial_transformations
from CameraControl.data.utils import (
    apply_thresholded_conv,
    constrain_to_multiple_of,
    remove_outliers,
)
from lvdm.models.ddpm3d import LatentVisualDiffusion
from scripts.gradio.preview import render
from utils.utils import instantiate_from_config

mainlogger = logging.getLogger('mainlogger')


class CamI2V(LatentVisualDiffusion):
    def __init__(self,
                 diffusion_model_trainable_param_list=[],
                 add_type='add_into_temporal_attn',
                 pose_encoder_trainable=True,
                 pose_encoder_config=None,
                 depth_predictor_config=None,
                 epipolar_config=None,
                 normalize_T0=False,
                 weight_decay=1e-2,
                 *args,
                 **kwargs):
        super(CamI2V, self).__init__(*args, **kwargs)
        self.add_type = add_type
        self.weight_decay = weight_decay
        self.normalize_T0 = normalize_T0
        self.diffusion_model_trainable_param_list = diffusion_model_trainable_param_list
        for p in self.model.parameters():
            p.requires_grad = False

        if 'TemporalTransformer.attn2' in self.diffusion_model_trainable_param_list:
            for n, m in self.model.named_modules():
                if m.__class__.__name__ == 'BasicTransformerBlock' and m.attn2.context_dim is None:  # attn2 of TemporalTransformer BasicBlocks
                    for p in m.attn2.parameters():
                        p.requires_grad = True
        elif 'TemporalTransformer.attn1' in self.diffusion_model_trainable_param_list:
            for n, m in self.model.named_modules():
                if m.__class__.__name__ == 'BasicTransformerBlock' and m.attn2.context_dim is None:  # attn1 of TemporalTransformer BasicBlocks
                    print(n)
                    for p in m.attn1.parameters():
                        p.requires_grad = True
        elif 'SpatialTransformer' in self.diffusion_model_trainable_param_list:
            for n, m in self.model.named_modules():
                if m.__class__.__name__ == 'SpatialTransformer':
                    print(n)
                    for p in m.parameters():
                        p.requires_grad = True

        # camera control module
        self.pose_encoder_trainable = pose_encoder_trainable
        if pose_encoder_config is not None:
            pose_encoder = instantiate_from_config(pose_encoder_config)
            if pose_encoder_trainable:
                self.pose_encoder = pose_encoder.train()
                for param in self.pose_encoder.parameters():
                    param.requires_grad = True
            else:
                self.pose_encoder = pose_encoder.eval()
                for param in self.pose_encoder.parameters():
                    param.requires_grad = False
        else:
            self.pose_encoder = None

        self.depth_predictor_config = depth_predictor_config
        if depth_predictor_config is not None:
            self.depth_predictor = instantiate_from_config(depth_predictor_config)
            self.depth_predictor.load_state_dict(torch.load(depth_predictor_config["pretrained_model_path"], map_location='cpu', weights_only=True))
            self.max_depth = depth_predictor_config["params"]["max_depth"]
            self.register_buffer("depth_predictor_normalizer_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.register_buffer("depth_predictor_normalizer_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
            self.depth_predictor = self.depth_predictor.eval()
            for n, p in self.depth_predictor.named_parameters():
                p.requires_grad = False
        else:
            self.depth_predictor = None

        self.epipolar_config = epipolar_config
        if self.epipolar_config is not None:
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
                if _module.context_dim is None and _module.attn1.to_k.in_features != self.model.diffusion_model.init_attn[
                    0].proj_in.out_channels:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

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

                        self.is_3d_full_attn = epipolar.is_3d_full_attn
                        _module.add_module('epipolar', epipolar)
                        # _module.add_module('norm_epipolar', nn.LayerNorm(_module.attn1.to_k.in_features))

    def get_traj_features(self, extra_cond):
        b, c, t, h, w = extra_cond.shape
        ## process in 2D manner
        extra_cond = rearrange(extra_cond, 'b c t h w -> (b t) c h w')
        traj_features = self.omcm(extra_cond)
        traj_features = [rearrange(feature, '(b t) c h w -> b c t h w', b=b, t=t) for feature in traj_features]
        return traj_features

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        weight_decay = self.weight_decay

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.pose_encoder_trainable:
            params_pose_encoder = [p for p in self.pose_encoder.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_pose_encoder)}] Paramters for pose_encoder.")
            params.extend(params_pose_encoder)

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        self.log_dict(loss_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    @torch.autocast(device_type="cuda", enabled=False)
    def ray_condition(self, K, c2w, H, W, device, flip_flag=None):
        # c2w: B, V, 4, 4
        # K: B, V, 3, 3

        def custom_meshgrid(*args):
            # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
            if pver.parse(torch.__version__) < pver.parse('1.10'):
                return torch.meshgrid(*args)
            else:
                return torch.meshgrid(*args, indexing='ij')

        B, V = K.shape[:2]

        j, i = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
        )
        i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
        j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

        n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
        if n_flip > 0:
            j_flip, i_flip = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
            )
            i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            i[:, flip_flag, ...] = i_flip
            j[:, flip_flag, ...] = j_flip

        fx = K[..., 0, 0].unsqueeze(-1)
        fy = K[..., 1, 1].unsqueeze(-1)
        cx = K[..., 0, 2].unsqueeze(-1)
        cy = K[..., 1, 2].unsqueeze(-1)

        zs = torch.ones_like(i)  # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
        rays_o = c2w[..., :3, 3]  # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
        # c2w @ dirctions
        rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
        # plucker = plucker.permute(0, 1, 4, 2, 3)
        plucker = rearrange(plucker, "b f h w c -> b c f h w")  # [b, 6, f, h, w]
        return plucker

    @torch.autocast(device_type="cuda", enabled=False)
    def get_relative_pose(self, RT_4x4: Tensor, cond_frame_index: Tensor, mode='left', normalize_T0=False):
        '''
        :param
            RT: (b,t,4,4)
            cond_frame_index: (b,)
        :return:
        '''
        b, t, _, _ = RT_4x4.shape  # b,t,4,4
        # cond_frame_index = cond_frame_index.view(b, 1, 1, 1).expand(-1, 1, 4, 4)
        # first_frame_RT = torch.gather(RT_4x4, dim=1, index=cond_frame_index)  # (b, 1, 4, 4)
        first_frame_RT = RT_4x4[torch.arange(b, device=RT_4x4.device), cond_frame_index, ...].unsqueeze(1)  # (b, 1, 4, 4)
        if normalize_T0:
            scale = first_frame_RT.reshape(b, -1).norm(p=2, dim=-1).reshape(b, 1, 1, 1)
            first_frame_RT = first_frame_RT / scale
            RT_4x4 = RT_4x4 / scale

        if mode == 'left':
            relative_RT_4x4 = first_frame_RT.inverse() @ RT_4x4
        elif mode == 'right':
            relative_RT_4x4 = RT_4x4 @ first_frame_RT.inverse()

        return relative_RT_4x4

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

        grid_y, grid_x = torch.meshgrid(y, x)  # H * W
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

    @torch.no_grad
    @torch.autocast(device_type="cuda", enabled=False)
    def get_sample_grid(self, F: Tensor, T: int, H: int, W: int, downsample: int):
        """
        modified to take in batch inputs

        Args:
            grid: (H*W, 3)
            F: camera fundamental matrix (B, 3, 3)
            resolution: feature map resolution H * W
            downsample: downsample scale

        return: weight matrix M(HW * HW)
        """
        B = F.shape[0]
        device = F.device

        y = torch.arange(0, H, dtype=torch.float, device=device)  # 0 .. 128
        x = torch.arange(0, W, dtype=torch.float, device=device)  # 0 .. 84

        y = pix2coord(y, downsample)  # H
        x = pix2coord(x, downsample)  # W

        grid_y, grid_x = torch.meshgrid(y, x)  # H * W
        # grid_y: 84x128
        # 3 x HW
        # TODO check whether yx or xy
        grid = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=2).view(-1, 3).float()  # H*W, 3

        pixel_coordinates = grid.unsqueeze(0).repeat(B, 1, 1)  # (B, H*W, 3)
        lines = F @ pixel_coordinates.permute(0, 2, 1)  # [B, 3, H*W]
        lines = lines / torch.norm(lines[:, :2, :], dim=1, keepdim=True)  # [B, 3, HW]
        lines = lines.permute(0, 2, 1)  # B, HW, 3

        epsilon = 1e-3
        EPS = torch.zeros(1, device=device)
        xmin, xmax = 0, W - 1
        ymin, ymax = 0, H - 1
        tmp_tensor = torch.tensor([True, True, False, False], device=device)
        outrange_tensor = torch.tensor(
            [xmin - 10000, ymin - 10000, xmin - 10000, ymin - 10000], dtype=lines.dtype, device=device
        ).view(2, 2)

        # Precompute constants for numerical stability
        sign_lines_1 = torch.sign(lines[..., 1])
        sign_lines_0 = torch.sign(lines[..., 0])
        max_abs_lines_1 = torch.max(torch.abs(lines[..., 1]), EPS)
        max_abs_lines_0 = torch.max(torch.abs(lines[..., 0]), EPS)

        # B x HW
        by1 = -(xmin * lines[..., 0] + lines[..., 2]) / (sign_lines_1 * max_abs_lines_1)
        by2 = -(xmax * lines[..., 0] + lines[..., 2]) / (sign_lines_1 * max_abs_lines_1)
        bx0 = -(ymin * lines[..., 1] + lines[..., 2]) / (sign_lines_0 * max_abs_lines_0)
        bx3 = -(ymax * lines[..., 1] + lines[..., 2]) / (sign_lines_0 * max_abs_lines_0)

        # B x HW x 4
        intersections = torch.stack((bx0, by1, by2, bx3), -1)

        # B x HW x 4 x 2
        intersections = intersections.view(B, H * W, 4, 1).repeat(1, 1, 1, 2)
        intersections[..., 0, 1] = ymin
        intersections[..., 1, 0] = xmin
        intersections[..., 2, 0] = xmax
        intersections[..., 3, 1] = ymax

        # B x HW x 4
        mask = torch.stack((
            (bx0 >= xmin + epsilon) & (bx0 < xmax - epsilon),
            (by1 > ymin + epsilon) & (by1 <= ymax - epsilon),
            (by2 >= ymin + epsilon) & (by2 < ymax - epsilon),
            (bx3 > xmin + epsilon) & (bx3 <= xmax - epsilon),
        ), -1)

        # B x HW
        Nintersections = mask.sum(-1)

        # rule out all lines have no intersections
        mask[Nintersections < 2] = tmp_tensor

        # B x HW x 2 x 2
        valid_intersections = intersections[mask].view(B, H * W, 2, 2)
        valid_intersections[Nintersections < 2] = outrange_tensor

        # sample_size x B x HW x 2
        sample_steps = torch.arange(0, self.sample_size, device=device).view(-1, 1, 1, 1) / (self.sample_size - 1)  # sample_size,B, H * W, 2
        sample_locs = torch.lerp(valid_intersections[..., 0, :], valid_intersections[..., 1, :], sample_steps)

        # normalize
        sample_locs = coord2pix(sample_locs, downsample)

        # sample_size*B x H x W x 2
        sample_locs = normalize(sample_locs, H, W)
        sample_locs = rearrange(sample_locs, 'S (B T1 T2) HW p -> (B T2) T1 (HW S) p', T1=T, T2=T)

        return sample_locs

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                        return_cond_frame_index=False, return_cond_frame=False, return_original_input=False, rand_cond_frame=None,
                        enable_camera_condition=True, return_camera_data=False, return_video_path=False, trace_scale_factor=1.0, cond_frame_index=None, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]

        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        if not hasattr(self, "null_prompt"):
            self.null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, self.null_prompt, cond_emb.detach())

        ## get conditioning frame
        if cond_frame_index is None:
            cond_frame_index = torch.zeros(batch_size, device=device, dtype=torch.long)
            rand_cond_frame = self.rand_cond_frame if rand_cond_frame is None else rand_cond_frame
            if rand_cond_frame:
                cond_frame_index = torch.randint(0, self.model.diffusion_model.temporal_length, (batch_size,), device=device)

        img = x[torch.arange(batch_size, device=device), :, cond_frame_index, ...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img)  ## b l c
        img_emb = self.image_proj_model(img_emb)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
                img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[torch.arange(batch_size, device=device), :, cond_frame_index, :, :]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond]  # b c t h w
            cond["c_cond_frame_index"] = cond_frame_index
            cond["origin_z_0"] = z.clone()
        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)]  ## concat in the seq_len dim

        ########################################### only change here, add RT input ###########################################
        if enable_camera_condition:
            with torch.no_grad():
                with torch.autocast('cuda', enabled=False):
                    camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
                    w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
                    c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
                    B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device
                    relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
                    relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

                    if self.epipolar_config is not None and not self.is_3d_full_attn:
                        relative_c2w_RT_4x4_pairs = self.get_relative_c2w_RT_pairs(relative_c2w_RT_4x4)  # b,t,t,4,4
                        R = relative_c2w_RT_4x4_pairs[..., :3, :3]  # b,t,t,3,3
                        t = relative_c2w_RT_4x4_pairs[..., :3, 3:4]  # b,t,t,3,1
                        K = camera_intrinsics_3x3.unsqueeze(1)
                        F = self.get_fundamental_matrix(K, R, t)
                        sample_locs_dict = {d: self.get_epipolar_mask(F, T, H // d, W // d, d) for d in [int(8 * ds) for ds in self.epipolar_config.attention_resolution]}
                    else:
                        sample_locs_dict = None

            if self.pose_encoder is not None:
                with torch.no_grad():
                    with torch.autocast('cuda', enabled=False):
                        pluker_embedding = self.ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # b, 6, t, H, W
                pluker_embedding_features = self.pose_encoder(pluker_embedding)  # bf c h w
                pluker_embedding_features = [rearrange(_, '(b f) c h w -> b c f h w', b=batch_size) for _ in pluker_embedding_features]
            else:
                pluker_embedding_features = None

            cond["camera_condition"] = {
                "pluker_embedding_features": pluker_embedding_features,
                "sample_locs_dict": sample_locs_dict,
                "cond_frame_index": cond_frame_index,
                "add_type": self.add_type
            }

        ########################################### only change here, add RT input ###########################################

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame_index:
            out.append(cond_frame_index)
        if return_cond_frame:
            out.append(x[torch.arange(batch_size, device=device), :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_camera_data:
            camera_data = batch.get('camera_data', None)
            out.append(camera_data)

        if return_video_path:
            out.append(batch['video_path'])

        return out

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, unconditional_guidance_scale=1.0, mask=None,
                   sampled_img_num=1, enable_camera_condition=True, trace_scale_factor=1.0,
                   cond_frame_index=None,
                   **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_frame_index, cond_x, x, camera_data, video_path = self.get_batch_input(batch, random_uncond=False,
                                                                                                        return_first_stage_outputs=True,
                                                                                                        return_original_cond=True,
                                                                                                        return_fs=True,
                                                                                                        return_cond_frame_index=True,
                                                                                                        return_cond_frame=True,
                                                                                                        rand_cond_frame=False,
                                                                                                        enable_camera_condition=enable_camera_condition,
                                                                                                        return_original_input=True,
                                                                                                        return_camera_data=True,
                                                                                                        return_video_path=True,
                                                                                                        trace_scale_factor=trace_scale_factor,
                                                                                                        cond_frame_index=cond_frame_index,
                                                                                                        )

        N = xrec.shape[0]
        log["camera_data"] = camera_data
        log["video_path"] = video_path
        log["gt_video"] = x
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)

                img = torch.zeros_like(xrec[:, :, 0])  ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img)  ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            if "preview" in kwargs:
                assert kwargs["preview"] is True
                device = z.device
                camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3

                w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
                c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
                B, T = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1]
                relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
                relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

                if 'per_frame_scale' in batch:
                    per_frame_scale = super().get_input(batch, 'per_frame_scale').float()  # b, t
                    depth_scale = per_frame_scale[torch.arange(B, device=device), cond_frame_index]  # b
                    relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * depth_scale[..., None, None]
                else:
                    depth_scale = torch.ones((B,), device=device)

                resized_H, resized_W = batch['resized_H'][0], batch['resized_W'][0]
                resized_H2, resized_W2 = batch['resized_H2'][0], batch['resized_W2'][0]
                points, colors, points_x, points_y = self.construct_3D_scene(
                    x, cond_frame_index, camera_intrinsics_3x3, device,
                    resized_H=resized_H, resized_W=resized_W
                )
                F, H, W = x.shape[-3:]

                result_dir = kwargs["result_dir"]
                preview_path = f"{result_dir}/preview_{uuid4()}.mp4"
                log["scene_frames_path"] = preview_path
                log["scene_points"] = points
                log["scene_colors"] = colors
                log["scene_points_x"] = points_x
                log["scene_points_y"] = points_y
                if os.path.exists(preview_path):
                    os.remove(preview_path)
                scene_frames = render(
                    (W, H),
                    camera_intrinsics_3x3[0, 0, :, :].cpu().numpy(),
                    relative_c2w_RT_4x4[0].inverse().cpu().numpy(),
                    points,
                    colors,
                )  # b, h, w, c
                print('scene_frames1', scene_frames.shape, (resized_H, resized_W), (H, W))
                scene_frames = transforms.CenterCrop((min(resized_H, H), min(resized_W, W)))(
                    rearrange(torch.from_numpy(scene_frames), 'b h w c -> b c h w')
                )
                scene_frames = torch.nn.functional.interpolate(
                    scene_frames,  # b c h w
                    (
                        constrain_to_multiple_of(scene_frames.shape[-2], multiple_of=16),
                        constrain_to_multiple_of(scene_frames.shape[-1], multiple_of=16),
                    ),
                    mode="bilinear", align_corners=True
                )
                scene_frames = rearrange(scene_frames, 'b c h w -> b h w c').numpy()
                print('scene_frames2', scene_frames.shape)

                print(scene_frames.shape, scene_frames.dtype, scene_frames.min(), scene_frames.max())
                imageio.mimsave(preview_path, scene_frames, fps=10)

                if 'paste_3d_scene' in kwargs and kwargs['paste_3d_scene']:
                    if "spatial_transform" in kwargs:
                        spatial_transform = kwargs.pop("spatial_transform")
                    else:
                        spatial_transform = make_spatial_transformations((H, W), type='resize_center_crop')
                    # video_reader = VideoReader(preview_path, ctx=cpu(0))
                    # scene_frames = video_reader.get_batch(list(range(F))).asnumpy()
                    scene_frames = rearrange(
                        transforms.CenterCrop((H, W))(
                            rearrange(torch.from_numpy(scene_frames), 'b h w c -> b c h w')
                        ),
                        'b c h w -> b h w c'
                    ).numpy()
                    scene_frames = torch.from_numpy(scene_frames).permute(3, 0, 1, 2).float().to(device)  # [t,h,w,c] -> [c,t,h,w]
                    scene_frames = spatial_transform(scene_frames)
                    scene_frames = (scene_frames / 255 - 0.5) * 2  # c,f,h,w

                    kwargs['paste_3d_scene_frames'] = self.encode_first_stage(scene_frames.unsqueeze(0)).clone()  # b=1,c,f,h,w
                    kwargs['paste_3d_scene_frames'][0, :, :1, :, :] = z[0, :, :1, :, :]
                    kwargs['paste_3d_scene_mask'] = torch.where(
                        (scene_frames - (-1.0)).abs() < 1e-4,
                        0.0,
                        1.0
                    ).amax(0)  # c,f,h,w --> f,h,w
                    kwargs['paste_3d_scene_mask'] = apply_thresholded_conv(kwargs['paste_3d_scene_mask'].unsqueeze(0), kernel_size=5, threshold=1.0).squeeze(0)
                    # kwargs['paste_3d_scene_mask'] = kwargs['paste_3d_scene_mask'] * kwargs['paste_3d_scene_mask'][cond_frame_index, :, :] # (f,h,w) * (1,h,w) --> (f,h,w)
                    kwargs['paste_3d_scene_mask'] = torch.nn.functional.interpolate(
                        kwargs['paste_3d_scene_mask'].unsqueeze(0),  # f,h,w --> 1,f,h,w
                        (H // 8, W // 8),
                        mode="bilinear", align_corners=True
                    ).unsqueeze(1)  # 1,1,f,h,w
                    kwargs['paste_3d_scene_mask'] = torch.where(
                        kwargs['paste_3d_scene_mask'] < 0.95,
                        0.0,
                        1.0
                    ).repeat(1, 4, 1, 1, 1).to(device)  # b=1,1,f,h,w --> b=1,4,f,h,w

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z,
                                                         enable_camera_condition=enable_camera_condition, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            if 'resized_H' in batch:
                F, H, W = x.shape[-3:]
                resized_H, resized_W = batch['resized_H'][0], batch['resized_W'][0]
                resized_H2, resized_W2 = batch['resized_H2'][0], batch['resized_W2'][0]
                # b, c, f, h, w
                crop_H, crop_W = min(min(resized_H, H), resized_H2), min(min(resized_W, W), resized_W2)
                log["samples"] = transforms.CenterCrop((crop_H, crop_W))(
                    rearrange(x_samples, '1 c f h w -> (1 c) f h w')
                )
                log["samples"] = torch.nn.functional.interpolate(
                    log["samples"],  # b c h w
                    (
                        constrain_to_multiple_of(crop_H, multiple_of=2),
                        constrain_to_multiple_of(crop_W, multiple_of=2),
                    ),
                    mode="bilinear", align_corners=True
                )
                log["samples"] = rearrange(
                    log["samples"],
                    '(1 c) f h w -> 1 c f h w'
                )
                print('crop valid part to', log['samples'].shape)

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def construct_3D_scene(self, video, cond_frame_index, camera_intrinsics, device,
                           is_remove_outliers=True, resized_H=None, resized_W=None):
        '''
        :param video: b=1,c,f,h,w;  value between [0,1]
        :param cond_frame_index: (b,)
        :param camera_intrinsics: (b,f,3,3)
        :return:
        '''
        B, C, F, H, W = video.shape
        cond_camera_intrinsics = camera_intrinsics[0, 0, :, :].cpu()  # b,f,3,3 --> 3x3
        fx, fy = cond_camera_intrinsics[0][0].item(), cond_camera_intrinsics[1][1].item()
        cx, cy = cond_camera_intrinsics[0][2].item(), cond_camera_intrinsics[1][2].item()
        if resized_H is not None and resized_W is not None:
            H, W = min(resized_H, H), min(resized_W, W)
            cx, cy = W // 2, H // 2
            video = transforms.CenterCrop((H, W))(
                rearrange(video, 'b c f h w -> (b c) f h w')
            )
            video = rearrange(video, '(b c) f h w -> b c f h w', b=B, c=C)
            print(H, W, video.shape)

        x, y = np.meshgrid(np.arange(W), np.arange(H))
        x = (x - cx) / fx
        y = (y - cy) / fy

        if hasattr(self, "depth_predictor") and self.depth_predictor is not None:
            color_image = torch.nn.functional.interpolate(
                video[torch.arange(B, device=device), :, cond_frame_index, ...] / 2 + 0.5,  # (b, c, H, W), [-1,1] --> [0,1]
                (constrain_to_multiple_of(H, multiple_of=14), constrain_to_multiple_of(W, multiple_of=14)),
                mode="bilinear", align_corners=True
            )  # b,3,H,W;  value from 0 to 1
            color_image = (color_image - self.depth_predictor_normalizer_mean) / self.depth_predictor_normalizer_std  # b,3,H,W;  value from -1 to 1
            depth_map = self.depth_predictor(color_image) / self.max_depth  # x:b,3,H,W --> b,H,W, value from 0 to 1
            depth_map = (depth_map - 0.5) * 2  # [-1,1]
            depth_map = torch.nn.functional.interpolate(depth_map[:, None], (H, W), mode="bilinear", align_corners=True)  # b,1,H//8,W///8
            depth_map = (
                    (depth_map[0] + 1).cpu() / 2 * self.max_depth
            )  # 1,h,w;  value in [0,1] --> [0, max_depth]
            color_image = video[0, :, 0, :, :]  # B=1 x c x 1 x h x w --> c x h x w
            color_image = color_image.permute(1, 2, 0).cpu().numpy() / 2 + 0.5  # h x w x c; value between [0,1]

            z = np.array(depth_map)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
            colors = color_image.reshape(-1, 3)  # [0,1]
        else:
            points = np.array([[0, 0, 0]]).reshape(-1, 3)
            colors = np.array([[0, 0, 0]]).reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if is_remove_outliers:
            pcd = remove_outliers(pcd)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(type(points), points.shape, colors.shape)
        return points, colors, x, y  # N,3;  N,3