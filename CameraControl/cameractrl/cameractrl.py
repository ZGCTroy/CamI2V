import logging

import torch
from einops import rearrange, repeat
from torch import nn

from CameraControl.base.base import CameraControlLVDM
from CameraControl.cameractrl.cameractrl_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
)

mainlogger = logging.getLogger('mainlogger')


class CameraCtrl(CameraControlLVDM):
    def __init__(self, *args, **kwargs):
        super(CameraCtrl, self).__init__(*args, **kwargs)

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

                    cc_projection = nn.Linear(_module.attn1.to_k.in_features, _module.attn1.to_k.in_features)
                    nn.init.zeros_(list(cc_projection.parameters())[0])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)

    def get_batch_input_camera_condition_process(self, batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame, *args, **kwargs):
        return_log = {}
        return_kwargs = {}

        batch_size, num_frames, device, H, W = x.shape[0], x.shape[2], self.model.device, x.shape[3], x.shape[4]
        with torch.no_grad(),  torch.autocast('cuda', enabled=False):
            camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
            w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
            c2w_RT_4x4 = w2c_RT_4x4.inverse()  # w2c --> c2w
            B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

            relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
            relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * trace_scale_factor

        if self.pose_encoder is not None:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                pluker_embedding = self.ray_condition(camera_intrinsics_3x3, relative_c2w_RT_4x4, H, W, device, flip_flag=None)  # b, 6, t, H, W

            pluker_embedding_features = self.pose_encoder(pluker_embedding)  # bf c h w
            pluker_embedding_features = [rearrange(_, '(b f) c h w -> b c f h w', b=batch_size) for _ in pluker_embedding_features]
        else:
            pluker_embedding_features = None

        return_kwargs["camera_condition"] = {
                "pluker_embedding_features": pluker_embedding_features,
            }

        return return_log, return_kwargs
