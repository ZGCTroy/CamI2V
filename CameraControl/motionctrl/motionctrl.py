import logging
import os
import pdb
import random
from math import sqrt
from uuid import uuid4

import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from einops import rearrange, repeat
from packaging import version as pver
from torch import Tensor, nn

from CameraControl.data.realestate10k import make_spatial_transformations
from CameraControl.data.utils import apply_thresholded_conv, constrain_to_multiple_of
from CameraControl.motionctrl.motionctrl_modified_modules import (
    new__forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_BasicTransformerBlock_of_TemporalTransformer,
    new_forward_for_TemporalTransformer,
    new_forward_for_TimestepEmbedSequential,
    new_forward_for_unet,
)
from lvdm.models.ddpm3d import LatentVisualDiffusion
from scripts.gradio.preview import render
from utils.utils import instantiate_from_config

mainlogger = logging.getLogger('mainlogger')



class MotionCtrl(LatentVisualDiffusion):
    def __init__(self,
                 pose_dim=12,
                 context_dim=1024,
                 # cc_projection_list_trainable=False,
                 diffusion_model_trainable_param_list=[],
                 depth_predictor_config=None,
                 normalize_T0=False,
                 weight_decay=1e-2,
                 continuation_mode=False,
                 *args,
                 **kwargs):
        super(MotionCtrl, self).__init__(*args, **kwargs)
        self.weight_decay = weight_decay
        self.normalize_T0 = normalize_T0
        self.diffusion_model_trainable_param_list = diffusion_model_trainable_param_list
        for p in self.model.parameters():
            p.requires_grad = False

        if 'TemporalTransformer.attn2' in self.diffusion_model_trainable_param_list:
            for n, m in self.model.named_modules():
                if m.__class__.__name__ == 'BasicTransformerBlock' and m.attn2.context_dim is None:       # attn2 of TemporalTransformer BasicBlocks
                    for p in m.attn2.parameters():
                        p.requires_grad = True

        self.depth_predictor_config = depth_predictor_config
        if depth_predictor_config is not None:
            self.depth_predictor = instantiate_from_config(depth_predictor_config)
            self.depth_predictor.load_state_dict(torch.load(depth_predictor_config["pretrained_model_path"], map_location='cpu', weights_only=True))
            self.max_depth = depth_predictor_config["params"]["max_depth"]
            self.register_buffer("depth_predictor_normalizer_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1))
            self.register_buffer("depth_predictor_normalizer_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
            self.depth_predictor = self.depth_predictor.eval()
            for n, p in self.depth_predictor.named_parameters():
                p.requires_grad = False
        else:
            self.depth_predictor = None

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
                if _module.context_dim is None:  # BasicTransformerBlock of TemporalTransformer, only self attn, context_dim=None

                    bound_method = new_forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, 'forward', bound_method)

                    bound_method = new__forward_for_BasicTransformerBlock_of_TemporalTransformer.__get__(_module, _module.__class__)
                    setattr(_module, '_forward', bound_method)

                    cc_projection = nn.Linear(_module.attn2.to_k.in_features + pose_dim, _module.attn2.to_k.in_features)
                    nn.init.zeros_(list(cc_projection.parameters())[0])
                    nn.init.eye_(list(cc_projection.parameters())[0][:_module.attn2.to_k.in_features, :_module.attn2.to_k.in_features])
                    nn.init.zeros_(list(cc_projection.parameters())[1])
                    cc_projection.requires_grad_(True)

                    _module.add_module('cc_projection', cc_projection)

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        weight_decay = self.weight_decay

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

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
            scale = first_frame_RT.reshape(b,-1).norm(p=2, dim=-1).reshape(b,1,1,1)
            first_frame_RT = first_frame_RT / scale
            RT_4x4 = RT_4x4 / scale

        if mode == 'left':
            relative_RT_4x4 = first_frame_RT.inverse() @ RT_4x4
        elif mode == 'right':
            relative_RT_4x4 = RT_4x4 @ first_frame_RT.inverse()

        return relative_RT_4x4


    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                        return_cond_frame_index=False, return_depth_scale=False,
                        return_cond_frame=False, return_original_input=False, rand_cond_frame=None, enable_camera_condition=True,
                        return_camera_data=False, return_video_path=False,
                        trace_scale_factor=1.0, cond_frame_index=None, **kwargs):
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

        ########################################### only change here, add camera_condition input ###########################################
        depth_scale = torch.ones((batch_size,), device=device)
        if enable_camera_condition:
            with torch.no_grad():
                with torch.autocast('cuda', enabled=False):
                    camera_intrinsics_3x3 = super().get_input(batch, 'camera_intrinsics').float()  # b, t, 3, 3
                    w2c_RT_4x4 = super().get_input(batch, 'RT').float()  # b, t, 4, 4
                    c2w_RT_4x4 = w2c_RT_4x4.inverse()   # w2c --> c2w
                    B, T, device = c2w_RT_4x4.shape[0], c2w_RT_4x4.shape[1], c2w_RT_4x4.device

                    relative_c2w_RT_4x4 = self.get_relative_pose(c2w_RT_4x4, cond_frame_index, mode='left', normalize_T0=self.normalize_T0)  # b,t,4,4
                    relative_c2w_RT_4x4[:,:,:3,3] = relative_c2w_RT_4x4[:,:,:3,3] * trace_scale_factor

                    if 'per_frame_scale' in batch:
                        per_frame_scale = super().get_input(batch, 'per_frame_scale').float()  # b, t
                        depth_scale = per_frame_scale[torch.arange(B, device=device), cond_frame_index] # b
                        relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * depth_scale[..., None, None]

                    relative_w2c_RT_4x4 = relative_c2w_RT_4x4.inverse()

            cond["camera_condition"] = {
                "RT": rearrange(relative_w2c_RT_4x4[:,:,:3,:4], 'b t x y -> b t (x y)'),
            }

        ########################################### only change here, add camera_condition input ###########################################

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
        if return_depth_scale:
            out.append(depth_scale)
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
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, unconditional_guidance_scale=1.0, mask=None, sampled_img_num=1, enable_camera_condition=True,
    trace_scale_factor=1.0, cond_frame_index=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_frame_index, depth_scale, cond_x, x, camera_data, video_path = self.get_batch_input(batch, random_uncond=False,
                                                          return_first_stage_outputs=True,
                                                          return_original_cond=True,
                                                          return_fs=True,
            return_cond_frame_index=True,
            return_depth_scale=True,
                                                          return_cond_frame=True,
                                                          rand_cond_frame=False,
                                                          enable_camera_condition=enable_camera_condition,
                                                          return_original_input=True,
                                                          return_camera_data=True,
            return_video_path=True,
            trace_scale_factor=trace_scale_factor,
            cond_frame_index=cond_frame_index
                                                          )

        N = xrec.shape[0]
        log["depth_scale"] = depth_scale
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
                kwargs.pop("preview")
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

                points, colors, points_x, points_y = self.construct_3D_scene(x, cond_frame_index, camera_intrinsics_3x3, device)
                F, H, W = x.shape[-3:]

                preview_path = f"/tmp/realcam-i2v_preview_{uuid4().fields[0]:x}.mp4"
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
                )
                print(scene_frames.shape, scene_frames.dtype, scene_frames.min(), scene_frames.max())
                # print(depth_frames.shape, depth_frames.dtype, depth_frames.min(), depth_frames.max())
                imageio.mimsave(preview_path, scene_frames, fps=8)
                # os.system(f"python scripts/gradio/preview.py {data_tmp_path} --output_path {preview_path}")

                if 'paste_3d_scene' in kwargs and kwargs['paste_3d_scene']:
                    if "spatial_transform" in kwargs:
                        spatial_transform = kwargs.pop("spatial_transform")
                    else:
                        spatial_transform = make_spatial_transformations((H,W), type='resize_center_crop')
                    # video_reader = VideoReader(preview_path, ctx=cpu(0))
                    # scene_frames = video_reader.get_batch(list(range(F))).asnumpy()
                    scene_frames = torch.from_numpy(scene_frames).permute(3, 0, 1, 2).float().to(device)  # [t,h,w,c] -> [c,t,h,w]
                    scene_frames = spatial_transform(scene_frames)
                    scene_frames = (scene_frames / 255 - 0.5) * 2   # c,f,h,w

                    kwargs['paste_3d_scene_frames'] = self.encode_first_stage(scene_frames.unsqueeze(0)).clone() # b=1,c,f,h,w
                    kwargs['paste_3d_scene_frames'][0,:,:1,:,:] = z[0,:,:1,:,:]
                    kwargs['paste_3d_scene_mask'] = torch.where(
                        (scene_frames - (-1.0)).abs() < 1e-4,
                        0.0,
                        1.0
                    ).amax(0)  # c,f,h,w --> f,h,w
                    kwargs['paste_3d_scene_mask'] = apply_thresholded_conv(kwargs['paste_3d_scene_mask'].unsqueeze(0), kernel_size=5, threshold=1.0).squeeze(0)
                    kwargs['paste_3d_scene_mask'] = kwargs['paste_3d_scene_mask'] * kwargs['paste_3d_scene_mask'][cond_frame_index, :, :] # (f,h,w) * (1,h,w) --> (f,h,w)
                    kwargs['paste_3d_scene_mask'] = torch.nn.functional.interpolate(
                        kwargs['paste_3d_scene_mask'].unsqueeze(0), # f,h,w --> 1,f,h,w
                        (H // 8, W // 8),
                        mode="bilinear", align_corners=True
                    ).unsqueeze(1) # 1,1,f,h,w
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

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def construct_3D_scene(self, video, cond_frame_index, camera_intrinsics, device):
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
            points = np.array([[0,0,0]]).reshape(-1, 3)
            colors = np.array([[0, 0, 0]]).reshape(-1, 3)

        return points, colors, x, y  # N,3;  N,3