import logging

import torch
from einops import rearrange, repeat
from packaging import version as pver
from torch import Tensor

from CameraControl.dynamicrafter.dynamicrafter import DynamiCrafter
from utils.utils import instantiate_from_config

mainlogger = logging.getLogger('mainlogger')


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


class CameraControlLVDM(DynamiCrafter):
    def __init__(self,
                 diffusion_model_trainable_param_list=[],
                 pose_encoder_trainable=True,
                 pose_encoder_config=None,
                 normalize_T0=False,
                 weight_decay=1e-2,
                 *args,
                 **kwargs):
        super(CameraControlLVDM, self).__init__(*args, **kwargs)
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

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate
        weight_decay = self.weight_decay

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.pose_encoder is not None and self.pose_encoder_trainable:
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
    @torch.autocast(device_type="cuda", enabled=False)
    def ray_condition(self, K, c2w, H, W, device, flip_flag=None):
        # c2w: B, V, 4, 4
        # K: B, V, 3, 3

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
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, V, HW, 3
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

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False,
                        return_cond_frame_index=False, return_cond_frame=False, return_original_input=False, rand_cond_frame=None,
                        enable_camera_condition=True, return_camera_data=False, return_video_path=False, return_depth_scale=False,
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
            camera_condition_log, camera_condition_kwargs = self.get_batch_input_camera_condition_process(
                batch, x, cond_frame_index, trace_scale_factor, rand_cond_frame
            )
            if "depth_scale" in camera_condition_log:
                depth_scale = camera_condition_log["depth_scale"]
            cond.update(camera_condition_kwargs)
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
        if return_cond_frame:
            out.append(x[torch.arange(batch_size, device=device), :, cond_frame_index, ...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_camera_data:
            camera_data = batch.get('camera_data', None)
            out.append(camera_data)

        if return_video_path:
            out.append(batch['video_path'])

        if return_depth_scale:
            out.append(depth_scale)

        return out

    @torch.no_grad()
    def log_images(
        self,
        batch,
        sample=True,
        ddim_steps=50,
        ddim_eta=1.0,
        plot_denoise_rows=False,
        unconditional_guidance_scale=1.0,
        mask=None,
        sampled_img_num=1,
        enable_camera_condition=True,
        trace_scale_factor=1.0,
        cond_frame_index=None,
        **kwargs,
    ):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_frame_index, cond_x, x, camera_data, video_path, depth_scale = self.get_batch_input(
            batch,
            random_uncond=False,
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
            return_depth_scale=True,
            trace_scale_factor=trace_scale_factor,
            cond_frame_index=cond_frame_index,
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
                elif self.uncond_type == "negative_prompt":
                    prompts = N * [kwargs["negative_prompt"]]
                    uc_prompt = self.get_learned_conditioning(prompts)

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

            pre_process_log, pre_process_kwargs = self.log_images_sample_log_pre_process(
                batch, z, x, cond_frame_index, trace_scale_factor, **kwargs
            )
            log.update(pre_process_log)
            kwargs.update(pre_process_kwargs)

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z,
                                                         enable_camera_condition=enable_camera_condition, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples

            log.update(self.log_images_sample_log_post_process(x_samples, **pre_process_log))

            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def get_batch_input_camera_condition_process(self, *args, **kwargs):
        return {}, {}

    def log_images_sample_log_pre_process(self, *args, **kwargs):
        return {}, {}

    def log_images_sample_log_post_process(self, *args, **kwargs):
        return {}
