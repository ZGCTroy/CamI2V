import logging
import os
from uuid import uuid4

import imageio
import numpy as np
import open3d as o3d
import torch
from einops import rearrange

from CameraControl.CamI2V.cami2v import CamI2V
from CameraControl.data.utils import apply_thresholded_conv, constrain_to_multiple_of, remove_outliers
from demo.preview import render
from utils.utils import instantiate_from_config

mainlogger = logging.getLogger('mainlogger')


class RealCam_I2V(CamI2V):
    def __init__(self, depth_predictor_config=None, continuation_mode=False, *args, **kwargs):
        super(RealCam_I2V, self).__init__(*args, **kwargs)
        self.continuation_mode = continuation_mode

        self.depth_predictor_config = depth_predictor_config
        if depth_predictor_config is not None:
            self.depth_predictor = instantiate_from_config(depth_predictor_config)
            self.depth_predictor.load_state_dict(torch.load(depth_predictor_config["pretrained_model_path"], map_location='cpu', weights_only=True))
            self.register_buffer("depth_predictor_normalizer_mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
            self.register_buffer("depth_predictor_normalizer_std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
            self.depth_predictor = self.depth_predictor.eval()
            for n, p in self.depth_predictor.named_parameters():
                p.requires_grad = False
        else:
            self.depth_predictor = None

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
            relative_c2w_RT_4x4[:,:,:3,3] = relative_c2w_RT_4x4[:,:,:3,3] * trace_scale_factor

            if 'per_frame_scale' in batch:
                per_frame_scale = super().get_input(batch, 'per_frame_scale').float()  # b, t
                depth_scale = per_frame_scale[torch.arange(B, device=device), cond_frame_index] # b
                return_log["depth_scale"] = depth_scale
                relative_c2w_RT_4x4[:, :, :3, 3] = relative_c2w_RT_4x4[:, :, :3, 3] * depth_scale[..., None, None]

            if self.epipolar_config is not None and not self.epipolar_config.is_3d_full_attn:
                relative_c2w_RT_4x4_pairs = self.get_relative_c2w_RT_pairs(relative_c2w_RT_4x4)  # b,t,t,4,4
                R = relative_c2w_RT_4x4_pairs[..., :3, :3] # b,t,t,3,3
                t = relative_c2w_RT_4x4_pairs[..., :3, 3:4] # b,t,t,3,1

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

        return_kwargs["camera_condition"]= {
            "pluker_embedding_features": pluker_embedding_features,
            "sample_locs_dict": sample_locs_dict,
            "cond_frame_index": cond_frame_index,
            "add_type": self.add_type,
        }

        return return_log, return_kwargs

    def log_images_sample_log_pre_process(self, batch, z, x, cond_frame_index, trace_scale_factor, **kwargs):
        return_log = {}
        return_kwargs = {}

        if kwargs.get("preview", False):
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
            return_log["H"] = H
            return_log["W"] = W
            return_log["resized_H"] = resized_H
            return_log["resized_W"] = resized_W
            return_log["resized_H2"] = resized_H2
            return_log["resized_W2"] = resized_W2

            result_dir = kwargs["result_dir"]
            preview_path = f"{result_dir}/preview_{uuid4().fields[0]:x}.mp4"
            return_log["scene_frames_path"] = preview_path
            return_log["scene_points"] = points
            return_log["scene_colors"] = colors
            return_log["scene_points_x"] = points_x
            return_log["scene_points_y"] = points_y
            if os.path.exists(preview_path):
                os.remove(preview_path)
            scene_frames = render(
                (W, H),
                camera_intrinsics_3x3[0, 0, :, :].cpu().numpy().astype(np.float64),
                relative_c2w_RT_4x4[0].inverse().cpu().numpy().astype(np.float64),
                points.astype(np.float64),
                colors.astype(np.float64),
            )  # b, h, w, c
            imageio.mimsave(preview_path, scene_frames, fps=10)

            if kwargs.get("noise_shaping", False):
                scene_frames = torch.from_numpy(scene_frames).permute(3, 0, 1, 2).float().to(device)  # [t,h,w,c] -> [c,t,h,w]
                scene_frames = (scene_frames / 255 - 0.5) * 2  # c,f,h,w

                return_kwargs['scene_frames'] = self.encode_first_stage(scene_frames.unsqueeze(0)).clone()  # b=1,c,f,h,w
                return_kwargs['scene_frames'][0, :, :1, :, :] = z[0, :, :1, :, :]
                return_kwargs['scene_mask'] = torch.where(
                    (scene_frames - (-1.0)).abs() < 1e-4,
                    0.0,
                    1.0
                ).amax(0)  # c,f,h,w --> f,h,w
                return_kwargs['scene_mask'] = apply_thresholded_conv(return_kwargs['scene_mask'].unsqueeze(0), kernel_size=5, threshold=1.0).squeeze(0)
                return_kwargs['scene_mask'] = torch.nn.functional.interpolate(
                    return_kwargs['scene_mask'].unsqueeze(0),  # f,h,w --> 1,f,h,w
                    (H // 8, W // 8),
                    mode="bilinear", align_corners=True
                ).unsqueeze(1)  # 1,1,f,h,w
                return_kwargs['scene_mask'] = torch.where(
                    return_kwargs['scene_mask'] < 0.9,
                    0.0,
                    1.0
                ).repeat(1, 4, 1, 1, 1).to(device)  # b=1,1,f,h,w --> b=1,4,f,h,w
                # return_kwargs['scene_mask'] = return_kwargs['scene_mask'].clone().repeat(1, 4, 1, 1, 1).to(device)  # b=1,1,f,h,w --> b=1,4,f,h,w

        return return_log, return_kwargs

    def log_images_sample_log_post_process(self, x_samples,  **kwargs):
        return_log = {}
        return return_log

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
        # if resized_H is not None and resized_W is not None:
        #     H, W = min(resized_H, H), min(resized_W, W)
        #     cx, cy = W // 2, H // 2
        #     video = transforms.CenterCrop((H, W))(
        #         rearrange(video, 'b c f h w -> (b c) f h w')
        #     )
        #     video = rearrange(video, '(b c) f h w -> b c f h w', b=B, c=C)
        #     print(H, W, video.shape)

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
            depth_map = self.depth_predictor(color_image) / self.depth_predictor.max_depth  # x:b,3,H,W --> b,H,W, value from 0 to 1
            depth_map = (depth_map - 0.5) * 2  # [-1,1]
            depth_map = torch.nn.functional.interpolate(depth_map[:, None], (H, W), mode="bilinear", align_corners=True)  # b,1,H//8,W///8
            depth_map = (
                    (depth_map[0] + 1).cpu() / 2 * self.depth_predictor.max_depth
            )  # 1,h,w;  value in [0,1] --> [0, max_depth]
            color_image = video[0, :, 0, :, :]  # B=1 x c x 1 x h x w --> c x h x w
            color_image = color_image.permute(1, 2, 0).cpu().numpy() / 2 + 0.5  # h x w x c; value between [0,1]

            z = np.array(depth_map)
            points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3).astype(np.float64)
            colors = color_image.reshape(-1, 3)  # [0,1]
            colors = colors.astype(np.float64)
        else:
            points = np.array([[0, 0, 0]]).reshape(-1, 3).astype(np.float64)
            colors = np.array([[0, 0, 0]]).reshape(-1, 3).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        if is_remove_outliers:
            pcd = remove_outliers(pcd)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        print(type(points), points.shape, colors.shape)
        return points, colors, x, y  # N,3;  N,3
