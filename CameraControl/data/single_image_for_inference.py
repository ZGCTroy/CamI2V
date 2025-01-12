import os
import sys

import torch
from torch import Tensor
from einops import rearrange, repeat
from torchvision import transforms
import pdb

def rt34_to_44(rt: Tensor) -> Tensor:
    return torch.cat([rt, torch.FloatTensor([[[0, 0, 0, 1]]] * rt.size(0))], dim=1)


def make_spatial_transformations(resolution, type):
    """
    resolution: target resolution, a list of int, [h, w]
    """
    if type == "random_crop":
        transformations = transforms.RandomCropss(resolution)
    elif type == "resize_center_crop":
        is_square = (resolution[0] == resolution[1])
        if is_square:
            transformations = transforms.Compose([
                transforms.Resize(resolution[0], antialias=True),
                transforms.CenterCrop(resolution[0]),
            ])
        else:
            transformations = transforms.Compose([
                transforms.Resize(min(resolution)),
                transforms.CenterCrop(resolution),
            ])
    else:
        raise NotImplementedError
    return transformations


class SingleImageForInference():
    def __init__(self,
                 video_length=16,
                 resolution=[256, 256],
                 spatial_transform_type=None,
                 device='cpu'
                 ):

        self.video_length = video_length
        self.resolution = resolution
        self.device= device
        self.spatial_transform_type = spatial_transform_type
        assert self.spatial_transform_type in ['resize_center_crop']

    def _resize_for_rectangle_crop(self, frames, H, W):
        '''
        :param frames: C,F,H,W
        :param image_size: H,W
        :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
        '''
        ori_H, ori_W = frames.shape[-2:]

        if ori_W / ori_H < 1.0:
            tmp_H, tmp_W = int(H), int(W)
            H, W = tmp_W, tmp_H

        if ori_W / ori_H > W / H:
            frames = transforms.functional.resize(
                frames,
                size=[H, int(ori_W * H / ori_H)],
            )
        else:
            frames = transforms.functional.resize(
                frames,
                size=[int(ori_H * W / ori_W), W],
            )

        resized_H, resized_W = frames.shape[2], frames.shape[3]
        frames = frames.squeeze(0)

        delta_H = resized_H - H
        delta_W = resized_W - W

        top, left = delta_H // 2, delta_W // 2
        frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

        return frames, resized_H, resized_W

    def get_batch_input(self, ref_img, caption, camera_pose_3x4, frame_stride=1, fps=29.97, ref_img2=None):
        '''
        :param ref_img: (h, w, c), tensor
        :param camera pose: (t,3,4), tensor
        :param caption: str
        :return:
        '''

        ref_img = rearrange(torch.from_numpy(ref_img), 'h w c -> c 1 h w').to(self.device)
        ref_img, resized_H, resized_W = self._resize_for_rectangle_crop(
            ref_img,
            self.resolution[0], self.resolution[1],  # H, W
        )
        ref_img = (ref_img / 255 - 0.5) * 2
        ref_img = repeat(ref_img, "c 1 h w -> c f h w", f=self.video_length).clone()

        if ref_img2 is not None:
            ref_img2 = rearrange(torch.from_numpy(ref_img2), 'h w c -> c 1 h w').to(self.device)
            ref_img2, resized_H2, resized_W2 = self._resize_for_rectangle_crop(
                ref_img2,
                self.resolution[0], self.resolution[1],  # H, W
            )
            ref_img2 = (ref_img2 / 255 - 0.5) * 2
            ref_img[:, -1:, :, :] = ref_img2.clone()
        else:
            resized_H2, resized_W2 = resized_H, resized_W

        camera_pose_4x4 = rt34_to_44(camera_pose_3x4).to(device=self.device)  # (t,3,4) --> (t,4,4)

        fx = 0.5 * max(resized_H, resized_W)
        fy = fx
        cx = 0.5 * ref_img.shape[-1]
        cy = 0.5 * ref_img.shape[-2]
        camera_intrinsics = torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1.0], device=self.device).reshape(1, 1, 3, 3).repeat(1, self.video_length, 1, 1)

        data = {
            'video': ref_img.unsqueeze(0),  # B=1 x c x f x h x w
            'caption': [caption],
            'video_path': [''],
            'fps': torch.tensor([fps // frame_stride], device=self.device), # B=1
            'frame_stride': torch.tensor([frame_stride], device=self.device),   # B=1
            'RT': camera_pose_4x4.unsqueeze(0),  # B=1 x T x 4 x 4
            'camera_intrinsics': camera_intrinsics,  # B=1 x T x 3 x 3
            'resized_W': [resized_W],
            'resized_H': [resized_H],
            'resized_W2': [resized_W2],
            'resized_H2': [resized_H2],
        }
        return data

