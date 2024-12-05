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
        if spatial_transform_type is not None:
            self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type)
        else:
            self.spatial_transform = None

    def get_batch_input(self, ref_img, caption, camera_pose_3x4, frame_stride=1, fps=29.97, ref_img2=None):
        '''
        :param ref_img: (h, w, c), tensor
        :param camera pose: (t,3,4), tensor
        :param caption: str
        :return:
        '''

        # ref_img = rearrange(torch.from_numpy(np.array(Image.open(ref_img_path).convert("RGB"))), "h w c -> c 1 h w")
        ref_img = rearrange(torch.from_numpy(ref_img), 'h w c -> c 1 h w').to(self.device)
        if ref_img2 is not None:
            ref_img2 = rearrange(torch.from_numpy(ref_img2), 'h w c -> c 1 h w').to(self.device)
        ori_H, ori_W = ref_img.shape[-2:]
        resized_ref_img = self.spatial_transform.transforms[0](ref_img)
        resized_H, resized_W = resized_ref_img.shape[-2:]
        ref_img = self.spatial_transform.transforms[1](resized_ref_img)
        ref_img = (ref_img / 255 - 0.5) * 2
        ref_img = repeat(ref_img, "c 1 h w -> c f h w", f=self.video_length).clone()
        if ref_img2 is not None:
            resized_ref_img2 = self.spatial_transform.transforms[0](ref_img2)
            resized_H2, resized_W2 = resized_ref_img2.shape[-2:]
            ref_img2 = self.spatial_transform.transforms[1](resized_ref_img2)
            ref_img2 = (ref_img2 / 255 - 0.5) * 2
            ref_img[:,-1:,:,:]  = ref_img2.clone()
            print((ref_img[:,0,:,:] == ref_img[:,-1,:,:]).all())
        else:
            resized_W2, resized_H2 = resized_W, resized_H

        camera_pose_4x4 = rt34_to_44(camera_pose_3x4).to(device=self.device)  # (t,3,4) --> (t,4,4)

        fx = 0.5 * max(ori_H, ori_W) / ori_W
        fy = 0.5 * max(ori_H, ori_W) / ori_H
        cx = 0.5
        cy = 0.5

        if self.spatial_transform_type == 'resize_center_crop':
            sample_H, sample_W = self.resolution[0], self.resolution[1]
            if sample_H <= sample_W:
                scale = sample_H / ori_H
            else:
                scale = sample_W / ori_W
            fx *= ori_W * scale
            fy *= ori_H * scale
            cx *= sample_W
            cy *= sample_H

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

