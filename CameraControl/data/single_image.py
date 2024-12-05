import json
import os
import pdb
import random
import sys

import cv2
import numpy as np
import omegaconf
import torch
from torch import Tensor
from einops import rearrange, repeat
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def rt34_to_44(rt: Tensor) -> Tensor:
    return torch.cat([rt, torch.FloatTensor([[[0, 0, 0, 1]]] * rt.size(0))], dim=1)

def make_spatial_transformations(resolution, type, ori_resolution=None):
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


class SingleImage(Dataset):
    """
    RealEstate10K Dataset.
    For each video, its meta info is stored in a txt file whose contents are as follows:
    line 0: video_url
    line 1: empty
    line 2: caption

    In the rest, each line is a frame, including frame path, 4 camera intrinsics, and 3*4 camera pose (the matrix is row-major order).

    e.g.
    line 3: 0_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    line 4: 1_frame_path focal_length_x focal_length_y principal_point_x principal_point_y 3*4_extrinsic_matrix
    ...

    meta_path: path to the meta file
    meat_list: path to the meta list file
    data_dir: path to the data folder
    video_length: length of the video clip for training
    resolution: target resolution, a list of int, [h, w]
    frame_stride: stride between frames, int or list of int, [min, max], do not larger than 32 when video_length=16
    spatial_transform: spatial transformation, ["random_crop", "resize_center_crop"]

    """

    def __init__(self,
                 metadata_json,
                 camera_pose_path,
                 camera_pose_sections,
                 video_length=16,
                 fps=24,
                 resolution=[256, 256],
                 frame_stride=1,  # [min, max], do not larger than 32 when video_length=16
                 spatial_transform=None,
                 metadata=None
                 ):
        if metadata is None:
            with open(metadata_json, 'r') as f:
                self.metadata = list(json.load(f).items())
        else:
            self.metadata = metadata

        # self.camera_data = [torch.from_numpy(np.loadtxt(x, comments="https")) for x in camera_pose_paths]
        self.camera_data = torch.from_numpy(np.loadtxt(camera_pose_path, comments="https"))
        self.camera_pose_sections = camera_pose_sections
        
        print(dict(self.metadata), self.camera_data.shape)
        
        self.fps = fps
        self.video_length = video_length
        self.resolution = (resolution, resolution) if isinstance(resolution, int) else resolution
        self.frame_stride = frame_stride
        self.spatial_transform_type = spatial_transform

        # make saptial transformations
        self.spatial_transform = None
        if isinstance(self.resolution[0], int):
            self.spatial_transform = make_spatial_transformations(self.resolution, type=self.spatial_transform_type) \
                if self.spatial_transform_type is not None else None

    def __getitem__(self, index):
        ## get frames until success
        index = index % len(self.metadata)
        ref_img_path, caption = self.metadata[index]
        ref_img = rearrange(torch.from_numpy(np.array(Image.open(ref_img_path).convert("RGB"))), "h w c -> c 1 h w")
        ori_H, ori_W = ref_img.shape[-2:]

        ref_img = self.spatial_transform(ref_img)
        ref_img = (ref_img / 255 - 0.5) * 2
        ref_img = repeat(ref_img, "c 1 h w -> c f h w", f=self.video_length)
        
        last = 0
        RT_list, camera_intrinsics_list, camera_data_list = [], [], []
        for _ in range(self.camera_pose_sections):
            frame_indices = range(self.video_length)
            camera_data = self.camera_data[last:][frame_indices]
            last += self.video_length

            camera_data_list.append(camera_data)
            
            # fx, fy, cx, cy = camera_data[:, 1:5].chunk(4, dim=-1) # [t,4]
            camera_pose_3x4 = camera_data[:, 7:].reshape(-1, 3, 4)  # [t, 3, 4]
            RT_list.append(rt34_to_44(camera_pose_3x4))  # [t, 4, 4]
            
            fx = 0.5 * max(ori_H, ori_W) / ori_W
            fy = 0.5 * max(ori_H, ori_W) / ori_H
            cx = 0.5
            cy = 0.5

            ## spatial transformations
            if self.spatial_transform is not None:
                if self.spatial_transform_type == 'resize_center_crop':
                    sample_H, sample_W = self.resolution[0], self.resolution[1]
                    scale = min(sample_H, sample_W) / min(ori_H, ori_W)
                    fx *= ori_W * scale
                    fy *= ori_H * scale
                    cx *= sample_W
                    cy *= sample_H
                else:
                    raise NotImplementedError

            camera_intrinsics_list.append(torch.tensor([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(-1, 3, 3).repeat(self.video_length, 1, 1))
        
        data = {
            'video': ref_img,
            'caption': caption,
            'video_path': ref_img_path,
            'fps': torch.tensor(self.fps // self.frame_stride),
            'frame_stride': torch.tensor(self.frame_stride),
            'RT': torch.cat(RT_list, dim=0),  # (camera_pose_sections x T) x 4 x 4
            'camera_intrinsics': torch.cat(camera_intrinsics_list, dim=0), # (camera_pose_sections x T) x 3 x 3
            'camera_data': torch.cat(camera_data_list),  # (camera_pose_sections x T) x 19
        }
        return data

    def __len__(self):
        return len(self.metadata)
