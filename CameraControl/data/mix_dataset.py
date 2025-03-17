import os
import random

import numpy as np
import omegaconf
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms


class MixDataset(Dataset):

    def __init__(self,
                 metadata_path,
                 data_root,
                 video_length=16,
                 resolution=[256, 256],  # H, W
                 frame_stride=1,  # [min, max], do not larger than 32 when video_length=16
                 frame_stride_for_condition=0,
                 spatial_transform=None,
                 load_raw_resolution=True,
                 ):
        self.metadata_path = metadata_path
        self.data_root = data_root

        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.frame_stride_for_condition = frame_stride_for_condition
        self.frame_stride = frame_stride
        self.spatial_transform_type = spatial_transform
        assert self.spatial_transform_type in ['resize_center_crop']

        self.load_raw_resolution = load_raw_resolution

        all_metadata = torch.load(self.metadata_path)
        self.captions = np.array([metadata['caption'] for metadata in all_metadata], dtype=np.unicode_)
        self.video_paths = np.array([metadata['video_path'] for metadata in all_metadata], dtype=np.unicode_)
        self.num_frames = np.array([metadata['camera_extrinsics'].shape[0] for metadata in all_metadata])
        self.camera_extrinsics = torch.cat([metadata['camera_extrinsics'] for metadata in all_metadata], dim=0).numpy()
        self.end_idx = self.num_frames.cumsum()
        self.start_idx = self.end_idx - self.num_frames
        self.camera_intrinsics = []
        for metadata in all_metadata:
            camera_intrinsics = metadata['camera_intrinsics']
            if camera_intrinsics.ndim == 1:
                fx, fy, cx, cy = camera_intrinsics # [1, 4]
                camera_intrinsics_3x3 = torch.Tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ])  # 3x3
                self.camera_intrinsics.append(camera_intrinsics_3x3)
            else:
                self.camera_intrinsics.append(camera_intrinsics)
        self.camera_intrinsics = torch.stack(self.camera_intrinsics, dim=0).numpy() # num_data,3,3

        del all_metadata

        print(f'============= length of dataset {self.__len__()} =============')

    def _resize_for_rectangle_crop(self, frames, H, W, fx, fy, cx, cy):
        '''
        :param frames: C,F,H,W
        :param image_size: H,W
        :return: frames: C,F,crop_H,crop_W;  camera_intrinsics: F,3,3
        '''
        ori_H, ori_W = frames.shape[-2:]
        if ori_W / ori_H > W / H:
            frames = transforms.functional.resize(frames, size=[H, int(ori_W * H / ori_H)])
        else:
            frames = transforms.functional.resize(frames, size=[int(ori_H * W / ori_W), W])

        resized_H, resized_W = frames.shape[2], frames.shape[3]
        frames = frames.squeeze(0)

        delta_H = resized_H - H
        delta_W = resized_W - W

        top, left = delta_H // 2, delta_W // 2
        frames = transforms.functional.crop(frames, top=top, left=left, height=H, width=W)

        fx = fx * resized_W
        fy = fy * resized_H
        cx = cx * W
        cy = cy * H
        _1, _0 = torch.ones_like(fx), torch.zeros_like(fx)
        camera_intrinsics = torch.hstack([fx, _0, cx, _0, fy, cy, _0, _0, _1]).reshape(-1, 3, 3)  # [F, 3, 3]

        return frames, camera_intrinsics, resized_H, resized_W

    def __getitem__(self, index):
        index = index % len(self)
        video_path = os.path.join(self.data_root, self.video_paths[index])
        caption = self.captions[index]
        camera_extrinsics = torch.from_numpy(self.camera_extrinsics[self.start_idx[index]:self.end_idx[index]])  # [F, 4, 4]
        camera_intrinsics = torch.from_numpy(self.camera_intrinsics[index])  # [3, 3]
        fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]
        frame_num = camera_extrinsics.shape[0]

        if self.load_raw_resolution:
            video_reader = VideoReader(video_path, ctx=cpu(0))
        else:
            video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
            assert len(video_reader) < self.video_length, print(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")

        fps_ori = video_reader.get_avg_fps()

        frame_stride_drop = 0
        while True:
            if isinstance(self.frame_stride, int):
                frame_stride = max(self.frame_stride - frame_stride_drop, 1)
            elif (isinstance(self.frame_stride, list) or isinstance(self.frame_stride, omegaconf.listconfig.ListConfig)) and len(self.frame_stride) == 2:  # [min, max]
                assert (self.frame_stride[0] <= self.frame_stride[1]), f"frame_stride[0]({self.frame_stride[0]}) > frame_stride[1]({self.frame_stride[1]})"
                frame_stride = random.randint(self.frame_stride[0], self.frame_stride[1])
            else:
                print(type(self.frame_stride))
                print(len(self.frame_stride))
                print(f"frame_stride={self.frame_stride}")
                raise NotImplementedError

            required_frame_num = frame_stride * (self.video_length - 1) + 1
            if frame_num < required_frame_num:
                if isinstance(self.frame_stride, int) and frame_num < required_frame_num * 0.5:
                    frame_stride_drop += 1
                    continue
                else:
                    frame_stride = frame_num // self.video_length
                    required_frame_num = frame_stride * (self.video_length - 1) + 1
            break

        ## select a random clip
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices = [start_idx + frame_stride * i for i in range(self.video_length)]

        frames = video_reader.get_batch(frame_indices)
        del video_reader

        ## process data
        frames = torch.from_numpy(frames.asnumpy()).permute(3, 0, 1, 2).float()  # [f,h,w,c] -> [c,f,h,w]
        camera_extrinsics = camera_extrinsics[frame_indices]
        camera_data = torch.cat([
            torch.Tensor([[0, fx, fy, cx, cy, 0, 0]]).repeat(self.video_length, 1),  # F, 7
            camera_extrinsics[:, :3, :4].reshape(self.video_length, -1)
        ], dim=-1)  # F, 19

        ## spatial transformations
        if self.spatial_transform_type == 'resize_center_crop':
            frames, camera_intrinsics, resized_H, resized_W = self._resize_for_rectangle_crop(
                frames,
                self.resolution[0], self.resolution[1],  # H, W
                fx, fy, cx, cy
            )
            camera_data[:, 1:5] = torch.stack(
                [
                    camera_intrinsics[:, 0, 0],  # fx
                    camera_intrinsics[:, 1, 1],  # fy
                    camera_intrinsics[:, 0, 2],  # cx
                    camera_intrinsics[:, 1, 2],  # cy
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError

        frames = (frames / 255 - 0.5) * 2
        fps_clip = fps_ori // frame_stride

        data = {
            'video': frames,  # [c,f,h,w]
            'caption': caption,
            'video_path': video_path,
            'fps': fps_clip,
            'frame_stride': frame_stride if self.frame_stride_for_condition == 0 else self.frame_stride_for_condition,
            'RT': camera_extrinsics,  # Fx4x4
            'camera_data': camera_data,  # F, 19
            'camera_intrinsics': camera_intrinsics,  # Fx3x3
        }

        return data

    def __len__(self):
        return len(self.video_paths)
