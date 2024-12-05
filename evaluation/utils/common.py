import os
import re
import subprocess
from glob import glob
from typing import Literal

import imageio
import numpy as np
import torch
from moviepy.editor import VideoFileClip
from torch import Tensor


def get_frames(file: str, output_dir: str, ex: bool = False) -> tuple[int, int]:
    mp4 = VideoFileClip(file, audio=False)
    width, height = mp4.size
    align = {1088: 1080, 368: 360}
    if height in align:
        height = align[height]
        mp4 = VideoFileClip(file, audio=False, target_resolution=(height, width))
    # print(round(mp4.fps * mp4.duration), *mp4.size)

    os.makedirs(output_dir, exist_ok=True)
    for idx, frame in enumerate(mp4.iter_frames()):
        name = f"{idx:04d}" if ex else f"{idx:03d}"
        imageio.imwrite(f"{output_dir}/{name}.png", frame)
    mp4.close()

    return width, height


def load_rt_from_txt(file_path: str, comments: str = None) -> Tensor:
    return torch.from_numpy(np.loadtxt(file_path, comments=comments, dtype=np.float64))


def get_rt(folder: str) -> Tensor:
    files = sorted([x for x in glob(f"{folder}/*.txt") if re.search(r"(\d+)\.txt$", x)])
    return torch.stack([load_rt_from_txt(file) for file in files])


def rt34_to_44(rt: Tensor) -> Tensor:
    dummy = torch.tensor([[[0, 0, 0, 1]]] * rt.size(0), dtype=rt.dtype, device=rt.device)
    return torch.cat([rt, dummy], dim=1)


def relative_pose(rt: Tensor, mode: Literal["left", "right"]) -> Tensor:
    if mode == "left":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[:1].inverse() @ rt[1:]], dim=0)
    elif mode == "right":
        rt = torch.cat([torch.eye(4).unsqueeze(0), rt[1:] @ rt[:1].inverse()], dim=0)
    return rt


def normalize_t(rt: Tensor, ref: Tensor = None, eps: float = 1e-9):
    if ref is None:
        ref = rt
    scale = ref[:, :3, 3:4].norm(p=2, dim=1).amax() + eps
    return rt34_to_44(torch.cat([rt[:, :3, :3], rt[:, :3, 3:4] / scale], dim=-1))


def cli_wrapper(*cmd: list[str], redirect: bool = False, verbose: bool = False):
    if verbose:
        print(" ".join(cmd))
    file = subprocess.DEVNULL if redirect else None
    subprocess.run(cmd, stdout=file, stderr=file)
