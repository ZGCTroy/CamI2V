import argparse
import os
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
from moviepy import VideoFileClip
from torch import Tensor
from torchvision import transforms

from utils.common import (
    cli_wrapper,
    get_frames,
    get_rt,
    load_rt_from_txt,
    normalize_t,
    relative_pose,
    rt34_to_44,
)
from utils.convert import write_depth_pose_from_colmap_format
from utils.read_write_model import (
    read_cameras_text,
    read_images_text,
    read_points3D_text,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "CameraControl"))
from depth_anything_v2.dpt import DepthAnythingV2


def run_glomap(
    img_dir: str, pose_dir: str, f: float, cx: float, cy: float, depth_map: Tensor
) -> tuple[Tensor, float]:
    def convert(config: dict) -> list[str]:
        return sum([[f"--{k}", f"{v}"] for k, v in config.items()], [])

    def runner(exec: str, stage: str, redirect: bool = False):
        cli_wrapper(exec, stage, *convert(config[stage]), redirect=redirect)

    model_dir = f"{pose_dir}/model"
    os.makedirs(model_dir, exist_ok=True)

    db_path = f"{pose_dir}/database.db"

    config = {
        "feature_extractor": {
            "database_path": db_path,
            "image_path": img_dir,
            "ImageReader.single_camera": 1,
            "ImageReader.camera_model": "SIMPLE_PINHOLE",
            "ImageReader.camera_params": f"{f},{cx},{cy}",
            "SiftExtraction.estimate_affine_shape": 1,
            "SiftExtraction.domain_size_pooling": 1,
        },
        "sequential_matcher": {
            "database_path": db_path,
            "SiftMatching.guided_matching": 1,
            "SiftMatching.max_num_matches": 65536,
        },
        "mapper": {
            "database_path": db_path,
            "image_path": img_dir,
            "output_path": model_dir,
            "output_format": "txt",
            "RelPoseEstimation.max_epipolar_error": 4,
            "BundleAdjustment.optimize_intrinsics": 0,
        },
    }

    runner("colmap", "feature_extractor", redirect=True)
    runner("colmap", "sequential_matcher", redirect=True)
    runner("glomap", "mapper", redirect=True)

    write_depth_pose_from_colmap_format(f"{model_dir}/0", model_dir, ext=".txt")

    w2c = rt34_to_44(get_rt(f"{model_dir}/poses"))

    colmap_camera = read_cameras_text(f"{model_dir}/0/cameras.txt")[1]
    colmap_point_clouds = read_points3D_text(f"{model_dir}/0/points3D.txt")
    colmap_images = read_images_text(f"{model_dir}/0/images.txt")
    colmap_image_name_to_image_id = {x.name: x.id for x in colmap_images.values()}

    image_id = colmap_image_name_to_image_id["000.png"]
    colmap_point_cloud_in_this_view = [x for x in colmap_point_clouds.values() if image_id in x.image_ids]

    colmap_points3d = np.stack([x.xyz for x in colmap_point_cloud_in_this_view])
    colmap_points3d = np.concatenate([colmap_points3d, np.ones_like(colmap_points3d[..., :1])], axis=-1)
    colmap_points3d_cam = (colmap_points3d @ w2c[0].T.numpy())[:, :3]

    colmap_points2d_idxs = np.concatenate(
        [x.point2D_idxs[np.argwhere(x.image_ids == image_id)[0]] for x in colmap_point_cloud_in_this_view]
    )
    colmap_images_uv = colmap_images[image_id].xys[colmap_points2d_idxs]

    dpt_u = colmap_images_uv[:, 0] / colmap_camera.width * depth_map.shape[1]
    dpt_v = colmap_images_uv[:, 1] / colmap_camera.height * depth_map.shape[0]
    dpt_d = depth_map[dpt_v.astype(np.int32), dpt_u.astype(np.int32)]
    depth_scale = np.median(dpt_d / colmap_points3d_cam[:, 2]).item()

    c2w = w2c.inverse()
    rel_c2w = relative_pose(c2w, mode="left")

    return rel_c2w, depth_scale


def estimate(video_path: str, output_dir: str, cam_param: Tensor) -> tuple[Tensor, float]:
    img_dir = f"{output_dir}/img"
    pose_dir = f"{output_dir}/pose"

    os.makedirs(img_dir, exist_ok=True)
    get_frames(f"{video_path}/{file}", img_dir)

    raw_size, depth_map = get_depth_map(f"{video_path}/{file}")

    fx, fy, cx, cy = cam_param
    assert fx > 10 and fy > 10, "intrinsics here should not be normalized "

    return run_glomap(img_dir, pose_dir, fx, cx, cy, depth_map)


def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # N, 3, 3
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # N, 3
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # N, 3, 4
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)


def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float]:  # N, 3, 4
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()  # N, 3, 3

    # relateive metric, normalized by self
    c2w_1_rel = normalize_t(c2w_1, c2w_1)
    c2w_2_rel = normalize_t(c2w_2, c2w_2)

    TransErr_rel = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()  # N, 3, 1
    CamMC_rel = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()  # N, 3, 4

    # absolute metric, normalized by c2w_1
    c2w_1_abs = normalize_t(c2w_1, c2w_1)
    c2w_2_abs = normalize_t(c2w_2, c2w_1)

    TransErr_abs = calc_transerr(c2w_1_abs[:, :3, 3], c2w_2_abs[:, :3, 3]).sum().item()  # N, 3, 1
    CamMC_abs = calc_cammc(c2w_1_abs[:, :3, :4], c2w_2_abs[:, :3, :4]).sum().item()  # N, 3, 4

    return RotErr, TransErr_rel, CamMC_rel, TransErr_abs, CamMC_abs


@torch.no_grad
@torch.autocast("cuda")
def get_depth_map(video_path: str) -> tuple[tuple[int, int], np.ndarray]:
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
    tf = transforms.ToTensor()
    video_clip = VideoFileClip(video_path, audio=False)
    w, h = video_clip.size
    size_14x = (round(h / 14) * 14, round(w / 14) * 14)
    video_clip = VideoFileClip(video_path, audio=False, target_resolution=size_14x)
    for frame in video_clip.iter_frames():
        break
    input = ((tf(frame) - mean) / std).unsqueeze(0).cuda()
    output = model(input).squeeze(0).cpu().numpy()
    return video_clip.size, output


def evaluate(tmp_dir: str, file: str):
    name, _ = os.path.splitext(file)

    gt = load_rt_from_txt(f"{args.exp_dir}/camera_data/{name}.txt")
    gt_w2c = gt[:, 6:].reshape((-1, 3, 4))
    gt_c2w = rt34_to_44(gt_w2c).inverse()
    gt_rel_c2w = relative_pose(gt_c2w, mode="left")

    start = time.perf_counter()

    colmap_rel_c2w_sample, depth_scale_sample = estimate(f"{args.exp_dir}/samples", f"{tmp_dir}/samples", gt[0, :4])
    colmap_rel_c2w_gt, depth_scale_gt = estimate(f"{args.exp_dir}/gt_video", f"{tmp_dir}/gt_video", gt[0, :4])

    end = time.perf_counter()
    entry = [file, round(end - start, 2)]

    gt_traj_metrics = metric(gt_rel_c2w.clone(), colmap_rel_c2w_sample.clone())

    colmap_rel_c2w_sample[:, :3, 3] *= depth_scale_sample
    colmap_rel_c2w_gt[:, :3, 3] *= depth_scale_gt

    gt_video_metrics = metric(colmap_rel_c2w_gt.clone(), colmap_rel_c2w_sample.clone())

    metrics = ["RotErr", "TransErr_rel", "CamMC_rel", "TransErr_abs", "CamMC_abs"]
    items = gt_traj_metrics[:3] + gt_video_metrics[-2:]
    print(", ".join(map(lambda x: f"{x[0]}: {x[1]:.3f}", zip(metrics, items))))

    entry.extend(items)
    return entry


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
parser.add_argument("--tmp_dir", type=str, default="/tmp")
parser.add_argument("--trial_id", type=int, default=0)
parser.add_argument("--low_idx", type=int, default=0)
parser.add_argument("--high_idx", type=int, default=-1)
parser.add_argument("--model_path",type=str, default="../pretrained_models/Depth-Anything-V2-Metric-Hypersim-Large/depth_anything_v2_metric_hypersim_vitl.pth")

if __name__ == "__main__":
    args = parser.parse_args()

    tmp_dir = f"{args.tmp_dir}/{uuid.uuid4().fields[0]:x}"
    *_, method, _, _, setting = args.exp_dir.split("/")
    csv_dir = f"results/{method}/{setting}"
    csv_path = f"{csv_dir}/trial_{args.trial_id}.csv"
    os.makedirs(csv_dir, exist_ok=True)

    if os.path.exists(csv_path):
        names = pd.read_csv(csv_path).iloc[:, 0].values.tolist()
    else:
        names = []
        metrics = ["RotErr", "TransErr_rel", "CamMC_rel", "TransErr_abs", "CamMC_abs"]
        with open(csv_path, "w") as f:
            f.write("Name,Time," + ",".join(metrics) + "\n")

    files = sorted(os.listdir(f"{args.exp_dir}/gt_video"))
    if args.high_idx != -1:
        files = files[: args.high_idx]
    files = files[args.low_idx :]

    model = DepthAnythingV2("vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20).cuda().eval()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=False))

    for file in filter(lambda x: x not in names, files):
        print(f"[Trial ID {args.trial_id}] {file}")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        try:
            entry = evaluate(tmp_dir, file)
        except Exception as e:
            print(e)
            continue

        pd.DataFrame([entry]).to_csv(csv_path, mode="a", header=False, index=False)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
