import argparse
import os
import shutil
import time
import uuid

import pandas as pd
from torch import Tensor

from utils.common import cli_wrapper, get_frames, get_rt, load_rt_from_txt, normalize_t, relative_pose, rt34_to_44
from utils.convert import write_depth_pose_from_colmap_format


def run_glomap(img_dir: str, pose_dir: str, f: float, cx: float, cy: float) -> tuple[Tensor, float]:
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
    c2w = w2c.inverse()
    rel_c2w = relative_pose(c2w, mode="left")

    return rel_c2w


def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:  # N, 3, 3
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:  # N, 3
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:  # N, 3, 4
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)


def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float]:  # N, 3, 4
    RotErr = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).sum().item()  # N, 3, 3

    c2w_1_rel = normalize_t(c2w_1, c2w_1)
    c2w_2_rel = normalize_t(c2w_2, c2w_2)

    TransErr = calc_transerr(c2w_1_rel[:, :3, 3], c2w_2_rel[:, :3, 3]).sum().item()  # N, 3, 1
    CamMC = calc_cammc(c2w_1_rel[:, :3, :4], c2w_2_rel[:, :3, :4]).sum().item()  # N, 3, 4

    return RotErr, TransErr, CamMC


def evaluate(tmp_dir: str, name: str):
    gt = load_rt_from_txt(f"{args.exp_dir}/camera_data/{name}.txt")
    gt_w2c = gt[:, 6:].reshape((-1, 3, 4))
    gt_c2w = rt34_to_44(gt_w2c).inverse()
    gt_rel_c2w = relative_pose(gt_c2w, mode="left")

    img_dir = f"{tmp_dir}/img"
    os.makedirs(img_dir, exist_ok=True)
    get_frames(f"{args.exp_dir}/samples/{file}", img_dir)

    start = time.perf_counter()

    fx, fy, cx, cy = gt[0, :4]
    sample_rel_c2w = run_glomap(img_dir, f"{tmp_dir}/pose", fx, cx, cy)

    end = time.perf_counter()

    metrics = ["RotErr", "TransErr", "CamMC"]
    items = metric(gt_rel_c2w.clone(), sample_rel_c2w.clone())
    print(", ".join(map(lambda x: f"{x[0]}: {x[1]:.3f}", zip(metrics, items))))

    return file, round(end - start, 2), *items


parser = argparse.ArgumentParser()
parser.add_argument("--exp_dir", type=str, required=True)
parser.add_argument("--tmp_dir", type=str, default="/tmp")
parser.add_argument("--trial_id", type=int, default=0)
parser.add_argument("--low_idx", type=int, default=0)
parser.add_argument("--high_idx", type=int, default=-1)

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
        metrics = ["RotErr", "TransErr", "CamMC"]
        with open(csv_path, "w") as f:
            f.write("Name,Time," + ",".join(metrics) + "\n")

    files = sorted(os.listdir(f"{args.exp_dir}/gt_video"))
    if args.high_idx != -1:
        files = files[: args.high_idx]
    files = files[args.low_idx :]

    for file in filter(lambda x: x not in names, files):
        name, _ = os.path.splitext(file)
        print(f"[Trial ID {args.trial_id}] {name}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        
        try:
            entry = evaluate(tmp_dir, name)
        except Exception as e:
            # print(e)
            # with open("failed_videos.txt", "a+") as f:
            #     f.write(f"{file}: {e}\n")
            continue

        pd.DataFrame([entry]).to_csv(csv_path, mode="a", header=False, index=False)
