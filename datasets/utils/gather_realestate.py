# modified from https://github.com/hehao13/CameraCtrl/blob/main/tools/gather_realestate.py

import argparse
import json
import os
from collections import defaultdict

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="datasets/RealEstate10K")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    folder = f"{args.dataroot}/pose_files/{args.split}"
    all_txts = os.listdir(folder)
    print(f"There are {len(all_txts)} video clips in the folder {folder}")

    video_paths = defaultdict(list)
    for txt in tqdm(all_txts):
        with open(f"{folder}/{txt}", "r") as f:
            lines = f.readlines()
        video_name = lines[0].strip().split("=")[-1]
        video_paths[video_name].append(txt.split(".")[0])

    print(f"There are {len(video_paths)} videos in the folder {folder}")

    with open(f"{args.dataroot}/{args.split}_video2clip.json", "w") as f:
        json.dump(video_paths, fp=f)
