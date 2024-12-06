# modified from https://github.com/hehao13/CameraCtrl/blob/main/tools/get_realestate_clips.py

import argparse
import json
import os

import imageio
from decord import VideoReader
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="datasets/RealEstate10K")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--low_idx", type=int, default=0, help="used for parallel processing")
    parser.add_argument("--high_idx", type=int, default=-1, help="used for parallel processing")
    parser.add_argument("--resolution", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    save_path = f"{args.dataset_root}/video_clips/{args.split}"
    os.makedirs(save_path, exist_ok=True)

    with open(f"{args.dataset_root}/{args.split}_video2clip.json", "r") as f:
        video2clips: list[tuple[str, list[str]]] = list(json.load(f).items())
    if args.high_idx != -1:
        video2clips = video2clips[: args.high_idx]
    video2clips = video2clips[args.low_idx :]

    for video_name, clip_list in tqdm(video2clips):
        video_path = f"{args.dataset_root}/videos/{args.split}/{video_name}.mp4"
        if not os.path.exists(video_path):
            continue

        video = VideoFileClip(video_path, audio=False, target_resolution=(args.resolution, None))

        clip_save_dir = f"{args.dataset_root}/video_clips/{args.split}/{video_name}"
        os.makedirs(clip_save_dir, exist_ok=True)

        for clip_name in tqdm(clip_list):
            with open(f"{args.dataset_root}/pose_files/{args.split}/{clip_name}.txt", "r") as f:
                timestamps = [int(x.split(" ")[0]) for x in f.readlines()[1:]]
            if timestamps[-1] <= timestamps[0]:
                continue

            clip_save_path = f"{clip_save_dir}/{clip_name}.mp4"
            if os.path.exists(clip_save_path):
                try:
                    video_reader = VideoReader(clip_save_path)
                    if len(video_reader) == len(timestamps):
                        continue
                except:
                    pass

            frames = [video.get_frame(t / 1_000_000) for t in timestamps]
            imageio.mimsave(clip_save_path, frames, fps=video.fps, macro_block_size=None)
