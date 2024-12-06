import argparse
import json
import os

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="datasets/RealEstate10K")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    with open(f"{args.dataset_root}/{args.split}_video2clip.json", "r") as file:
        video_paths = json.load(file)

    clip_id_to_video_path = {}
    for video_name, clip_id_list in video_paths.items():
        print(video_name, clip_id_list)
        for clip_id in clip_id_list:
            clip_id_to_video_path[clip_id] = video_name + "/" + clip_id

    pose_files_root = f"{args.dataset_root}/pose_files/{args.split}"
    valid_metadata_root = f"{args.dataset_root}/valid_metadata/{args.split}"
    os.makedirs(valid_metadata_root, exist_ok=True)

    video_root = f"{args.dataset_root}/video_clips/{args.split}"

    with open(f"{args.dataset_root}/{args.split}_captions.json", "r") as file:
        all_captions = json.load(file)

    num = 0
    valid_metadata_info_path = f"{args.dataset_root}/{args.split}_valid_list.txt"
    valid_metadata_info = []
    for clip_id, _ in tqdm(all_captions.items()):

        clip_id = clip_id.split(".")[0]
        caption = all_captions[clip_id + ".mp4"][0]
        if len(all_captions[clip_id + ".mp4"]) != 1:
            print(all_captions[clip_id + ".mp4"])
        video_path = clip_id_to_video_path[clip_id]
        if clip_id not in clip_id_to_video_path or (not os.path.exists(os.path.join(video_root, video_path) + ".mp4")):
            num += 1
            print("missing", num, os.path.join(video_root, video_path) + ".mp4")
        else:
            with open(f"{pose_files_root}/{clip_id}.txt", "r") as f:
                lines = f.readlines()
            lines = [line.strip() for line in lines]
            if len(lines[1:]) >= 16:
                valid_metadata_info.append(clip_id)
                url = lines[0]
                lines = [url] + [video_path] + [caption] + lines[1:]
                with open(f"{valid_metadata_root}/{clip_id}.txt", "w") as file:
                    for item in lines:
                        file.write(item + "\n")
            else:
                print("video path len is smaller than 16")

    with open(valid_metadata_info_path, "w") as file:
        for item in valid_metadata_info:
            file.write(item + "\n")
