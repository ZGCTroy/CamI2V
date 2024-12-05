import json
import os
from tqdm import tqdm

pose_files_root="/mnt/nfs/data/datasets/RealEstate10K/pose_files/train"
caption_root="/mnt/nfs/data/datasets/RealEstate10K/motionctrl_captions/train"
video_root="/mnt/nfs/data/datasets/RealEstate10K/videos/train"
train_txt_path="/mnt/nfs/data/datasets/RealEstate10K/motionctrl_train.txt"
caption_json="/mnt/nfs/data/datasets/RealEstate10K/motionctrl_train_caption.json"

with open(train_txt_path, 'r') as f:
    video_name_list = [line.strip() for line in f.readlines()]

# 读取 JSON 文件
with open(caption_json, 'r') as file:
    all_captions = json.load(file)

for video_name in tqdm(video_name_list):
    video_id, _ = os.path.splitext(video_name)
    caption = all_captions[video_id].strip()
    with open(f'{pose_files_root}/{video_name}', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    url = lines[0]
    lines = lines[:1] + [''] + [caption] + lines[1:]
    with open(f'{caption_root}/{video_name}', 'w') as file:
        for item in lines:
            file.write(item + '\n')









