import json
import os
from tqdm import tqdm
mode = 'test'
dataset_root="/mnt/nfs/data/datasets/RealEstate10K"
video_paths = f"{dataset_root}/CameraCtrl_RealEstate10K_{mode}.json"
with open(video_paths, 'r') as file:
    video_paths = json.load(file)

clip_id_to_video_path = {}
for video_name, clip_id_list in video_paths.items():
    print(video_name, clip_id_list)
    for clip_id in clip_id_list:
        clip_id_to_video_path[clip_id] =  video_name + "/" + clip_id

pose_files_root=f"{dataset_root}/pose_files/{mode}"
valid_metadata_root=f"{dataset_root}/CameraCtrl_valid_metadata/{mode}"
os.makedirs(valid_metadata_root, exist_ok=True)
all_captions_json=f"{dataset_root}/CameraCtrl_{mode}_captions.json"
video_root = f"{dataset_root}/video_clips/{mode}"




# 读取 JSON 文件
with open(all_captions_json, 'r') as file:
    all_captions = json.load(file)

num = 0
valid_metadata_info_path = f"{dataset_root}/CameraCtrl_{mode}_valid_list.txt"
valid_metadata_info = []
for clip_id, _ in tqdm(all_captions.items()):

    clip_id = clip_id.split('.')[0]
    caption = all_captions[clip_id+'.mp4'][0]
    if len(all_captions[clip_id+'.mp4']) != 1:
        print(all_captions[clip_id+'.mp4'])
    video_path = clip_id_to_video_path[clip_id]
    if clip_id not in clip_id_to_video_path or (not os.path.exists(os.path.join(video_root,video_path)+'.mp4')):
        num+=1
        print("missing", num, os.path.join(video_root,video_path)+'.mp4')
    else:
        with open(f'{pose_files_root}/{clip_id}.txt', 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        if len(lines[1:]) >= 16:
            valid_metadata_info.append(clip_id)
            url = lines[0]
            lines = [url] + [video_path] + [caption] + lines[1:]
            with open(f'{valid_metadata_root}/{clip_id}.txt', 'w') as file:
                for item in lines:
                    file.write(item + '\n')
        else:
            print('video path len is smaller than 16')

with open(valid_metadata_info_path, 'w') as file:
    for item in valid_metadata_info:
        file.write(item + '\n')










