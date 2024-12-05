import glob
from argparse import ArgumentParser

import torch
from fvdcal import FVDCalculation
from fvdcal.video_preprocess import load_video
from torch import Tensor
from tqdm import tqdm


class MyFVDCalculation(FVDCalculation):
    def calculate_fvd_by_video_list(self, real_videos: Tensor, generated_videos: Tensor, model_path="FVD/model"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._load_model(model_path, device)

        fvd = self._compute_fvd_between_video(model, real_videos, generated_videos, device)

        return fvd.detach().cpu().numpy()


def load_videos(paths, desc):
    return torch.stack([load_video(path, num_frames=None) for path in tqdm(paths, desc=desc)])


def metric(gt_folder, sample_folder):
    gt_video_paths = glob.glob(f"{gt_folder}/*.mp4")
    sample_video_paths = glob.glob(f"{sample_folder}/*.mp4")

    gt_videos = load_videos(gt_video_paths, "loading real videos")
    sample_videos = load_videos(sample_video_paths, "loading generated videos")

    score_videogpt = fvd_videogpt.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(score_videogpt)

    score_stylegan = fvd_stylegan.calculate_fvd_by_video_list(gt_videos, sample_videos)
    print(score_stylegan)

    return score_videogpt, score_stylegan


fvd_videogpt = MyFVDCalculation(method="videogpt")
fvd_stylegan = MyFVDCalculation(method="stylegan")

parser = ArgumentParser()
parser.add_argument("--gt_folder", type=str)
parser.add_argument("--sample_folder", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    metric(args.gt_folder, args.sample_folder)
