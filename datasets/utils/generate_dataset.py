# modified from https://github.com/cashiwamochi/RealEstate10K_Downloader/blob/master/generate_dataset.py

import glob
import os
import pickle
from argparse import ArgumentParser
from time import sleep
from uuid import uuid4

from pytubefix import YouTube
from pytubefix.cli import on_progress
from pytubefix.streams import Stream
from tqdm import tqdm


class Data:
    def __init__(self, url: str, seqname: str, list_timestamps: list[str]):
        self.url: str = url
        self.list_seqnames: list[str] = []
        self.list_list_timestamps: list[list[str]] = []

        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def add(self, seqname, list_timestamps):
        self.list_seqnames.append(seqname)
        self.list_list_timestamps.append(list_timestamps)

    def __len__(self):
        return len(self.list_seqnames)


class DataDownloader:
    def __init__(self, dataroot: str, split: str):
        print("[INFO] Loading data list ... ", end="")
        self.dataroot = dataroot
        self.split = split
        self.output_root = f"{dataroot}/videos/{split}"
        os.makedirs(self.output_root, exist_ok=True)

        self.list_data_pkl = f"{dataroot}/{split}_list_data.pkl"
        self.list_seqnames = sorted(glob.glob(f"{dataroot}/pose_files/{split}/*.txt"))

        self.list_data = self.prepare_list_data()
        self.list_data.reverse()

        print(" Done! ")
        print("[INFO] {} movies are used in {} mode".format(len(self.list_data), self.split))

    def prepare_list_data(self) -> list[Data]:
        if os.path.exists(self.list_data_pkl):
            with open(self.list_data_pkl, "rb") as f:
                return pickle.load(f)

        list_data: list[Data] = []
        for txt_file in tqdm(self.list_seqnames, desc="Loading metadata"):
            dir_name = txt_file.split("/")[-1]
            seq_name = dir_name.split(".")[0]

            # extract info from txt
            seq_file = open(txt_file, "r")
            lines = seq_file.readlines()
            youtube_url = ""
            list_timestamps = []
            for idx, line in enumerate(lines):
                if idx == 0:
                    youtube_url = line.strip()
                else:
                    timestamp = int(line.split(" ")[0])
                    list_timestamps.append(timestamp)
            seq_file.close()

            isRegistered = False
            for i in range(len(list_data)):
                if youtube_url == list_data[i].url:
                    isRegistered = True
                    list_data[i].add(seq_name, list_timestamps)
                else:
                    pass

            if not isRegistered:
                list_data.append(Data(youtube_url, seq_name, list_timestamps))

        with open(self.list_data_pkl, "wb") as f:
            pickle.dump(list_data, f)
        
        return list_data

    def run(self):
        print("[INFO] Start downloading {} movies".format(len(self.list_data)))

        for global_count, data in enumerate(self.list_data):
            filepath = f"{self.output_root}/{data.url.split('=')[-1]}.mp4"
            if os.path.exists(filepath):
                continue

            print(f"[INFO] Downloading {global_count:04d}: {data.url}")

            tmpname = f"re10k_{uuid4().fields[0]:x}.mp4"
            try:
                # sometimes this fails because of known issues of pytube and unknown factors
                yt = YouTube(data.url, use_oauth=True, on_progress_callback=on_progress)
                stream: Stream = yt.streams.filter().order_by("resolution").last()
                # stream.download(output_path=self.output_root, filename=filename)
                stream.download(output_path="/tmp", filename=tmpname)
            except Exception as e:
                print(e)
                with open(f"failed_videos_{self.split}.txt", "a") as f:
                    f.writelines(os.path.basename(filepath) + "\n")
                continue

            os.system(f"mv /tmp/{tmpname} {filepath}")

            sleep(1)

    def show(self):
        print("########################################")
        global_count = 0
        for data in self.list_data:
            print(" URL : {}".format(data.url))
            for idx in range(len(data)):
                print(" SEQ_{} : {}".format(idx, data.list_seqnames[idx]))
                print(" LEN_{} : {}".format(idx, len(data.list_list_timestamps[idx])))
                global_count = global_count + 1
            print("----------------------------------------")

        print("TOTAL : {} sequnces".format(global_count))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataroot", type=str, default="datasets/RealEstate10K")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    args = parser.parse_args()

    downloader = DataDownloader(args.dataroot, args.split)
    downloader.show()
    downloader.run()
