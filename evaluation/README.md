# Evaluation Code for CamI2V

This repo contains evaluation pipeline of CamI2V for camera controllability (RotErr, TranErr, CamMC) and visual quality (FVD).

## Installation

First update submodules required by this repo:

```shell
git submodule update --init
```

For system dependencies and tool compilation, please refer to [installation guide](install.md).

## Prepare Clip IDs for Test

### Option 1: Use Provided `pth` File (Recommended)

Download `CamI2V_test_metadata_1k.pth` and put under folder `datasets/RealEstate10K`:

```
wget https://huggingface.co/MuteApo/CamI2V/resolve/main/CamI2V_test_metadata_1k.pth
mv CamI2V_test_metadata_1k.pth datasets/RealEstate10K/
```

### Option 2: Follow Instructions in Data Processing

Please refer to [datasets](../datasets).

## Generate Videos for Test

Example usage:

```shell
config_file=configs/inference/003_cami2v_256x256.yaml
save_root=../test_results
suffix_name=256_CamI2V
torchrun --standalone --nproc_per_node 8 main/trainer.py --test \
    --base $config_file --logdir $save_root --name $suffix_name
```

Resulting file structure would be like:

```
─┬─ test_results/                <-- "save_root" variable above
 └─┬─ 256_CamI2V/                <-- "suffix_name" variable above
   └──┬─ images/
      └─┬─ test/
        └─┬─ <CONFIG_SUFFIX>/    <-- auto generated from "config_file" by trainer
          ├─── camera_data/
          ├─── condition/
          ├─── gt_video/
          ├─── image_condition/
          ├─── reconst/
          ├─── samples/
          └─── video_path/
```

For convenience, we use `EXP_DIR=${save_root}/${suffix_name}/images/test/<CONFIG_SUFFIX>` in evaluation code.

## Run Evaluation

### Camera Metrics

Evaluate RotErr, TranErr & CamMC simultaneously, conducting 5 trials for each video pair and averaging them for the results.

Quick start and example usage:

```shell
python glomap_evaluation.py --exp_dir $EXP_DIR
python utils/merge.py
python utils/summary.py
```

### FVD

Quick start and example usage:

```shell
python fvd_test.py --gt_folder $EXP_DIR/gt_video --sample_folder $EXP_DIR/samples
```
