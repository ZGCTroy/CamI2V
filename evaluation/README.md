# Evaluation Code for CamI2V

This repo contains evaluation pipeline of CamI2V for camera controllability (RotErr, TranErr, CamMC) and visual quality (FVD).

## Installation

First update submodules required by this repo:

```shell
git submodule update
```

For system dependencies and tool compilation, please refer to [installation guide](install.md).

## Generate Videos for Test

```shell
config_file=<YOUR_CONFIG_FILE>
save_root=<YOUR_SAVE_ROOT>
suffix_name=<YOUR_CUSTOM_SUFFIX>
torchrun --standalone --nproc_per_node 8 main/trainer.py --test --base $config_file --logdir $save_root --name $suffix_name
```

Resulting file structure would be like:

```
<YOUR_SAVE_ROOT>/                    <-- "save_root" variable above
  ├── <YOUR_CUSTOM_SUFFIX>/          <-- "suffix_name" variable above
    ├── images/
      ├── test/
        ├── <YOUR_CONFIG_SUFFIX>/    <-- auto generated from config yaml by trainer
          ├── camera_data/
          ├── condition/
          ├── gt_video/
          ├── image_condition/
          ├── reconst/
          ├── samples/
          ├── video_path/
```

For convenience, we use `EXP_DIR=<YOUR_SAVE_ROOT>/<YOUR_CUSTOM_SUFFIX>/images/test/<YOUR_CONFIG_SUFFIX>` in evaluation code.

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
