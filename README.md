# CamI2V: Camera-Controlled Image-to-Video Diffusion Model

<div align="center">
    <a href="https://arxiv.org/abs/2410.15957">
        <img src="https://img.shields.io/static/v1?label=arXiv&message=2410.15957&color=b21d1a">
    </a>
    <a href="https://zgctroy.github.io/CamI2V">
        <img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green">
    </a>
    <a href="https://huggingface.co/MuteApo/CamI2V/tree/main">
        <img src="https://img.shields.io/static/v1?label=HuggingFace&message=Checkpoints&color=blue">
    </a>
</div>

Official repo of paper for "CamI2V: Camera-Controlled Image-to-Video Diffusion Model".

Abstract:
Recent advancements have integrated camera pose as a user-friendly and physics-informed condition in video diffusion models, enabling precise camera control. In this paper, we identify one of the key challenges as effectively modeling noisy cross-frame interactions to enhance geometry consistency and camera controllability. We innovatively associate the quality of a condition with its ability to reduce uncertainty and interpret noisy cross-frame features as a form of noisy condition. Recognizing that noisy conditions provide deterministic information while also introducing randomness and potential misguidance due to added noise, we propose applying epipolar attention to only aggregate features along corresponding epipolar lines, thereby accessing an optimal amount of noisy conditions. Additionally, we address scenarios where epipolar lines disappear, commonly caused by rapid camera movements, dynamic objects, or occlusions, ensuring robust performance in diverse environments.
Furthermore, we develop a more robust and reproducible evaluation pipeline to address the inaccuracies and instabilities of existing camera control metrics. Our method achieves a 25.64\% improvement in camera controllability on the RealEstate10K dataset without compromising dynamics or generation quality and demonstrates strong generalization to out-of-domain images. Training and inference require only 24GB and 12GB of memory, respectively, for 16-frame sequences at 256×256 resolution. We will release all checkpoints, along with training and evaluation code. Dynamic videos are available for viewing on our project page.


## :star2: News and Todo List

- [x]  :fire: 2025/01/02: Release checkpoint of [CamI2V (512x320, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_50k.pt), which is suitable for research propose and comparison. We plan to release a more advanced model with longer training soon.
- [x]  :fire: 2024/12/24: Integrate [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) in gradio demo, you can now caption your own input image by this powerful VLM.
- [x]  :fire: 2024/12/23: Release checkpoint of [CamI2V (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cami2v.pt).
- [x]  :fire: 2024/12/16: Release non-officially reproduced checkpoints of [MotionCtrl (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_motionctrl.pt) and [CameraCtrl (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cameractrl.pt) on DynamiCrafter.
- [x]  :fire: 2024/12/09: Release training configs and scripts.
- [x]  :fire: 2024/12/06: Release [dataset pre-process code](datasets) for RealEstate10K.
- [x]  :fire: 2024/12/02: Release [evaluation code](evaluation) for RotErr, TransErr, CamMC and FVD.
- [x]  :seedling: 2024/11/16: Release model code of CamI2V for training and inference, including implementation for MotionCtrl and CameraCtrl.


## :chart_with_upwards_trend: Performance

Measured under 256x256 resolution, 16 frames, 25steps.

| Method                                                                              | RotErr $\downarrow$ | TransErr $\downarrow$ | CamMC $\downarrow$ | FVD $\downarrow$<br>(VideoGPT) | FVD $\downarrow$<br>(StyleGAN) |
| :---------------------------------------------------------------------------------- | :-----------------: | :-------------------: | :----------------: | :----------------------------: | :----------------------------: |
| DynamiCrafter                                                                       |       3.3415        |        9.8024         |       11.625       |             106.02             |             92.196             |
| + MotionCtrl                                                                        |       0.8636        |        2.5068         |       2.9536       |             70.820             |             60.363             |
| + Plucker Embedding<br>(Baseline, CameraCtrl)                                       |       0.7098        |        1.8877         |       2.2557       |           **66.077**           |             55.889             |
| + Plucker Embedding<br>+ Epipolar Attention Only on Reference Frame<br>(CamCo-like) |       0.5738        |        1.6014         |       1.8851       |             66.439             |             56.778             |
| + Plucker Embedding<br>+ Epipolar Attention<br>(Our CamI2V)                         |     **0.4758**      |      **1.4955**       |     **1.7153**     |             66.090             |           **55.701**           |
| + Plucker Embedding<br>+ 3D Full Attention                                          |       0.6299        |        1.8215         |       2.1315       |             71.026             |             60.00              |

- Our method demonstrates significant improvements over CameraCtrl, achieving a **32.96% reduction in Rotation Error**, a **25.64% decrease in CamMC**, and a **20.77% improvement in Translation Error**, **without decrease in FVD**. These results were obtained using text and image CFG set to 7.5, 25 steps, and **camera CFG set to 1.0 (no camera CFG)**. 

- Compared with CamCo-like (arXiv in June) method, we improve **17.08%, 6.61%, 9.00%** on RotErr, TransErr, and CamMC without FVD decrease, respectively.

### Inference Speed and GPU Memory

| Method                                                      | # Parameters | GPU Memory | Generation Time<br>(RTX 3090) |
| :---------------------------------------------------------- | :----------: | :--------: | :---------------------------: |
| DynamiCrafter                                               |    1.4 B     | 11.14 GiB  |            8.14 s             |
| + MotionCtrl                                                |   + 63.4 M   | 11.18 GiB  |            8.27 s             |
| + Plucker Embedding<br>(Baseline, CameraCtrl)               |   + 211 M    | 11.56 GiB  |            8.38 s             |
| + Plucker Embedding<br>+ Epipolar Attention<br>(Our CamI2V) |   + 261 M    | 11.67 GiB  |            10.3 s             |

## :gear: Environment

### Quick Start

```shell
conda create -n cami2v python=3.10
conda activate cami2v

conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y xformers -c xformers
pip install -r requirements.txt
```

## :dizzy: Inference

### Download Model Checkpoints

| Model      | Resolution | Training Steps |
| :--------- | :--------: | :------------: |
| CamI2V     |  512x320   |      50k       |
| CamI2V     |  256x256   |      50k       |
| CameraCtrl |  256x256   |      50k       |
| MotionCtrl |  256x256   |      50k       |

Currently we release checkpoints of DynamiCrafter-based CamI2V ([256x256](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cami2v.pt), [512x320](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_50k.pt)), CameraCtrl ([256x256](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cameractrl.pt)) and MotionCtrl ([256x256](https://huggingface.co/MuteApo/CamI2V/blob/main/256_motionctrl.pt)), with 50k training steps.
Download above checkpoints and put under `ckpts` folder.
Please edit `ckpt_path` in `configs/models.json` if you have a different model path.

### Download Qwen2-VL Captioner

Optional, not required but recommend.
It is used to caption a custom image in gradio demo for video generaion.
We prefer a quantized version of Qwen2-VL due to speed and GPU memory, like [GPTQ-Int8](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8) or [AWQ](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ) in official repo.
Download the pre-trained model and put under `pretrained_models` folder:

```shell
─┬─ pretrained_models/
 └─── Qwen2-VL-7B-Instruct-AWQ/
```

### Run Gradio Demo

```shell
python cami2v_gradio_app.py --use_qwenvl_captioner
```

Gradio may struggle to establish network connection, please re-try with `--use_host_ip`.

## :rocket: Training

### Prepare Dataset

Please follow instructions in [datasets](datasets) folder in this repo to download [RealEstate10K](https://google.github.io/realestate10k) dataset and pre-process necessary items like `video_clips` and `valid_metadata`.

### Download Pretrained Models

Download pretrained weights of base model DynamiCrafter ([256x256](https://huggingface.co/Doubiiu/DynamiCrafter), [512x320](https://huggingface.co/Doubiiu/DynamiCrafter_512)) and put under `pretrained_models` folder:

```shell
─┬─ pretrained_models/
 ├─┬─ DynamiCrafter/
 │ └─── model.ckpt
 └─┬─ DynamiCrafter_512/
   └─── model.ckpt
```

### Launch

Start training by passing config yaml to `--base` argument of `main/trainer.py`. Example training configs are provided in `configs` folder.

```shell
torchrun --standalone --nproc_per_node 8 main/trainer.py --train \
    --logdir $(pwd)/logs \
    --base configs/<YOUR_CONFIG_NAME>.yaml \
    --name <YOUR_LOG_NAME>
```

## :wrench: Evaluation

We calculate RotErr, TransErr, CamMC and FVD to evaluate camera controllability and visual quality. 
Code and installation guide for requirements are provided in [evaluation](evaluation) folder, including COLMAP and GLOMAP.
Support for VBench is planned in months as well.


## :hugs: Related Repo

[CameraCtrl: https://github.com/hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl)

[MotionCtrl: https://github.com/TencentARC/MotionCtrl](https://github.com/TencentARC/MotionCtrl)

[DynamiCrafter: https://github.com/Doubiiu/DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)


## :spiral_notepad: Citation

```
@article{zheng2024cami2v,
  title={CamI2V: Camera-Controlled Image-to-Video Diffusion Model},
  author={Zheng, Guangcong and Li, Teng and Jiang, Rui and Lu, Yehao and Wu, Tao and Li, Xi},
  journal={arXiv preprint arXiv:2410.15957},
  year={2024}
}
```
