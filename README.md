# CamI2V: Camera-Controlled Image-to-Video Diffusion Model

<div align="center">
    <a href="https://arxiv.org/abs/2410.15957"><img src="https://img.shields.io/static/v1?label=arXiv&message=2410.15957&color=b21d1a"></a>
    <a href="https://zgctroy.github.io/CamI2V"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=green"></a>
    <a href="https://huggingface.co/MuteApo/CamI2V/tree/main"><img src="https://img.shields.io/static/v1?label=HuggingFace&message=Checkpoints&color=blue"></a>
</div>

## 🎥 Gallery

<table>
    <tr>
        <td align="center">
            rightward rotation and zoom in<br>(CFG=4, FS=6, step=50, ratio=0.6, scale=0.1)
        </td>
        <td align="center">
            leftward rotation and zoom in<br>(CFG=4, FS=6, step=50, ratio=0.6, scale=0.1)
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/74a764f4-0631-4fbe-94b9-af51057f99a5" width="75%">
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/99309759-8355-4ee1-95c4-897f01c46720" width="75%">
        </td>
    </tr>
    <tr>
        <td align="center">
            zoom in and upward movement<br>(CFG=4, FS=6, step=50, ratio=0.8, scale=0.2)
        </td>
        <td align="center">
            downward movement and zoom-out<br>(CFG=4, FS=6, step=50, ratio=0.8, scale=0.2)
        </td>
    </tr>
    <tr>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/aef4cc2e-fd7e-46db-82bc-a7e59aab5963" width="75%">
        </td>
        <td align="center">
            <img src="https://github.com/user-attachments/assets/f204992a-d729-492c-a663-85f9b80680f5" width="75%">
        </td>
    </tr>
</table>

## 🌟 News and Todo List

- 🔥 25/03/26: Release our dataset [RealCam-Vid](https://huggingface.co/datasets/MuteApo/RealCam-Vid) v1, for metric-scale camera-controlled video generation!
- 🔥 25/03/17: Upload test metadata used in our paper to make easier evaluation!
- 🔥 25/02/15: Release demo of [RealCam-I2V](https://zgctroy.github.io/RealCam-I2V/) for real-world applications, code will be available at [repo](https://github.com/ZGCTroy/RealCam-I2V)!
- 🔥 25/01/12: Release [CamI2V (512x320, 100k)](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_100k.pt) checkpoint. We plan to release a more advanced model with longer training soon!
- 🔥 25/01/02: Release [CamI2V (512x320, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_50k.pt) checkpoint, which is suitable for research propose and comparison!
- 🔥 24/12/24: Integrate [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL) in gradio demo, caption your own input image by this powerful VLM!
- 🔥 24/12/23: Release checkpoint of [CamI2V (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cami2v.pt).
- 🔥 24/12/16: Release reproduced non-official [MotionCtrl (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_motionctrl.pt) and [CameraCtrl (256x256, 50k)](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cameractrl.pt) checkpoints.
- 🔥 24/12/09: Release training configs and scripts.
- 🔥 24/12/06: Release [dataset pre-process code](datasets) for RealEstate10K.
- 🔥 24/12/02: Release [evaluation code](evaluation) for RotErr, TransErr, CamMC and FVD.
- 🌱 24/11/16: Release model code of CamI2V on I2V model [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter), including implementation for MotionCtrl and CameraCtrl.

## 📈 Performance

Measured under 256x256 resolution, 50k training steps, 25 DDIM steps, text-image CFG 7.5, camera CFG 1.0 (no camera CFG).

| Method        |  RotErr↓   | TransErr↓  |   CamMC↓   | FVD↓<br>(VideoGPT) | FVD↓<br>(StyleGAN) |
| :------------ | :--------: | :--------: | :--------: | :----------------: | :----------------: |
| DynamiCrafter |   3.3415   |   9.8024   |   11.625   |       106.02       |       92.196       |
| MotionCtrl    |   0.8636   |   2.5068   |   2.9536   |       70.820       |       60.363       |
| CameraCtrl    |   0.7064   |   1.9379   |   2.3070   |       66.713       |       57.644       |
| CamI2V        | **0.4120** | **1.3409** | **1.5291** |     **62.439**     |     **53.361**     |

### Inference Speed and GPU Memory

| Method        | # Parameters | GPU Memory | Generation Time<br>(RTX 3090) |
| :------------ | :----------: | :--------: | :---------------------------: |
| DynamiCrafter |    1.4 B     | 11.14 GiB  |            8.14 s             |
| MotionCtrl    |   + 63.4 M   | 11.18 GiB  |            8.27 s             |
| CameraCtrl    |   + 211 M    | 11.56 GiB  |            8.38 s             |
| CamI2V        |   + 261 M    | 11.67 GiB  |            10.3 s             |

## ⚙️ Environment

### Quick Start

```shell
conda create -n cami2v python=3.10
conda activate cami2v

conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y xformers -c xformers
pip install -r requirements.txt
```

## 💫 Inference

### Download Model Checkpoints

| Model      | Resolution |                                                                    Training Steps                                                                    |
| :--------- | :--------: | :--------------------------------------------------------------------------------------------------------------------------------------------------: |
| CamI2V     |  512x320   | [50k](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_50k.pt), [100k](https://huggingface.co/MuteApo/CamI2V/blob/main/512_cami2v_100k.pt) |
| CamI2V     |  256x256   |                                         [50k](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cami2v.pt)                                         |
| CameraCtrl |  256x256   |                                       [50k](https://huggingface.co/MuteApo/CamI2V/blob/main/256_cameractrl.pt)                                       |
| MotionCtrl |  256x256   |                                       [50k](https://huggingface.co/MuteApo/CamI2V/blob/main/256_motionctrl.pt)                                       |

Currently we release 256x256 checkpoints with 50k training steps of DynamiCrafter-based CamI2V, CameraCtrl and MotionCtrl, which is suitable for research propose and comparison.

We also release 512x320 checkpoints of our CamI2V with longer training, make possible higher resolution and more advanced camera-controlled video generation.

Download above checkpoints and put under `ckpts` folder.
Please edit `ckpt_path` in `configs/models.json` if you have a different model path.

### Download Qwen2-VL Captioner (Optional)

Not required but recommend.
It is used to caption your custom image in gradio demo for video generaion.
We prefer the [AWQ](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-AWQ) quantized version of Qwen2-VL due to speed and GPU memory.

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

## 🚀 Training

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
    --base configs/training/003_cami2v_256x256.yaml \
    --name 256_CamI2V
```

## 🔧 Evaluation

We calculate RotErr, TransErr, CamMC and FVD to evaluate camera controllability and visual quality. 
Code and installation guide for requirements are provided in [evaluation](evaluation) folder, including COLMAP and GLOMAP.
Support for VBench is planned in months as well.

## 🤗 Related Repo

[RealCam-I2V: https://github.com/ZGCTroy/RealCam-I2V](https://github.com/ZGCTroy/RealCam-I2V)

[CameraCtrl: https://github.com/hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl)

[MotionCtrl: https://github.com/TencentARC/MotionCtrl](https://github.com/TencentARC/MotionCtrl)

[DynamiCrafter: https://github.com/Doubiiu/DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)

## 🗒️ Citation

```
@article{zheng2024cami2v,
  title={CamI2V: Camera-Controlled Image-to-Video Diffusion Model},
  author={Zheng, Guangcong and Li, Teng and Jiang, Rui and Lu, Yehao and Wu, Tao and Li, Xi},
  journal={arXiv preprint arXiv:2410.15957},
  year={2024}
}

@article{li2025realcam,
    title={RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control}, 
    author={Li, Teng and Zheng, Guangcong and Jiang, Rui and Zhan, Shuigen and Wu, Tao and Lu, Yehao and Lin, Yining and Li, Xi},
    journal={arXiv preprint arXiv:2502.10059},
    year={2025},
}
```
