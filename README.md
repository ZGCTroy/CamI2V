# CamI2V
 official repo of paper for "CamI2V: Camera-Controlled Image-to-Video Diffusion Model"

github io:  https://zgctroy.github.io/CamI2V/

Abstract:
Recently, camera pose, as a user-friendly and physics-related condition, has been introduced into text-to-video diffusion model for camera control. However, existing methods simply inject camera conditions through a side input. These approaches neglect the inherent physical knowledge of camera pose, resulting in imprecise camera control, inconsistencies, and also poor interpretability. In this paper, we emphasize the necessity of integrating explicit physical constraints into model design. Epipolar attention is proposed for modeling all cross-frame relationships from a novel perspective of noised condition. This ensures that features are aggregated from corresponding epipolar lines in all noised frames, overcoming the limitations of current attention mechanisms in tracking displaced features across frames, especially when features move significantly with the camera and become obscured by noise. Additionally, we introduce register tokens to handle cases without intersections between frames, commonly caused by rapid camera movements, dynamic objects, or occlusions. To support image-to-video, we propose the multiple guidance scale to allow for precise control for image, text, and camera, respectively. Furthermore, we establish a more robust and reproducible evaluation pipeline to solve the inaccuracy and instability of existing camera control measurement. We achieve a 25.5% improvement in camera controllability on RealEstate10K while maintaining strong generalization to out-of-domain images. With optimization, only 24GB and 12GB is required for training and inference, respectively. We plan to release checkpoints, along with training and evaluation codes.

## News and ToDo List

- [ ] 2024-10-14: Release of checkpoints, training, and evaluation codes in a month


## Related Repo
[CameraCtrl: https://github.com/hehao13/CameraCtrl](https://github.com/hehao13/CameraCtrl)

[MotionCtrl: https://github.com/TencentARC/MotionCtrl/tree/animatediff](https://github.com/TencentARC/MotionCtrl/tree/animatediff)


## Citation
```
@inproceedings{anonymous2025camiv,
    title={CamI2V: Camera-Controlled Image-to-Video Diffusion Model},
    author={Anonymous},
    booktitle={Submitted to The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=dIZB7jeSUv},
    note={under review}
}
```




