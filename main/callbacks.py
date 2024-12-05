import os
import time
import logging
from typing import Any

from pytorch_lightning.utilities.types import STEP_OUTPUT

mainlogger = logging.getLogger('mainlogger')

import torch
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
from utils.save_video import log_local, prepare_to_log
import pdb


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images=8, clamp=True, rescale=True, save_dir=None, \
                 to_local=False, to_tensorboard=True, log_images_kwargs=None, save_suffix=''):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.to_local = to_local
        self.to_tensorboard = to_tensorboard
        self.clamp = clamp
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.prefix = "ImageTextcfg{}_CameraCondition{}_CameraCfg{}_eta{}_guidanceRescale{}_cfgScheduler={}_steps{}".format(
            self.log_images_kwargs["unconditional_guidance_scale"] if 'unconditional_guidance_scale' in self.log_images_kwargs else 7.5,
            self.log_images_kwargs["enable_camera_condition"] if 'enable_camera_condition' in self.log_images_kwargs else 'True',
            self.log_images_kwargs["camera_cfg"] if 'camera_cfg' in self.log_images_kwargs else 'None',
            self.log_images_kwargs['ddim_eta'] if 'ddim_eta' in self.log_images_kwargs else 1.0,
            self.log_images_kwargs['guidance_rescale'] if 'guidance_rescale' in self.log_images_kwargs else 0.7,
            self.log_images_kwargs['camera_cfg_scheduler'] if 'camera_cfg_scheduler' in self.log_images_kwargs else 'constant',
            self.log_images_kwargs['ddim_steps'] if 'ddim_steps' in self.log_images_kwargs else 50,
        ) + save_suffix
        if self.to_local:
            ## default save dir
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(os.path.join(self.save_dir, "train", self.prefix), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "val", self.prefix), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir, "test", self.prefix), exist_ok=True)

    def log_to_tensorboard(self, pl_module, batch_logs, filename, split, save_fps=8):
        """ log images and videos to tensorboard """
        global_step = pl_module.global_step
        for key in batch_logs:
            value = batch_logs[key]
            tag = "gs%d-%s/%s-%s" % (global_step, split, filename, key)
            if isinstance(value, list) and isinstance(value[0], str):
                captions = ' |------| '.join(value)
                pl_module.logger.experiment.add_text(tag, captions, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 5:
                video = value
                n = video.shape[0]
                video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
                frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n), padding=0) for framesheet in video]  # [3, n*h, 1*w]
                grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
                grid = (grid + 1.0) / 2.0
                grid = grid.unsqueeze(dim=0)
                pl_module.logger.experiment.add_video(tag, grid, fps=save_fps, global_step=global_step)
            elif isinstance(value, torch.Tensor) and value.dim() == 4:
                img = value
                grid = torchvision.utils.make_grid(img, nrow=int(n), padding=0)
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                pl_module.logger.experiment.add_image(tag, grid, global_step=global_step)
            else:
                pass

    def log_batch_imgs(self, trainer, pl_module, batch, batch_idx, split="train"):
        """ generate images, then save and log to tensorboard """
        skip_freq = self.batch_freq if split == "train" else 1
        if split == 'train':
            is_log = (batch_idx + 1) % trainer.accumulate_grad_batches == 0 and trainer.global_step % skip_freq == 0
        elif split == 'val':
            is_log = (batch_idx + 1) % skip_freq == 0
        elif split == 'test':
            is_log = (batch_idx + 1) % skip_freq == 0
        if is_log:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            filename = "gs{}_ep{}_idx{}_rank{}".format(
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
                pl_module.global_rank
            )
            
            if self.to_local and os.path.exists(f"{self.save_dir}/{split}/{self.prefix}/samples/{filename}.mp4"):
                print(f"Skip {self.save_dir}/{split}/{self.prefix}/samples/{filename}.mp4")
                return 

            torch.cuda.empty_cache()
            with torch.no_grad():
                log_func = pl_module.log_images
                batch_logs = log_func(batch, split=split, **self.log_images_kwargs)

            ## process: move to CPU and clamp
            batch_logs = prepare_to_log(batch_logs, self.max_images, self.clamp)
            torch.cuda.empty_cache()

            if self.to_local:
                mainlogger.info("Log [%s] batch <%s> to local ..." % (split, filename))
                save_dir = os.path.join(self.save_dir, split, self.prefix)
                log_local(batch_logs, save_dir, filename, save_fps=10)

            if self.to_tensorboard:
                filename = self.prefix + '_' + filename
                mainlogger.info("Log [%s] batch <%s> to tensorboard ..." % (split, filename))
                self.log_to_tensorboard(pl_module, batch_logs, filename, split, save_fps=10)

            mainlogger.info('Finish!')
            if is_train:
                pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        if self.batch_freq != -1 and pl_module.logdir and pl_module.global_rank == 0:
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        ## different with validation_step() that saving the whole validation set and only keep the latest,
        ## it records the performance of every validation (without overwritten) by only keep a subset
        if self.batch_freq != -1 and pl_module.logdir:
            self.log_batch_imgs(trainer, pl_module, batch, batch_idx, split="test")


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        # lightning update
        gpu_index = trainer.strategy.root_device.index
        # if int((pl.__version__).split('.')[1])>=7:
        #     gpu_index = trainer.strategy.root_device.index
        # else:
        #     gpu_index = trainer.root_gpu
        torch.cuda.reset_peak_memory_stats(gpu_index)
        torch.cuda.synchronize(gpu_index)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        gpu_index = trainer.strategy.root_device.index
        # if int((pl.__version__).split('.')[1])>=7:
        #     gpu_index = trainer.strategy.root_device.index
        # else:
        #     gpu_index = trainer.root_gpu
        torch.cuda.synchronize(gpu_index)
        max_memory = torch.cuda.max_memory_allocated(gpu_index) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass
