import torch.nn as nn
from einops import rearrange

from lvdm.models.ddpm3d import LatentVisualDiffusion

import torch
from einops import rearrange, repeat
import logging
mainlogger = logging.getLogger('mainlogger')
import random
import pdb
class DynamiCrafter(LatentVisualDiffusion):
    def __init__(self,
                 diffusion_model_trainable_param_list=[],
                 *args,
                 **kwargs):
        super(DynamiCrafter, self).__init__(*args, **kwargs)
        self.diffusion_model_trainable_param_list = diffusion_model_trainable_param_list
        for p in self.model.parameters():
            p.requires_grad = False

        if 'TemporalTransformer.attn2' in self.diffusion_model_trainable_param_list:
            for n, m in self.model.named_modules():
                if m.__class__.__name__ == 'BasicTransformerBlock' and m.attn2.context_dim is None:       # attn2 of TemporalTransformer BasicBlocks
                    for p in m.attn2.parameters():
                        p.requires_grad = True

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        # params = list(self.model.parameters())
        params = [p for p in self.model.parameters() if p.requires_grad == True]
        mainlogger.info(f"@Training [{len(params)}] Trainable Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)

        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]

        return optimizer


