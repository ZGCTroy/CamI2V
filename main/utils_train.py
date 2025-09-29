import logging
import os

import torch
from omegaconf import ListConfig, OmegaConf
from safetensors.torch import load_file

mainlogger = logging.getLogger('mainlogger')

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def smart_load(ckpt: str):
    if ckpt.endswith(".safetensors"):
        return load_file(ckpt)
    elif ckpt.endswith((".pt", ".pth", ".ckpt")):
        return torch.load(ckpt, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format {ckpt}")

def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        if isinstance(model_cfg.pretrained_checkpoint, ListConfig):  # inference with partial ckpt
            pretrained_ckpt, inference_ckpt = model_cfg.pretrained_checkpoint
            mainlogger.info(f">>> Load weights from pretrained checkpoint {pretrained_ckpt} and {inference_ckpt}")
            pl_sd = smart_load(pretrained_ckpt)["state_dict"] | smart_load(inference_ckpt)
        else:
            pretrained_ckpt = model_cfg.pretrained_checkpoint
            mainlogger.info(f">>> Load weights from pretrained checkpoint {pretrained_ckpt}")
            pl_sd = smart_load(pretrained_ckpt)

            if "module" in pl_sd.keys():  # deepspeed
                pl_sd = pl_sd["module"]

            if "state_dict" in pl_sd.keys():  # ddp
                pl_sd = pl_sd["state_dict"]

        ## rename the keys for 256x256 model
        for k in list(pl_sd.keys()):
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                pl_sd[new_key] = pl_sd.pop(k)

        if check_config_attribute(model_cfg.params, "depth_predictor_config"):
            pl_sd |= {
                f"depth_predictor.{k}": v
                for k, v in smart_load(model_cfg.params.depth_predictor_config.pretrained_model_path).items()
            }
            pl_sd |= {
                "depth_predictor_normalizer_mean": torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1),
                "depth_predictor_normalizer_std": torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1),
            }

        model.load_state_dict(pl_sd)

    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
