# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List
import torch

from detectron2.config import CfgNode

from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params: List[Dict[str, Any]] = []
    lr = cfg.SOLVER.BASE_LR

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if key.endswith("norm.weight") or key.endswith("norm.bias"):
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
        elif key.endswith(".bias"):
            # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
            # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
            # hyperparameters are by default exactly the same as for regular
            # weights.
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER=="SGD":
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM,nesterov=cfg.SOLVER.NESTEROV)
    elif cfg.SOLVER.OPTIMIZER=="Adam":
        optimizer = torch.optim.Adam(params, lr)
    elif cfg.SOLVER.OPTIMIZER=="ADAMW":
        optimizer = torch.optim.AdamW(params, lr)
    else:
        optimizer = torch.optim.SGD(params, lr, momentum=cfg.SOLVER.MOMENTUM,nesterov=cfg.SOLVER.NESTEROV)
    return optimizer


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer, last_iter = -1
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.SOLVER.MAX_ITER,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
            last_epoch=last_iter
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
