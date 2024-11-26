import random
import os
import logging

import numpy as np
import torch


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(path):
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(path)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] : %(message)s ", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def print_num_params(model):
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        mul_value = param.numel()
        total_params += mul_value
        if param.requires_grad:
            trainable_params += mul_value
        else:
            non_trainable_params += mul_value
    
    print(f'Total params: {total_params / 1e6}M')
    print(f'Trainable params: {trainable_params / 1e6}M')
    print(f'Non-trainable params: {non_trainable_params / 1e6}M')
    
    return total_params
