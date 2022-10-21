import random

import numpy as np
import torch


def to_np(array):
    return array.detach().cpu().numpy()


def to_torch(array, device):
    return torch.from_numpy(array).float().to(device)


def threshold_linear_fn(current_epoch, epoch_range, value_range):
    frac = (current_epoch - epoch_range[0]) / (epoch_range[1] - epoch_range[0])

    return int(
        min(
            max(
                value_range[0] + frac * (value_range[1] - value_range[0]),
                value_range[0],
            ),
            value_range[1],
        )
    )


def trunc_normal_init(m, mean=0.0, std=None, a=-2, b=2, bias=0.0):
    if type(m) == torch.nn.Linear:
        input_dim = m.in_features
        if std is None:
            torch.nn.init.trunc_normal_(
                m.weight, mean=mean, std=1 / (2 * np.sqrt(input_dim)), a=a, b=b
            )
        else:
            torch.nn.init.trunc_normal_(m.weight, mean=mean, std=std, a=a, b=b)
        torch.nn.init.constant_(m.bias, bias)


def xavier_init(m, gain=1.0, bias=0.0):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain)
        torch.nn.init.constant_(m.bias, bias)


def uniform_init(m, a=-1e-2, b=1e-2, bias=0.0):
    if type(m) == torch.nn.Linear:
        torch.nn.init.uniform_(m.weight, a=a, b=b)
        torch.nn.init.constant_(m.bias, bias)


# Ref: https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
