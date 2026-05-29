import gymnasium as gym
import numpy as np
import torch.nn as nn


def parse_hidden_dims(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_make_env(env_name: str, render_mode=None):
    kwargs = {}
    if render_mode:
        kwargs["render_mode"] = render_mode
    return gym.make(env_name, **kwargs)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer
