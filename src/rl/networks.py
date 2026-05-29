import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .utils import layer_init


# ----------------------------------------------
# Networks (continuous / MuJoCo)
# ----------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden_dims:
            layers += [layer_init(nn.Linear(in_d, h)), nn.Tanh()]
            in_d = h
        self.net = nn.Sequential(*layers)
        self.mu_head = layer_init(nn.Linear(in_d, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def get_log_prob(self, x, actions):
        dist = self(x)
        return dist.log_prob(actions).sum(-1)

    def get_entropy(self, x):
        return self(x).entropy().sum(-1)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden_dims:
            layers += [layer_init(nn.Linear(in_d, h)), nn.Tanh()]
            in_d = h
        layers += [layer_init(nn.Linear(in_d, 1), std=1.0)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MultiHeadACNetwork(nn.Module):
    """Shared backbone, separate heads."""

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        layers, in_d = [], state_dim
        for h in hidden_dims:
            layers += [layer_init(nn.Linear(in_d, h)), nn.Tanh()]
            in_d = h
        self.shared = nn.Sequential(*layers)
        self.mu_head = layer_init(nn.Linear(in_d, action_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.v_head = layer_init(nn.Linear(in_d, 1), std=1.0)

    def forward(self, x):
        h = self.shared(x)
        mu = self.mu_head(h)
        std = self.log_std.exp().expand_as(mu)
        v = self.v_head(h).squeeze(-1)
        return Normal(mu, std), v
