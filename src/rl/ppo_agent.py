import copy

import numpy as np
import torch
import torch.nn as nn

from .buffer import RolloutBuffer
from .networks import ActorNetwork, CriticNetwork, MultiHeadACNetwork


# PPO Agent
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_space,
        network_type="Separate",
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=10,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden_dims=None,
        minibatch_size=64,
        buffer_size=2048,
        grad_clip=0.5,
        normalize_adv=True,
    ):

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.minibatch_size = minibatch_size
        self.buffer_size = buffer_size
        self.grad_clip = grad_clip
        self.normalize_adv = normalize_adv
        self.network_type = network_type

        # Continuous action space (MuJoCo)
        action_dim = action_space.shape[0]

        if network_type == "MultiHead":
            self.multi = MultiHeadACNetwork(state_dim, action_dim, hidden_dims).to(
                self.device
            )
            self.actor = None
            self.critic = None
            self.optimizer = torch.optim.Adam(self.multi.parameters(), lr=lr, eps=1e-5)
        else:
            self.actor = ActorNetwork(state_dim, action_dim, hidden_dims).to(
                self.device
            )
            self.critic = CriticNetwork(state_dim, hidden_dims).to(self.device)
            self.multi = None
            self.optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=lr,
                eps=1e-5,
            )

        self.buffer = RolloutBuffer()
        self.action_scale = torch.tensor(action_space.high, dtype=torch.float32).to(
            self.device
        )

    # action selection
    def select_action_train(self, state):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.network_type == "MultiHead":
                dist, val = self.multi(s)
            else:
                dist = self.actor(s)
                val = self.critic(s)
            action = dist.sample()
            lp = dist.log_prob(action).sum(-1)
        action_np = action.squeeze(0).cpu().numpy()
        action_clipped = np.clip(action_np, -1.0, 1.0) * self.action_scale.cpu().numpy()
        return action_np, lp.item(), val.item()

    def select_action_eval(self, state, snapshot=None):
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        net = (
            snapshot
            if snapshot
            else (self.multi if self.network_type == "MultiHead" else self.actor)
        )
        with torch.no_grad():
            if self.network_type == "MultiHead":
                dist, _ = net(s)
            else:
                dist = net(s)
            mu = dist.loc
        return mu.squeeze(0).cpu().numpy()

    def add_transition(self, s, a, r, d, lp, v):
        self.buffer.add(s, a, r, d, lp, v)

    def ready_to_update(self):
        return len(self.buffer) >= self.buffer_size

    # GAE computation
    def _compute_gae(self, last_state, last_done):
        s = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(
            self.device
        )
        a = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(
            self.device
        )
        r = np.array(self.buffer.rewards, dtype=np.float32)
        d = np.array(self.buffer.dones, dtype=np.float32)
        lp = torch.tensor(self.buffer.log_probs, dtype=torch.float32).to(self.device)
        v = np.array(self.buffer.values, dtype=np.float32)

        ls = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.network_type == "MultiHead":
                _, lv = self.multi(ls)
            else:
                lv = self.critic(ls)
        last_val = 0.0 if last_done else lv.item()

        advantages = np.zeros_like(r)
        gae = 0.0
        for t in reversed(range(len(r))):
            nv = last_val if t == len(r) - 1 else v[t + 1]
            delta = r[t] + self.gamma * nv * (1 - d[t]) - v[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d[t]) * gae
            advantages[t] = gae
        returns = advantages + v

        adv_t = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32).to(self.device)
        if self.normalize_adv:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return s, a, lp, adv_t, ret_t

    # PPO update
    def update(self, last_state, last_done):
        s, a, old_lp, adv, ret = self._compute_gae(last_state, last_done)
        actor_losses, critic_losses = [], []
        n = len(s)

        for _ in range(self.ppo_epochs):
            idx = torch.randperm(n)
            for start in range(0, n, self.minibatch_size):
                mb = idx[start : start + self.minibatch_size]
                sb, ab, olb, advb, retb = s[mb], a[mb], old_lp[mb], adv[mb], ret[mb]

                if self.network_type == "MultiHead":
                    dist, vb = self.multi(sb)
                else:
                    dist = self.actor(sb)
                    vb = self.critic(sb)

                new_lp = dist.log_prob(ab).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = (new_lp - olb).exp()

                pg1 = ratio * advb
                pg2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advb
                actor_loss = -torch.min(pg1, pg2).mean()
                critic_loss = nn.functional.mse_loss(vb, retb)
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.multi.parameters()
                    if self.multi
                    else list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.grad_clip,
                )
                self.optimizer.step()

                actor_losses.append(actor_loss.item())
                critic_losses.append(critic_loss.item())

        self.buffer.clear()
        return (
            float(np.mean(actor_losses)) if actor_losses else None,
            float(np.mean(critic_losses)) if critic_losses else None,
        )

    # -- snapshot for eval thread --------------
    def build_snapshot(self):
        if self.network_type == "MultiHead":
            snap = copy.deepcopy(self.multi).eval()
        else:
            snap = copy.deepcopy(self.actor).eval()
        return snap

    def save(self, path):
        torch.save(
            {
                "network_type": self.network_type,
                "actor": self.actor.state_dict() if self.actor else None,
                "critic": self.critic.state_dict() if self.critic else None,
                "multi": self.multi.state_dict() if self.multi else None,
            },
            path,
        )

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        if self.actor and ckpt["actor"]:
            self.actor.load_state_dict(ckpt["actor"])
        if self.critic and ckpt["critic"]:
            self.critic.load_state_dict(ckpt["critic"])
        if self.multi and ckpt["multi"]:
            self.multi.load_state_dict(ckpt["multi"])
