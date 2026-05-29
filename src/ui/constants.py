ENVS = [
    "HalfCheetah-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Walker2d-v4",
    "Hopper-v4",
]

ENV_DESC = {
    "HalfCheetah-v4": "2D cheetah — forward locomotion, 17-dim state, 6-dim action",
    "Ant-v4": "4-legged ant — 3D locomotion, 27-dim state, 8-dim action",
    "Humanoid-v4": "Bipedal human — 3D, 376-dim state, 17-dim action (hardest)",
    "Walker2d-v4": "2D bipedal walker — 17-dim state, 6-dim action",
    "Hopper-v4": "1-legged hopper — 11-dim state, 3-dim action (easiest)",
}

HP_DEFAULTS = {
    "episodes": "2000",
    "lr": "0.0003",
    "gamma": "0.99",
    "gae_lambda": "0.95",
    "clip_eps": "0.2",
    "ppo_epochs": "10",
    "entropy_coef": "0.0",
    "value_coef": "0.5",
    "buffer_size": "2048",
    "minibatch_size": "64",
    "hidden_dims": "256,256",
    "grad_clip": "0.5",
    "normalize_adv": "1",
    "render_sleep": "0.005",
}
