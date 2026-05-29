import pytest

pytest.importorskip("torch")

import numpy as np

from src.rl import PPOAgent, parse_hidden_dims


class DummyActionSpace:
    def __init__(self, shape, high):
        self.shape = shape
        self.high = np.asarray(high, dtype=np.float32)


def test_parse_hidden_dims_basic():
    assert parse_hidden_dims("256,128, 64") == [256, 128, 64]


def test_parse_hidden_dims_empty():
    assert parse_hidden_dims(" , , ") == []


def test_ppo_agent_action_and_update():
    action_space = DummyActionSpace((3,), [1.0, 1.0, 1.0])
    agent = PPOAgent(
        state_dim=4,
        action_space=action_space,
        hidden_dims=[32, 32],
        buffer_size=4,
        minibatch_size=2,
        ppo_epochs=1,
    )
    state = np.zeros(4, dtype=np.float32)

    for _ in range(4):
        action, logp, value = agent.select_action_train(state)
        assert action.shape == (3,)
        assert isinstance(logp, float)
        assert isinstance(value, float)
        agent.add_transition(state, action, 1.0, False, logp, value)

    actor_loss, critic_loss = agent.update(state, False)
    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
