"""Regression tests for the n-step returns used by the actor-critic learners.

With ``add_value_last_step=True``, the value of the state after the last
timestep in the batch is used to bootstrap the returns. This bootstrap must be
masked for episodes that ended before the last timestep of the batch, since
their remaining timesteps are only padding.
"""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from learners.actor_critic_learner import ActorCriticLearner  # noqa: E402
from learners.coma_learner import COMALearner  # noqa: E402
from learners.ppo_learner import PPOLearner  # noqa: E402

GAMMA = 0.99
NSTEPS = 5
T = 8  # number of transitions in the batch
EP_LENS = [T, 6]  # episode 0 runs until the end of the batch, episode 1 ends early


def _make_batch():
    n_agents = 2
    mask = th.zeros(len(EP_LENS), T, n_agents)
    for b, ep_len in enumerate(EP_LENS):
        mask[b, :ep_len] = 1
    rewards = th.ones(len(EP_LENS), T, n_agents) * mask
    values = th.full((len(EP_LENS), T + 1, n_agents), 0.5)
    # critic output on padded timesteps of episode 1 should never be used
    values[1, EP_LENS[1] + 1 :] = 100.0
    return rewards, mask, values


def _expected_return(b, t_start, rewards, values):
    ep_len = EP_LENS[b]
    ret = 0.0
    for step in range(NSTEPS + 1):
        t = t_start + step
        if t >= T or t >= ep_len:
            break
        if step == NSTEPS:
            ret += GAMMA**step * values[b, t, 0].item()
            break
        ret += GAMMA**step * rewards[b, t, 0].item()
        if t == T - 1:
            # last timestep of the batch: bootstrap from the next state
            ret += GAMMA ** (step + 1) * values[b, t + 1, 0].item()
    return ret


@pytest.mark.parametrize("learner_cls", [ActorCriticLearner, PPOLearner, COMALearner])
def test_last_step_bootstrap_is_masked(learner_cls):
    rewards, mask, values = _make_batch()
    learner = SimpleNamespace(
        args=SimpleNamespace(gamma=GAMMA, add_value_last_step=True)
    )
    returns = learner_cls.nstep_returns(learner, rewards, mask, values, NSTEPS)

    for b, ep_len in enumerate(EP_LENS):
        for t_start in range(ep_len):
            expected = _expected_return(b, t_start, rewards, values)
            assert returns[b, t_start, 0].item() == pytest.approx(expected), (
                f"episode {b}, t={t_start}"
            )
