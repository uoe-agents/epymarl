"""Episode-level metrics for environments with a mid-episode policy switch."""

from dataclasses import dataclass, field

import numpy as np


SWITCH_STATS_PREFIX = "_switch_metric_"


def is_lbf_env_key(key):
    """Return whether ``key`` identifies stock LBF or the Switching-LBF wrapper."""
    return str(key).startswith(("lbforaging:", "epymarl/Switching-LBF"))


def _scalar_reward(reward):
    """Use team reward semantics for scalar and vector rewards alike."""
    return float(np.asarray(reward, dtype=np.float64).sum())


@dataclass
class SwitchingEpisodeMetrics:
    """Collect event-aligned metrics for one episode.

    The transition whose ``info`` first reports ``switching_lbf_switched == 1``
    is the first post-switch transition. Consequently, a positive reward on that
    transition has ``first_positive_after_switch_steps == 1``.
    """

    windows: tuple = (5, 10, 20)
    switch_reached: bool = False
    pre_switch_return: float = 0.0
    post_switch_steps: int = 0
    post_switch_positive_steps: int = 0
    first_positive_after_switch_steps: int = None
    post_switch_returns: dict = field(default_factory=dict)

    def __post_init__(self):
        self.windows = tuple(sorted({int(window) for window in self.windows}))
        if not self.windows or self.windows[0] <= 0:
            raise ValueError("Switching metric windows must be positive integers")
        self.post_switch_returns = {window: 0.0 for window in self.windows}

    def observe(self, reward, info):
        reward_value = _scalar_reward(reward)
        if not self.switch_reached and bool(
            (info or {}).get("switching_lbf_switched", 0)
        ):
            self.switch_reached = True

        if not self.switch_reached:
            self.pre_switch_return += reward_value
            return

        self.post_switch_steps += 1
        for window in self.windows:
            if self.post_switch_steps <= window:
                self.post_switch_returns[window] += reward_value

        if reward_value > 0:
            self.post_switch_positive_steps += 1
            if self.first_positive_after_switch_steps is None:
                self.first_positive_after_switch_steps = self.post_switch_steps

    def as_stats(self):
        """Return additive stats; derived rates are logged by ``log_switching_stats``."""
        stats = {
            SWITCH_STATS_PREFIX + "episodes": 1,
            SWITCH_STATS_PREFIX + "pre_return_sum": self.pre_switch_return,
            SWITCH_STATS_PREFIX + "reached_count": int(self.switch_reached),
        }
        if not self.switch_reached:
            return stats

        stats.update(
            {
                SWITCH_STATS_PREFIX + "post_steps_sum": self.post_switch_steps,
                SWITCH_STATS_PREFIX
                + "post_positive_rate_sum": self.post_switch_positive_steps
                / max(1, self.post_switch_steps),
                SWITCH_STATS_PREFIX
                + "no_positive_count": int(
                    self.first_positive_after_switch_steps is None
                ),
            }
        )
        for window, value in self.post_switch_returns.items():
            stats[SWITCH_STATS_PREFIX + f"post_return_{window}_sum"] = value
        if self.first_positive_after_switch_steps is not None:
            stats[SWITCH_STATS_PREFIX + "first_positive_sum"] = (
                self.first_positive_after_switch_steps
            )
            stats[SWITCH_STATS_PREFIX + "first_positive_count"] = 1
        return stats


def add_switching_stats(stats, episode_metrics):
    for key, value in episode_metrics.as_stats().items():
        stats[key] = stats.get(key, 0) + value


def log_switching_stats(logger, stats, prefix, t_env):
    """Log derived switching metrics and leave unrelated runner stats untouched."""
    episodes = stats.get(SWITCH_STATS_PREFIX + "episodes", 0)
    if not episodes:
        return

    reached = stats.get(SWITCH_STATS_PREFIX + "reached_count", 0)
    logger.log_stat(prefix + "switch_reached_rate", reached / episodes, t_env)
    logger.log_stat(
        prefix + "terminated_before_switch_rate", (episodes - reached) / episodes, t_env
    )
    logger.log_stat(
        prefix + "pre_switch_return_mean",
        stats.get(SWITCH_STATS_PREFIX + "pre_return_sum", 0) / episodes,
        t_env,
    )

    reached_denominator = max(1, reached)
    logger.log_stat(
        prefix + "post_switch_steps_mean",
        stats.get(SWITCH_STATS_PREFIX + "post_steps_sum", 0)
        / reached_denominator,
        t_env,
    )
    logger.log_stat(
        prefix + "post_switch_positive_rate",
        stats.get(SWITCH_STATS_PREFIX + "post_positive_rate_sum", 0)
        / reached_denominator,
        t_env,
    )
    logger.log_stat(
        prefix + "no_positive_after_switch_rate",
        stats.get(SWITCH_STATS_PREFIX + "no_positive_count", 0)
        / reached_denominator,
        t_env,
    )

    for key, value in sorted(stats.items()):
        if key.startswith(SWITCH_STATS_PREFIX + "post_return_") and key.endswith(
            "_sum"
        ):
            window = key[len(SWITCH_STATS_PREFIX + "post_return_") : -len("_sum")]
            logger.log_stat(
                prefix + f"post_switch_return_{window}_mean",
                value / reached_denominator,
                t_env,
            )

    observed = stats.get(SWITCH_STATS_PREFIX + "first_positive_count", 0)
    if observed:
        logger.log_stat(
            prefix + "first_positive_after_switch_steps_observed_mean",
            stats.get(SWITCH_STATS_PREFIX + "first_positive_sum", 0) / observed,
            t_env,
        )

