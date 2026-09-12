import unittest
import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "runners" / "switching_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("switching_metrics", MODULE_PATH)
switching_metrics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(switching_metrics)

SWITCH_STATS_PREFIX = switching_metrics.SWITCH_STATS_PREFIX
SwitchingEpisodeMetrics = switching_metrics.SwitchingEpisodeMetrics
add_switching_stats = switching_metrics.add_switching_stats
is_lbf_env_key = switching_metrics.is_lbf_env_key
log_switching_stats = switching_metrics.log_switching_stats


class RecordingLogger:
    def __init__(self):
        self.values = {}

    def log_stat(self, key, value, _t_env):
        self.values[key] = value


class SwitchingEpisodeMetricsTest(unittest.TestCase):
    def test_switch_transition_belongs_to_post_window(self):
        metrics = SwitchingEpisodeMetrics(windows=(2, 3))
        metrics.observe(1.0, {"switching_lbf_switched": 0})
        metrics.observe(2.0, {"switching_lbf_switched": 1})
        metrics.observe(3.0, {"switching_lbf_switched": 1})
        metrics.observe(4.0, {"switching_lbf_switched": 1})

        self.assertEqual(metrics.pre_switch_return, 1.0)
        self.assertEqual(metrics.post_switch_returns[2], 5.0)
        self.assertEqual(metrics.post_switch_returns[3], 9.0)
        self.assertEqual(metrics.first_positive_after_switch_steps, 1)

    def test_no_positive_reward_is_censored(self):
        metrics = SwitchingEpisodeMetrics()
        metrics.observe(0.0, {"switching_lbf_switched": 1})
        metrics.observe(0.0, {"switching_lbf_switched": 1})
        stats = metrics.as_stats()

        self.assertEqual(stats[SWITCH_STATS_PREFIX + "no_positive_count"], 1)
        self.assertNotIn(SWITCH_STATS_PREFIX + "first_positive_sum", stats)

    def test_episode_ending_before_switch_is_reported(self):
        logger = RecordingLogger()
        stats = {}
        metrics = SwitchingEpisodeMetrics()
        metrics.observe(0.5, {"switching_lbf_switched": 0})
        add_switching_stats(stats, metrics)
        log_switching_stats(logger, stats, "test_", 0)

        self.assertEqual(logger.values["test_switch_reached_rate"], 0.0)
        self.assertEqual(
            logger.values["test_terminated_before_switch_rate"], 1.0
        )
        self.assertNotIn(
            "test_first_positive_after_switch_steps_observed_mean", logger.values
        )

    def test_observed_first_positive_uses_observed_denominator(self):
        logger = RecordingLogger()
        stats = {}
        recovered = SwitchingEpisodeMetrics()
        recovered.observe(0.0, {"switching_lbf_switched": 1})
        recovered.observe(1.0, {"switching_lbf_switched": 1})
        censored = SwitchingEpisodeMetrics()
        censored.observe(0.0, {"switching_lbf_switched": 1})
        add_switching_stats(stats, recovered)
        add_switching_stats(stats, censored)
        log_switching_stats(logger, stats, "test_", 0)

        self.assertEqual(
            logger.values[
                "test_first_positive_after_switch_steps_observed_mean"
            ],
            2.0,
        )
        self.assertEqual(logger.values["test_no_positive_after_switch_rate"], 0.5)

    def test_switching_lbf_is_counted_for_load_diagnostics(self):
        self.assertTrue(is_lbf_env_key("lbforaging:Foraging-v3"))
        self.assertTrue(is_lbf_env_key("epymarl/Switching-LBF-v0"))
        self.assertFalse(is_lbf_env_key("other:Env-v0"))


if __name__ == "__main__":
    unittest.main()
