import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run import evaluate_sequential


class FakeRunner:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.run_calls = 0
        self.closed = False

    def run(self, test_mode=False):
        self.assert_test_mode = test_mode
        self.run_calls += 1

    def save_replay(self):
        raise AssertionError("save_replay should not be called")

    def close_env(self):
        self.closed = True


class EvaluateEpisodeCountTest(unittest.TestCase):
    def test_parallel_batch_does_not_multiply_test_episode_count(self):
        runner = FakeRunner(batch_size=10)
        args = SimpleNamespace(test_nepisode=20, save_replay=False)
        evaluate_sequential(args, runner)
        self.assertEqual(runner.run_calls, 2)
        self.assertTrue(runner.assert_test_mode)
        self.assertTrue(runner.closed)

    def test_episode_runner_still_runs_each_requested_episode(self):
        runner = FakeRunner(batch_size=1)
        args = SimpleNamespace(test_nepisode=20, save_replay=False)
        evaluate_sequential(args, runner)
        self.assertEqual(runner.run_calls, 20)


if __name__ == "__main__":
    unittest.main()
