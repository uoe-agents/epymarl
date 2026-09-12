import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_switching_recovery.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_switching_recovery", MODULE_PATH
)
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


class SummarizeSwitchingRecoveryTest(unittest.TestCase):
    def test_parse_and_aggregate_completed_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "belief__left_left__seed0.log").write_text(
                "test_return_mean: 0.12\t"
                "test_post_switch_return_10_mean: 0.04\n"
                "[INFO] pymarl Completed\n",
                encoding="utf-8",
            )
            rows = summarizer.parse_logs(log_dir)
            summaries = summarizer.aggregate(rows)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "belief")
        self.assertEqual(rows[0]["test_return_mean"], 0.12)
        self.assertEqual(summaries[0]["n_seeds"], 1)
        self.assertEqual(
            summaries[0]["test_post_switch_return_10_mean__mean"], 0.04
        )

    def test_failed_log_is_not_aggregated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            (log_dir / "local__same__seed0.log").write_text(
                "ValueError: broken\npymarl Completed\n", encoding="utf-8"
            )
            rows = summarizer.parse_logs(log_dir)

        self.assertEqual(summarizer.aggregate(rows), [])


if __name__ == "__main__":
    unittest.main()
