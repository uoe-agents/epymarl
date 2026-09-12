"""Summarize Switching-LBF recovery-evaluation logs without extra dependencies."""

import argparse
import csv
import re
import statistics
from pathlib import Path


METRICS = (
    "test_return_mean",
    "test_pre_switch_return_mean",
    "test_post_switch_return_5_mean",
    "test_post_switch_return_10_mean",
    "test_post_switch_return_20_mean",
    "test_post_switch_positive_rate",
    "test_first_positive_after_switch_steps_observed_mean",
    "test_no_positive_after_switch_rate",
    "test_switch_reached_rate",
    "test_terminated_before_switch_rate",
    "test_load_actions_mean",
)
LOG_NAME = re.compile(
    r"^(?P<method>local|oracle|last_action|belief)"
    r"__(?P<condition>same|left_left|right_right|wait_wait|left_wait|right_wait)"
    r"__seed(?P<seed>[0-2])\.log$"
)


def last_metric(text, name):
    matches = re.findall(rf"(?:^|\s){re.escape(name)}:\s+([-+0-9.eE]+)", text)
    return float(matches[-1]) if matches else None


def parse_logs(log_dir):
    rows = []
    for path in sorted(log_dir.glob("*.log")):
        name_match = LOG_NAME.match(path.name)
        if not name_match:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        row = {
            **name_match.groupdict(),
            "completed": "pymarl Completed" in text,
            "has_error": "Traceback" in text or "ValueError" in text,
        }
        row.update({metric: last_metric(text, metric) for metric in METRICS})
        rows.append(row)
    return rows


def aggregate(rows):
    grouped = {}
    for row in rows:
        if not row["completed"] or row["has_error"]:
            continue
        grouped.setdefault((row["method"], row["condition"]), []).append(row)

    output = []
    for (method, condition), group in sorted(grouped.items()):
        summary = {"method": method, "condition": condition, "n_seeds": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group if row[metric] is not None]
            summary[metric + "__mean"] = statistics.mean(values) if values else None
            summary[metric + "__sd"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        output.append(summary)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = parse_logs(args.log_dir)
    summaries = aggregate(rows)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.log_dir / "summary_by_run.csv", rows)
    write_csv(args.log_dir / "summary_by_method_condition.csv", summaries)

    completed = sum(row["completed"] and not row["has_error"] for row in rows)
    print(f"Parsed {len(rows)} logs; {completed} completed without detected errors.")
    for row in summaries:
        return_mean = row.get("test_return_mean__mean")
        post10 = row.get("test_post_switch_return_10_mean__mean")
        print(
            f"{row['method']:>11} {row['condition']:<10} "
            f"n={row['n_seeds']} return={return_mean!s:<10} post10={post10!s}"
        )


if __name__ == "__main__":
    main()
