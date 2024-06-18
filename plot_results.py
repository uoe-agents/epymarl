import argparse
from collections import defaultdict
import json
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


ALPHA = 0.2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to directory containing (multiple) results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_return_mean",
        help="Metric to plot",
    )
    parser.add_argument(
        "--filter_by_algs",
        nargs="+",
        default=[],
        help="Filter results by algorithm names. Only showing results for algorithms that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--filter_by_envs",
        nargs="+",
        default=[],
        help="Filter results by environment names. Only showing results for environments that contain any of the specified strings in their names.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=Path.cwd(),
        help="Path to directory to save plots to",
    )
    parser.add_argument(
        "--y_min",
        type=float,
        default=None,
        help="Minimum value for y-axis",
    )
    parser.add_argument(
        "--y_max",
        type=float,
        default=None,
        help="Maximum value for y-axis",
    )
    parser.add_argument(
        "--log_scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=None,
        help="Smoothing window for data",
    )
    return parser.parse_args()


def extract_alg_name_from_config(config):
    return config["name"]


def extract_env_name_from_config(config):
    if "map" in config["env_args"]:
        env_name = config["env_args"]["map"]
    elif "key" in config["env_args"]:
        env_name = config["env_args"]["key"]
    else:
        env_name = None
    return env_name


def load_results(path, metric):
    path = Path(path)
    metrics_files = path.glob("**/metrics.json")

    data = defaultdict(list)
    for file in metrics_files:
        # load json
        with open(file, "r") as f:
            metrics = json.load(f)

        # find corresponding config file
        config_file = file.parent / "config.json"
        if not config_file.exists():
            warnings.warn(f"No config file found for {file} --> skipping")
            continue
        else:
            with open(config_file, "r") as f:
                config = json.load(f)

        if metric in metrics:
            steps = metrics[metric]["steps"]
            values = metrics[metric]["values"]
        elif "return" in metric and not config["common_reward"]:
            warnings.warn(
                f"Metric {metric} not found in {file}. To plot returns for runs with individual rewards (common_reward=False), you can plot 'total_return' metrics or returns of individual agents --> skipping"
            )
            continue
        else:
            warnings.warn(f"Metric {metric} not found in {file} --> skipping")
            continue
        del config["seed"]

        data[str(config)].append((config, steps, values))
    return data


def filter_results(data, filter_by_algs, filter_by_envs):
    filtered_data = data.copy()
    delete_keys = set()
    if filter_by_algs:
        for key, results in data.items():
            config = results[0][0]
            alg_name = extract_alg_name_from_config(config)
            if not any(alg in alg_name for alg in filter_by_algs):
                delete_keys.add(key)
    if filter_by_envs:
        for key, results in data.items():
            config = results[0][0]
            env_name = extract_env_name_from_config(config)
            if not any(env in env_name for env in filter_by_envs):
                delete_keys.add(key)
    for key in delete_keys:
        del filtered_data[key]
    return filtered_data


def aggregate_results(data):
    agg_data = defaultdict(list)
    for key, results in data.items():
        config = results[0][0]
        all_steps = [steps for _, steps, _ in results]
        all_values = [values for _, _, values in results]
        max_len = max([len(steps) for steps in all_steps])

        filtered_steps = []
        filtered_values = []
        for steps, values in zip(all_steps, all_values):
            if len(steps) < max_len:
                warnings.warn(
                    f"Length of steps ({len(steps)}) is less than max length ({max_len}) for run with key {key} --> skipping"
                )
                continue
            else:
                filtered_steps.append(steps)
                filtered_values.append(values)

        agg_steps = np.stack(filtered_steps).mean(axis=0)
        values = np.stack(filtered_values)
        means = values.mean(axis=0)
        stds = values.std(axis=0)
        agg_data[key] = (config, agg_steps, means, stds)
    return agg_data


def smooth_data(data, window_size):
    for key, results in data.items():
        config, steps, means, stds = results
        assert (
            len(steps) == len(means) == len(stds)
        ), "Lengths of steps, means, and stds should be the same for smoothing"
        smoothed_steps = []
        smoothed_means = []
        smoothed_stds = []
        for i in range(len(means) - window_size + 1):
            smoothed_steps.append(np.mean(steps[i : i + window_size]))
            smoothed_means.append(np.mean(means[i : i + window_size]))
            smoothed_stds.append(np.mean(stds[i : i + window_size]))
        data[key] = (
            config,
            np.array(smoothed_steps),
            np.array(smoothed_means),
            np.array(smoothed_stds),
        )
    return data


def group_data_by_task(data):
    grouped_data = defaultdict(dict)
    for results in data.values():
        config, steps, means, stds = results
        alg_name = extract_alg_name_from_config(config)
        env_name = extract_env_name_from_config(config)
        env_args = config["env_args"]
        common_reward = config["common_reward"]
        reward_scalarisation = config["reward_scalarisation"]
        grouped_data[(str(env_args), env_name, common_reward, reward_scalarisation)][
            alg_name
        ] = (steps, means, stds)
    return grouped_data


def plot_results(grouped_data, metric, save_dir, y_min, y_max, log_scale):
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    for (_, env, cr, rs), data in grouped_data.items():
        plt.figure()
        for alg_name, results in data.items():
            steps, means, stds = results
            plt.plot(steps, means, label=alg_name)
            plt.fill_between(steps, means - stds, means + stds, alpha=ALPHA)
        plt.title(f"Common Reward: {cr} (scalarisation: {rs})")
        plt.xlabel("Timesteps")
        plt.ylabel(metric)
        plt.legend()
        if log_scale:
            plt.yscale("log")
        if y_min is not None or y_max is not None:
            plt.ylim(y_min, y_max)
        if save_dir is not None:
            plt.savefig(save_dir / f"{env}_{metric}_{cr}.pdf")


def main():
    args = parse_args()
    data = load_results(args.path, args.metric)
    data = filter_results(data, args.filter_by_algs, args.filter_by_envs)
    data = aggregate_results(data)
    if args.smoothing_window is not None:
        data = smooth_data(data, args.smoothing_window)
    grouped_data = group_data_by_task(data)
    plot_results(
        grouped_data,
        args.metric,
        Path(args.save_dir),
        args.y_min,
        args.y_max,
        args.log_scale,
    )


if __name__ == "__main__":
    main()
