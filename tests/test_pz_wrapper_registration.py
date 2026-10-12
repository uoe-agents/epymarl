"""Regression tests for PettingZoo environment registration.

See https://github.com/uoe-agents/epymarl/issues/78
"""

import importlib.util
import re
from pathlib import Path

import gymnasium as gym
import pettingzoo

PZ_WRAPPER = Path(__file__).resolve().parents[1] / "src" / "envs" / "pz_wrapper.py"


def _load_pz_wrapper():
    spec = importlib.util.spec_from_file_location(
        "pz_wrapper_under_test", PZ_WRAPPER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_versioned_pettingzoo_envs_are_registered():
    # Issue #78: file discovery used a single-character version glob
    # (`**/*_v?.py`), which silently dropped environments whose version
    # suffix has more than one digit (e.g. knights_archers_zombies_v11.py).
    _load_pz_wrapper()

    root = Path(pettingzoo.__path__[0])
    expected_keys = {
        f"pz-{path.parent.stem}-{path.stem.replace('_', '-')}"
        for path in root.rglob("*.py")
        if re.search(r"_v\d+\.py$", path.name)
    }

    registered = {k for k in gym.envs.registry.keys() if k.startswith("pz-")}

    missing = sorted(expected_keys - registered)
    assert not missing, f"PettingZoo envs not registered: {missing}"

    # Guard against over-matching: loose globs/regexes (e.g. `_v*.py`) would also
    # register non-env modules like `record_video.py` as `pz-*` environments.
    extra = sorted(registered - expected_keys)
    assert not extra, f"Unexpected extra PettingZoo envs registered: {extra}"
