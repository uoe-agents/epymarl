import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from envs.switching_lbf import SwitchingLBFEnv, register_switching_lbf
    from gymnasium.envs.registration import registry
except ModuleNotFoundError as error:
    SwitchingLBFEnv = None
    IMPORT_ERROR = error
else:
    IMPORT_ERROR = None


@unittest.skipIf(SwitchingLBFEnv is None, f"LBF dependencies unavailable: {IMPORT_ERROR}")
class SwitchingLBFFixedTest(unittest.TestCase):
    def test_intent_registrations_use_coop_environment(self):
        register_switching_lbf()
        expected = {
            "epymarl/Switching-LBF-Intent-v1": (False, False, False),
            "epymarl/Switching-LBF-TypeOracle-Intent-v1": (True, False, False),
            "epymarl/Switching-LBF-LastAction-Intent-v1": (False, True, False),
            "epymarl/Switching-LBF-Belief-Intent-v1": (False, True, False),
            "epymarl/Switching-LBF-Intent-v2": (False, False, True),
            "epymarl/Switching-LBF-TypeOracle-Intent-v2": (True, False, True),
            "epymarl/Switching-LBF-LastAction-Intent-v2": (False, True, True),
            "epymarl/Switching-LBF-Belief-Intent-v2": (False, True, True),
        }
        for env_id, expected_flags in expected.items():
            kwargs = registry[env_id].kwargs
            self.assertIn("-coop-v3", kwargs["base_key"])
            self.assertTrue(kwargs["use_load_positions"])
            self.assertEqual(
                (
                    kwargs.get("reveal_teammate_modes", False),
                    kwargs.get("include_teammate_last_actions", False),
                    kwargs.get("shared_teammate_mode", False),
                ),
                expected_flags,
            )

    def test_coordinated_modes_stay_shared_across_switch(self):
        env = SwitchingLBFEnv(
            base_key="lbforaging:Foraging-2s-10x10-3p-3f-coop-v3",
            teammate_modes=("left_priority", "right_priority", "wait"),
            use_load_positions=True,
            shared_teammate_mode=True,
            fixed_switch_step=1,
            seed=0,
        )
        try:
            for episode in range(20):
                env.reset(seed=episode)
                initial_mode = env._current_modes[0]
                self.assertEqual(env._current_modes[0], env._current_modes[1])
                env.step([0])
                env.step([0])
                self.assertEqual(env._current_modes[0], env._current_modes[1])
                self.assertNotEqual(env._current_modes[0], initial_mode)
        finally:
            env.close()

    def test_fixed_teammates_load_from_adjacent_cells(self):
        env = SwitchingLBFEnv(
            use_load_positions=True,
            initial_mode_ids=(0, 0),
            switch_mode_ids=(1, 1),
            fixed_switch_step=10,
            seed=0,
        )
        teammate_loads = 0
        try:
            for episode in range(20):
                env.reset(seed=episode)
                terminated = truncated = False
                while not (terminated or truncated):
                    actions = [
                        env._teammate_action(1, env._current_modes[0]),
                        env._teammate_action(2, env._current_modes[1]),
                    ]
                    for player_id, action in zip((1, 2), actions):
                        if action == env._action_size - 1:
                            teammate_loads += 1
                            player_pos = tuple(
                                int(v) for v in env.base_env.players[player_id].position
                            )
                            self.assertTrue(
                                any(
                                    env._manhattan(player_pos, food) == 1
                                    for food in env._food_positions()
                                )
                            )
                    _, _, terminated, truncated, _ = env.step([0])
        finally:
            env.close()

        self.assertGreater(teammate_loads, 0)

    def test_legacy_v0_behavior_remains_reproducible(self):
        env = SwitchingLBFEnv(
            use_load_positions=False,
            initial_mode_ids=(0, 0),
            switch_mode_ids=(1, 1),
            fixed_switch_step=10,
            seed=0,
        )
        teammate_loads = 0
        try:
            for episode in range(5):
                env.reset(seed=episode)
                terminated = truncated = False
                while not (terminated or truncated):
                    actions = [
                        env._teammate_action(1, env._current_modes[0]),
                        env._teammate_action(2, env._current_modes[1]),
                    ]
                    teammate_loads += sum(
                        action == env._action_size - 1 for action in actions
                    )
                    _, _, terminated, truncated, _ = env.step([0])
        finally:
            env.close()

        self.assertEqual(teammate_loads, 0)


if __name__ == "__main__":
    unittest.main()
