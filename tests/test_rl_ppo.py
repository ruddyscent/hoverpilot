import unittest

try:
    import numpy as np
    from hoverpilot.rl.ppo import PPOConfig, PPOTrainer, reset_env_with_wait
    IMPORT_ERROR = None
except Exception as exc:
    np = None
    PPOConfig = None
    PPOTrainer = None
    reset_env_with_wait = None
    IMPORT_ERROR = exc


class ResetWaitEnv:
    def __init__(self):
        self.reset_calls = 0
        self.poll_calls = 0

    def reset(self):
        self.reset_calls += 1
        raise TimeoutError("trainer reset pending")

    def poll_wait_for_next_episode(self, action=None):
        self.poll_calls += 1
        observation = np.zeros(12, dtype=np.float32)
        info = {
            "debug_state": {
                "x_m": 0.0,
                "y_m": 0.0,
                "altitude_agl_m": 1.5,
            },
            "episode_start_reason": "trainer_repositioned",
        }
        return True, observation, info


class PPOTrainingModuleTests(unittest.TestCase):
    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_ppo_config_and_trainer_import(self):
        config = PPOConfig(timesteps=1, max_episode_steps=1)
        trainer = PPOTrainer(config)
        self.assertEqual(trainer.config.timesteps, 1)
        self.assertEqual(trainer.config.max_episode_steps, 1)

    @unittest.skipIf(IMPORT_ERROR is not None, f"RL dependencies unavailable: {IMPORT_ERROR}")
    def test_reset_wait_helper_recovers_via_polling(self):
        env = ResetWaitEnv()

        observation, info = reset_env_with_wait(env)

        self.assertEqual(env.reset_calls, 1)
        self.assertEqual(env.poll_calls, 1)
        self.assertEqual(observation.shape, (12,))
        self.assertEqual(info["episode_start_reason"], "trainer_repositioned")
