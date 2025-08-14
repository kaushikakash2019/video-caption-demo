import numpy as np

class FakeActionSpace:
    def sample(self):
        return None

class FakeEnv:
    def __init__(self, *args, **kwargs):
        self.max_episode_steps = 10
        self.step_count = 0
        self.action_space = FakeActionSpace()
        self.observation_shape = (64, 64, 3)

    def reset(self):
        self.step_count = 0
        return {"rgb": np.zeros(self.observation_shape, dtype=np.uint8)}

    def step(self, act):
        self.step_count += 1
        done = self.step_count >= self.max_episode_steps
        val = (self.step_count * 25) % 255
        obs = {"rgb": np.full(self.observation_shape, val, dtype=np.uint8)}
        return obs, 0.0, done, {}

    def close(self):
        pass

def create_env(env_name, obs_mode="rgb"):
    return FakeEnv()
