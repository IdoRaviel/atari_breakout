import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation
import ale_py
import numpy as np
import cv2
from gymnasium.spaces import Box


class AtariPreprocessing(gym.Wrapper):
    def __init__(self, env, clip_reward=True):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=(84, 84), dtype=np.float32)
        self.lives = 0
        self.clip_reward = clip_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = info.get('lives', 0)
        if current_lives < self.lives and current_lives > 0:
            terminated = True
        self.lives = current_lives

        obs = self._preprocess(obs)

        if self.clip_reward:
            reward = np.clip(reward, -1, 1)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        return self._preprocess(obs), info

    def _preprocess(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        img = img[18:102, :]
        img = img.astype(np.float32) / 255.0
        return img


class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=1, shape=(k, *shp), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.stack(self.frames, axis=0)


def make_env(render_mode=None, clip_reward=True):
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)
    env = AtariPreprocessing(env, clip_reward=clip_reward)
    env = FrameStack(env, k=4)
    return env
