import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation
import ale_py
import numpy as np
import cv2
from gymnasium.spaces import Box

class AtariPreprocessing(gym.Wrapper):
    """
    LAYER 2: Image Processing Wrapper.
    Handles pixels (Grayscale, Resize, Crop) and Reward signal.
    """
    def __init__(self, env, clip_reward=True):
        super().__init__(env)
        # Define the 'processed' view the Agent will see
        self.observation_space = Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.lives = 0
        self.clip_reward = clip_reward

    def step(self, action):
        # 1. Pass action DOWN to Layer 1 (MaxAndSkip) -> then to Raw ALE
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Terminal on life loss: Force 'terminated' so Agent learns to value 'life'
        # even if the game hasn't fully ended.
        current_lives = info.get('lives', 0)
        if current_lives < self.lives and current_lives > 0:
            terminated = True
        self.lives = current_lives

        # 3. Transform pixels from (210, 160, 3) RGB to (84, 84) Grayscale
        obs = self._preprocess(obs)
        
        # 4. Reward shaping: sqrt normalization preserves relative brick value
        #    (red/orange=7pts, yellow/green=4pts, aqua/blue=1pt) -> range [0, 1]
        #    instead of paper's clipping which treats all bricks equally
        if self.clip_reward:
            reward = np.clip(reward, -1, 1)
        
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = info.get('lives', 0)
        return self._preprocess(obs), info

    def _preprocess(self, frame):
        """Math for reducing Atari complexity."""
        # RGB -> Gray
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Downsample to 110x84
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        # Crop 84x84 from the bottom playing area (removes score)
        img = img[18:102, :]
        return img  # uint8 (0-255); normalization happens at sample time

class FrameStack(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(k, *shp), dtype=np.uint8)
        self._obs_buf = np.zeros((k, *shp), dtype=np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.k):
            self._obs_buf[i] = obs
        return self._obs_buf.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs_buf[:-1] = self._obs_buf[1:]  # shift frames left (drop oldest)
        self._obs_buf[-1] = obs                  # write newest at the end
        return self._obs_buf.copy(), reward, terminated, truncated, info

def make_env(render_mode=None, clip_reward=True):
    """
    THE WRAPPER ONION (Inner to Outer):
    ALE (Raw Game) -> MaxAndSkip (L1) -> AtariPreprocessing (L2) -> FrameStack (L3)
    """
    # 0. Raw ALE Environment
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    
    # 1. Action Repeat & Flicker Fix (takes 2 frames, returns 1 maxed observation)
    env = MaxAndSkipObservation(env, skip=4)
    
    # 2. Custom Grayscale/Resizing/Life-Loss logic
    env = AtariPreprocessing(env, clip_reward=clip_reward)
    
    # 3. Temporal stacking (Final output shape: 4x84x84)
    env = FrameStack(env, k=4)
    
    return env
