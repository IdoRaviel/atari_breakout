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
        self.observation_space = Box(low=0, high=1, shape=(84, 84), dtype=np.float32)
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
        
        # 4. Standard DQN Reward Clipping: All positive points = +1
        if self.clip_reward:
            reward = np.sign(reward)
        
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
        # 0-255 -> 0.0-1.0 for NN stability
        img = img.astype(np.float32) / 255.0
        return img

class FrameStack(gym.Wrapper):
    """
    LAYER 3 (Outer): History Wrapper.
    Collects 1-frame outputs from AtariPreprocessing into 4-frame 'video clips'.
    """
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = []
        shp = env.observation_space.shape
        # New space is (4, 84, 84)
        self.observation_space = Box(low=0, high=1, shape=(k, *shp), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Start of game: fill the 'history' with 4 identical copies of the first frame
        self.frames = [obs] * self.k
        return self._get_obs(), info

    def step(self, action):
        # 1. Request a processed frame from Layer 2 (AtariPreprocessing)
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 2. Update the Sliding Window: Remove oldest, add newest
        self.frames.pop(0)
        self.frames.append(obs)
        
        # 3. Return the full 4-frame stack to the Agent
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Combine 4 frames into a single (4, 84, 84) array."""
        return np.stack(self.frames, axis=0)

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
