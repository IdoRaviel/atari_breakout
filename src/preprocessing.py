import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation
import ale_py
import numpy as np
import cv2
from gymnasium.spaces import Box

class NoopResetEnv(gym.Wrapper):
    """Perform 1–noop_max random no-ops after FIRE to randomize the starting ball position.
    Placed after MaxAndSkipObservation and FireResetEnv: each no-op = 4 raw ALE frames.
    noop_max=7 -> up to 28 raw frames, matching the paper's ~30 raw frame budget.
    Must come after FireResetEnv so the ball is already moving during no-ops — otherwise
    no-ops on a static screen do nothing and the randomization is wasted.
    """
    def __init__(self, env, noop_max=7):
        super().__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        n_noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(n_noops):
            obs, _, terminated, truncated, info = self.env.step(0)  # NOOP
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE (action 1) after every reset to launch the ball.
    Required for Breakout: without FIRE the ball never moves and the episode stalls.
    Placed after MaxAndSkipObservation so the FIRE action runs with frame skip applied.
    """
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class AtariPreprocessing(gym.Wrapper):
    """
    LAYER 2: Image Processing Wrapper.
    Handles pixels (Grayscale, Resize, Crop) and Reward signal.
    """
    def __init__(self, env, clip_reward=True, terminal_on_life_loss=True):
        super().__init__(env)
        # Define the 'processed' view the Agent will see
        self.observation_space = Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)
        self.lives = 0
        self.clip_reward = clip_reward
        self.terminal_on_life_loss = terminal_on_life_loss

    def step(self, action):
        # 1. Pass action DOWN to Layer 1 (MaxAndSkip) -> then to Raw ALE
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 2. Terminal on life loss: Force 'terminated' so Agent learns to value 'life'
        # even if the game hasn't fully ended.
        current_lives = info.get('lives', 0)
        life_lost = current_lives < self.lives and current_lives > 0
        if life_lost:
            if self.terminal_on_life_loss:
                # Training: treat life loss as episode end for correct TD targets.
                terminated = True
            else:
                # Eval/inference: auto-inject FIRE so next ball launches without a reset.
                obs, _, _, _, info = self.env.step(1)  # FIRE
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

def make_env(render_mode=None, clip_reward=True, terminal_on_life_loss=True):
    """
    Wrapper stack (inner to outer):
      Raw ALE -> MaxAndSkip -> FireReset -> NoopReset -> AtariPreprocessing -> FrameStack

    MaxAndSkip is innermost so every action (FIRE, no-op, policy) = 4 raw ALE frames.
    FireReset launches the ball before NoopReset randomizes, so no-ops actually shift state.
    """
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=render_mode)
    env = MaxAndSkipObservation(env, skip=4)        # action repeat: 4 raw frames per action
    env = FireResetEnv(env)                         # launch ball on reset
    env = NoopResetEnv(env, noop_max=7)             # 1-7 no-ops while ball moves (~28 raw frames)
    env = AtariPreprocessing(env, clip_reward=clip_reward, terminal_on_life_loss=terminal_on_life_loss)
    env = FrameStack(env, k=4)
    return env
