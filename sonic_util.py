import gym
import numpy as np

def process_obs(obs):
    """
    Each timestep advances the game by 4 frames, and each observation 
    is the pixels on the screen for the current frame, a shape (224, 320, 3) 
    array of uint8 values.

    Where:
    * Height of input planes in pixels = 224
    * Width in pixels = 320
    * Number of channels = 3

    However, each conv2d layers expects inputs (observations) of shape (N, Cin, H, W).

    Where:
    * N  is a batch size
    * C denotes a number of channels
    * H is a height of input planes in pixels
    * W is width in pixels

    Therefore, the axes of the observation pixel array need to be re-arranged.
    """

    # add new axis at the beginning (N)
    obs = obs[np.newaxis]
    # move last axis (C) to the second position
    obs = np.moveaxis(obs,-1,1)

    return obs

class SonicEnvWrapper(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(SonicEnvWrapper, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        obs = self.env.reset(**kwargs)
        return process_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0.0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return process_obs(obs), rew, done, info
