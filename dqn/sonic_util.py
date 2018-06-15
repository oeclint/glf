import gym
import numpy as np
from collections import deque

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

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return process_obs(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        # Don't punish too much for going back
        if rew < 0:
            rew = rew * 0.33
        return process_obs(obs), rew, done, info

class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
