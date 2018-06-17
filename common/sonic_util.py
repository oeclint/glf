import gym
from gym.spaces.box import Box

import numpy as np
from collections import deque

import os
import json

from glob import glob
from collections.abc import Sequence

class SonicObsWrapper(gym.ObservationWrapper):
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
    def __init__(self, env=None):
        super(SonicObsWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)
        
    def observation(self, observation):
        # move last axis to first
        return np.moveaxis(observation,-1,0)

class SonicActDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, actions = None):
        super(SonicActDiscretizer, self).__init__(env)

        if actions is None:

            buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
            actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                       ['DOWN', 'B'], ['B']]
            self._actions = []
            for action in actions:
                arr = np.array([False] * 12)
                for button in action:
                    arr[buttons.index(button)] = True
                self._actions.append(arr)
        else:
            self._actions = actions

        self._action_indexer = {}

        for i, a in enumerate(self._actions):
            self._action_indexer[tuple(a)] = i

        self.action_space = gym.spaces.Discrete(len(self._actions))

    @classmethod
    def from_human_data(cls, env, game, state, scenario, root='../play/human'):

        path = os.path.join(root,game,scenario)

        all_actions = []

        for fi in glob(os.path.join(path,'{}-{}.json'.format(game,state))):

            with open(fi) as f:
                data = json.load(f)

            for ep in data:

                arr = SonicActions(data[ep])
                data[ep] = arr

                all_actions+=arr.data.tolist()

        return cls(env, np.array(list(set(map(tuple, all_actions)))))

    def action(self, a):
        return self._actions[a].copy()

class SonicActions(Sequence):

    def __init__(self, arr,
                    buttons = None,
                    same_as = None,
                    inactive = None):

        super(SonicActions, self).__init__()

        if buttons is None:
            buttons = []

        if same_as is None:
            same_as = {}

        if inactive is None:
            inactive = []

        self.data = np.array(arr)

        same_as_ind = {}
        for button in same_as:
            key = buttons.index(button)
            same_as_ind[key] = []
            for same in same_as[button]:
                same_as_ind[key].append(buttons.index(same))

        inactive_ind = [buttons.index(button) for button in inactive]

        for i in inactive_ind:
            self.data[:, i] = 0

        for actions in self.data:
            for i in same_as_ind:
                if actions[same_as_ind[i]].any():
                    actions[same_as_ind[i]] = 0
                    actions[i] = 1

    @classmethod
    def from_sonic_config(cls, arr):
        return cls( arr,
                    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"],
                    same_as = {'B':['B','A','C']},
                    inactive = ["MODE", "START", "Y", "X", "Z"])
                    
    def pad_zeros(self,shape):
        result = np.zeros(shape,'int')
        result[:self.data.shape[0],:self.data.shape[1]]=self.data
        return result
        
    def group_by(self,by):
        r,c = self.data.shape
        data = self.data
        if r % by != 0:
            while r % by != 0:
                r = r+1
            data = self.pad_zeros((r,c))
        return data.reshape(int(r/by),by,c)

    def map(self, indexer):
        arr = []
        for i in self.data:
            val = indexer[tuple(i)]
            if isinstance(val, (Sequence, np.ndarray)):
                val = list(val)
            else:
                val = [val]
            arr.append(val)
        return SonicActions(arr)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class SonicActionsVec(object):

    def __init__(self,actions):

        self.actions = actions
        self.row = max([s.data.shape[0] for s in actions])
        self.col = actions[0].data.shape[1]

        if not all([s.data.shape[1]==self.col for s in actions]):
            raise ValueError('Expected all arrays to have same number of columns')

    def stack_by(self,by):

        arr = []
        for action in self.actions:
            if action.data.shape[0] < self.row:
                action = SonicActions(action.pad_zeros((self.row, self.col)))
            arr.append(action)

        return np.stack([a.group_by(by) for a in arr],axis=2)


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info
