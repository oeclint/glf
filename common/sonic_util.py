import gym
from gym.spaces.box import Box

import numpy as np
from collections import deque

import os
import json

from glob import glob
from collections.abc import Sequence

from retro_contest.local import make as contest_make
from retro import make
from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

import torch

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

def actions_from_human_data(game_state, scenario='scenario', play_path='../play/human', order=True):

    all_actions = []
    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    actions = {}

    for game, state in set(game_state):

        path = os.path.join(play_path,game,scenario)

        fi = os.path.join(path,'{}-{}.json'.format(game,state))

        with open(fi) as f:
            actions[(game,state)] = json.load(f)

        for ep in actions[(game,state)]:

            arr = SonicActions.from_sonic_config(actions[(game,state)][ep])
            actions[(game,state)][ep] = arr

            all_actions+=arr.data.tolist()

    unique_actions = [np.array(a, 'int') for a in list(set(map(tuple, all_actions)))]

    if order:

        left = [0] * 12
        left[buttons.index("LEFT")] = 1
        left = tuple(left)
        right = [0] * 12
        right[buttons.index("RIGHT")] = 1
        right = tuple(right)
        jump = [0] * 12
        jump[buttons.index("B")] = 1
        jump = tuple(jump)

        core_actions = tuple([left, right, jump])
        actions_tup = tuple(map(tuple, unique_actions))

        for core_action in core_actions:
            if core_action not in actions_tup:
                raise ValueError("core actions missing in action set")

        sort_index = []

        for act in actions_tup:
            if act in core_actions:
                sort_index.append(core_actions.index(act))
            else:
                sort_index.append(len(sort_index)+len(core_actions))

        unique_actions = [np.array(a) for _,a in sorted(zip(sort_index,actions_tup))]

    return unique_actions, actions

class SonicActDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, actions = None):
        super(SonicActDiscretizer, self).__init__(env)

        if actions is None:

            buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

            sonic_actions = [['LEFT'], ['RIGHT'], ['B'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                       ['DOWN', 'B']]
            actions = []
            for action in sonic_actions:
                arr = np.array([0] * 12)
                for button in action:
                    arr[buttons.index(button)] = 1
                actions.append(arr)

        self._actions = actions

        self.action_space = gym.spaces.Discrete(len(self._actions))

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
            data = self.tile((r,c))
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

    def tile(self, shape):
        tile_len = len(self.data)
        arr = self.pad_zeros(shape)
        r,c = shape
        for i in range(0, r, tile_len):
            if i != 0:
                slc = arr[i:i+tile_len]
                arr[i:i+len(slc)] = arr[0:len(slc)]

        return arr
        
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

        max_row = self.row

        if max_row % by != 0:
            while max_row % by != 0:
                max_row = max_row+1

        arr = []
        for action in self.actions:
            if action.data.shape[0] < max_row:
                action = SonicActions(action.tile((max_row, self.col)))
            arr.append(action)

        return np.stack([a.group_by(by) for a in arr],axis=2)

    def discretize(self, actions):

        action_indexer = {}

        for i, a in enumerate(actions):
            action_indexer[tuple(a)] = i

        return SonicActionsVec([action.map(action_indexer) for action in self.actions])

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

def make_env(game, state, seed, rank, log_dir=None, scenario=None, actions=None):

    def _thunk():

        if scenario is None:
            env = contest_make(game,state)
        else:
            env = make(game, state, scenario=scenario)

        env.seed(seed + rank)

        if log_dir is not None:
            log_path = os.path.join(log_dir, state)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            env = bench.Monitor(env, os.path.join(log_path, str(rank)), allow_early_resets=True)

        env = SonicObsWrapper(env)
        env = AllowBacktracking(env)
        env = SonicActDiscretizer(env, actions)

        return env

    return _thunk

def make_envs(game_state, seed = 1, log_dir=None, scenario = None, actions=None):

    num_processes = len(game_state)

    envs = [make_env(game_state[i][0], game_state[i][1], seed, i, log_dir, scenario, actions)
            for i in range(num_processes)]

    if num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    return envs

def update_current_obs(current_obs,obs,envs,num_stack):
    
    shape_dim0 = envs.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs
