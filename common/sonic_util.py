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
from glf.common.containers import OrderedSet

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

    unique_actions = [np.array(a, 'int') for a in set(map(tuple, all_actions))]

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

class EnvRecorder(gym.Wrapper):
    """
    Record gym environment every n episodes
    """
    def __init__(self, env, record_dir, interval):
        super(EnvRecorder, self).__init__(env)
        self.record_dir = record_dir
        self.interval = interval
        self.ep_count = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        if self.ep_count % self.interval == 0:
            self.env.unwrapped.movie_id = self.ep_count
            self.env.unwrapped.auto_record(self.record_dir)
        else:
            self.env.unwrapped.stop_record()
        
        obs = self.env.reset(**kwargs)
        self.ep_count += 1

        return obs

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        return obs, rew, done, info

class HumanPlay(gym.Wrapper):
    """
    Record gym environment every n episodes
    """
    def __init__(self, env, actions):

        super(HumanPlay, self).__init__(env)

        self.human_actions = []

        action_indexer = {}

        for i, a in enumerate(env._actions):
            action_indexer[tuple(a)] = i

        for a in actions:
            val = action_indexer[tuple(a)]
            self.human_actions.append(val)

        self.human_step = 0
        self.rews = []

    @property
    def curr_action(self):
        if self.human_step < len(self.human_actions):
            return self.human_actions[self.human_step]
        else:
            return None

    @property
    def prev_action(self):
        if self.human_step-1>=0:
            return self.human_actions[self.human_step-1]
        else:
            return None

    @property
    def next_action(self):
        if self.human_step + 1 < len(self.human_actions):
            return self.human_actions[self.human_step + 1]
        else:
            return None

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        self.human_step = 0
        self.rews = []

        return obs

    def step(self, action=None, exceeds_human_steps = False):

        if action is None:
            action = self.curr_action
        
        obs, rew, done, info = self.env.step(action)
        info['action'] = self.curr_action

        if not exceeds_human_steps:
            # done if more steps than human
            if self.next_action is None:
                done = True

        self.human_step += 1
        self.rews.append(rew)

        return obs, rew, done, info

    def fast_forward(self, steps):
        for i in range(steps):
            obs, rew, done, info = self.step()


class ReversePlay(gym.Wrapper):
    """
    Record gym environment every n episodes
    """
    def __init__(self, env, interval):

        super(ReversePlay, self).__init__(env)

        self.step_backward = 1

        done = False
        i = 0
        self._chk_point = []

        self.env.reset()

        while not done:
            if i % interval == 0:
                    self._chk_point.append((i, self.env.unwrapped.em.get_state()))

            obs, rew, done, info = self.env.step()
                    
            i += 1

    def reset(self, **kwargs):
        
        if self.step_backward <= len(self._chk_point):
            step, state = self._chk_point[-1*self.step_backward]
        else:
            step, state = self._chk_point[0]

        obs = self.env.reset(**kwargs)

        self.env.unwrapped.em.set_state(state)

        self.env.human_step = step

        return obs

    def step(self, action=None, rew_goal = 9000):
        
        obs, rew, done, info = self.env.step(action)

        if sum(self.env.rews) >= rew_goal:
            # only step backward when beats level
            self.step_backward+=1

        return obs, rew, done, info

class EnvMaker(object):
    def __init__(self, game_state, num_processes, actions=None, human_actions=None, scenario=None, 
            log_dir='log', record_dir='bk2s', record_interval=10, order_actions=True):

        chunks = [num_processes//len(game_state)]*len(game_state)
        chunks[-1]+=num_processes-sum(chunks)

        processes = []

        for i, size in enumerate(chunks):
            for j in range(size):
                processes.append(game_state[i])

        self.processes = tuple(processes)

        seed = 1

        if human_actions is not None:
            # supervised actions per process
            all_actions = []
            for arr in human_actions:
                all_actions+=arr

            action_set = [np.array(a, 'int') for a in set(map(tuple, all_actions))]

            if order_actions:
                # order key actions, helps with debugging
                buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
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
                actions_tup = tuple(map(tuple, action_set))

                for core_action in core_actions:
                    if core_action not in actions_tup:
                        raise ValueError("core actions missing in action set")

                sort_index = []

                for act in actions_tup:
                    if act in core_actions:
                        sort_index.append(core_actions.index(act))
                    else:
                        sort_index.append(len(sort_index)+len(core_actions))

                action_set = [np.array(a) for _,a in sorted(zip(sort_index,actions_tup))]

            envs = [_make_env(processes[i][0], processes[i][1], seed, i, 
                        log_dir, scenario, action_set, human_actions[i], record_dir, record_interval)
                            for i in range(num_processes)]

        elif actions is not None:
            # unsupervised
            action_set = [np.array(a, 'int') for a in OrderedSet(map(tuple, actions))]

            envs = [_make_env(processes[i][0], processes[i][1], seed, i, 
                        log_dir, scenario, action_set, None, record_dir, record_interval)
                            for i in range(num_processes)]


        else:
            action_set = None
            envs = [_make_env(processes[i][0], processes[i][1], seed, i, 
                        log_dir, scenario, action_set, None, record_dir, record_interval)
                            for i in range(num_processes)]

        self.action_set = action_set
        self.vec_env = make_vec_env(envs)

    @classmethod
    def from_human_play(cls, game_state, play_path, scenario='contest', log_dir='log_human', 
            record_dir='supervised_bk2s', record_interval=10):

        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions_map = {}

        for game, state in set(game_state):

            path = os.path.join(play_path,game,scenario)

            fi = os.path.join(path,'{}-{}.json'.format(game,state))

            with open(fi) as f:
                actions_map[(game,state)] = json.load(f)

            for ep in actions_map[(game,state)]:

                arr = SonicActions.from_sonic_config(actions_map[(game,state)][ep])
                actions_map[(game,state)][ep] = arr.data.tolist()

        sonic_actions = []
        processes = []

        for k in actions_map:
            for ep in actions_map[k]:
                sonic_actions.append(actions_map[k][ep])
                processes.append(k)

        num_processes = len(processes)
        return cls(processes, num_processes, None, sonic_actions, scenario, log_dir, record_dir, record_interval)


def _make_env(game, state, seed, rank, log_dir=None, scenario=None, action_set=None, 
        actions=None,record_dir=None, record_interval=10000):

    def _thunk():

        if scenario is None:
            env = contest_make(game,state)
        else:
            env = make(game, state, scenario=scenario)

        env.seed(seed + rank)

        if record_dir is not None:
            record_path = os.path.join(record_dir,str(rank))
            os.makedirs(record_path, exist_ok=True)
            env = EnvRecorder(env, record_path, record_interval)

        if log_dir is not None:
            log_path = os.path.join(log_dir, state)
            os.makedirs(log_path, exist_ok=True)
            env = bench.Monitor(env, os.path.join(log_path, str(rank)), allow_early_resets=True)

        env = SonicObsWrapper(env)
        env = AllowBacktracking(env)
        env = SonicActDiscretizer(env, action_set)
        if actions is not None:
            env = HumanPlay(env, actions)
            env = ReversePlay(env, 500)

        return env

    return _thunk

def make_vec_env(envs):

    num_processes = len(envs)

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

if __name__ == "__main__":

    maker = EnvMaker.from_human_play(game_state=[('SonicTheHedgehog-Genesis','GreenHillZone.Act1'),
         ('SonicTheHedgehog-Genesis','GreenHillZone.Act3')], play_path='../../glf/play/human', scenario='contest', log_dir=None,
        record_dir=None,record_interval=None)
    envs = maker.vec_env

    envs.reset()
    while True:
        _obs, _rew, done, _info = envs.step(np.random.randint(0,10,envs.num_envs))
        envs.render()
