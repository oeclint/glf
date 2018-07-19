import gym
from gym.spaces.box import Box

import numpy as np
from collections import deque

import os
import json

from glob import glob
from collections.abc import Sequence

import gym
from retro_contest.local import make as contest_make
from retro_contest import StochasticFrameSkip
from retro import make

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from glf.common.containers import OrderedSet

import torch
from enum import Enum

class SonicConfig(Enum):
    BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    BUTTONS_SAME_AS = {'B':['B','A','C']},
    BUTTONS_INACTIVE = ["MODE", "START", "Y", "X", "Z"]
    DEFAULT_ACTIONS = [['LEFT'], ['RIGHT'], ['B'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'], ['DOWN', 'B']]
    OBS_SHAPE = (3, 300, 200)

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

        if actions is None or len(actions) == 0:

            buttons = SonicConfig.BUTTONS.value

            sonic_actions = SonicConfig.DEFAULT_ACTIONS.value

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
            self.env.unwrapped.set_movie_id(self.ep_count)
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

        self.human_actions = actions
        self.human_step = 0
 
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

        return obs

    def step(self, action = None, exceeds_human_steps = True):

        action = self.curr_action
        
        obs, rew, done, info = self.env.step(action)

        if not exceeds_human_steps:
            # done if more steps than human
            if self.next_action is None:
                done = True

        self.human_step += 1

        return obs, rew, done, info

    def fast_forward(self, steps):
        for i in range(steps):
            obs, rew, done, info = self.step()

class StochasticHumanPlay(gym.Wrapper):
    def __init__(self, env, henv, humanprob):
        super(StochasticHumanPlay, self).__init__(env)
        self._humanprob = humanprob
        self._is_human = False
        self.henv = henv
        self.rng = np.random.RandomState()
        env.unwrapped.set_stoch_env(self)

    def reset(self, **kwargs):

        self._is_human = False
        if self.rng.rand() < self._humanprob:
            self._is_human = True

        if self._is_human:
            obs = self.henv.reset(**kwargs)
        else:
            obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        if self._is_human:
            obs, rew, done, info = self.henv.step(None)
        else:
            obs, rew, done, info = self.env.step(action)

        return obs, rew, done, info

    @property
    def is_human(self):
        return self._is_human

class RetroWrapper(gym.Wrapper):

    def __init__(self, env, pid=None):
        super(RetroWrapper, self).__init__(env)
        self._stoch_env = None
        self.pid = pid

    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        
        obs, rew, done, info = self.env.step(action)

        return obs, rew, done, info

    def set_stoch_env(self, env):
        self._stoch_env = env

    @property
    def is_human(self):
        if self._stoch_env is None:
            return False
        else:
            return self._stoch_env.is_human

    @property
    def unwrapped(self):
        return self

    def auto_record(self, *args, **kwargs):
        self.env.unwrapped.auto_record(*args, **kwargs)

    def stop_record(self):
        self.env.unwrapped.stop_record()

    def set_movie_id(self, mid):
        self.env.unwrapped.movie_id = mid

class SubprocVecEnvWrapper(VecEnvWrapper):
    def __init__(self, env_fns, spaces=None):
        import baselines.common.vec_env.subproc_vec_env as VecEnv
        def worker(remote, parent_remote, env_fn_wrapper):
            parent_remote.close()
            env = env_fn_wrapper.x()
            while True:
                cmd, data = remote.recv()
                if cmd == 'step':
                    ob, reward, done, info = env.step(data)
                    if done:
                        ob = env.reset()
                    remote.send((ob, reward, done, info))
                elif cmd == 'reset':
                    ob = env.reset()
                    remote.send(ob)
                elif cmd == 'render':
                    remote.send(env.render(mode='rgb_array'))
                elif cmd == 'close':
                    remote.close()
                    break
                elif cmd == 'get_spaces':
                    remote.send((env.observation_space, env.action_space))
                elif cmd == 'is_human':
                    remote.send(env.unwrapped.is_human)
                else:
                    raise NotImplementedError
        VecEnv.worker = worker
        venv = VecEnv.SubprocVecEnv(env_fns, spaces)
        VecEnvWrapper.__init__(self, venv)

    def reset(self):
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return obs, rews, dones, infos

    @property
    def is_human(self):
        for remote in self.venv.remotes:
            remote.send(('is_human', None))
        return np.stack([remote.recv() for remote in self.venv.remotes])

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
                    buttons = SonicConfig.BUTTONS.value,
                    same_as = SonicConfig.BUTTONS_SAME_AS.value[0],
                    inactive = SonicConfig.BUTTONS_INACTIVE.value)
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class EnvManager(object):
    def __init__(self, supervised_levels=None, num_stack = 4, play_path='human', scenario='contest', order_actions=True):

        all_actions = []

        if supervised_levels == None:
            supervised_levels = []

        self.num_stack
        self.play_path = play_path
        self.scenario = scenario

        actions_map = self.get_human_actions(supervised_levels)

        all_actions = []

        for game, state in actions_map:
            for ep in actions_map[(game,state)]:
                all_actions+=actions_map[(game,state)][ep]

        action_set = [np.array(a, 'int') for a in set(map(tuple, all_actions))]

        if len(action_set)>0:
            if order_actions:
                # order key actions, helps with debugging
                buttons = SonicConfig.BUTTONS.value
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

        self.action_set = action_set
        self._actions_map = actions_map

    @property
    def obs_shape(self):
        obs_shape = SonicConfig.OBS_SHAPE.value
        return (obs_shape[0] * self.num_stack, *obs_shape[1:])
         
    def get_human_actions(self, game_state):

        actions_map = {}

        for game, state in set(game_state):

            path = os.path.join(self.play_path,game,self.scenario)

            fi = os.path.join(path,'{}-{}.json'.format(game,state))

            with open(fi) as f:
                actions_map[(game,state)] = json.load(f)

            for ep in actions_map[(game,state)]:

                arr = SonicActions.from_sonic_config(actions_map[(game,state)][ep])
                actions_map[(game,state)][ep] = arr.data.tolist()

        return actions_map

    def make_vec_env(self, game_state, num_processes, log_dir=None, record_dir=None, record_interval=None):

        chunks = [num_processes//len(game_state)]*len(game_state)
        chunks[-1]+=num_processes-sum(chunks)

        processes = []

        for i, size in enumerate(chunks):
            for j in range(size):
                processes.append(game_state[i])

        envs =  [_make_env(game = processes[i][0], state = processes[i][1], seed = 1, rank = i, 
                    log_dir = log_dir, scenario = self.scenario, action_set = self.action_set, 
                        actions = None, record_dir = record_dir, record_interval = record_interval)
                            for i in range(num_processes)]

        return _make_vec_env(envs)

    def make_human_vec_env(self, game_state=None, log_dir=None, record_dir=None, record_interval=None, human_prob=0.1, max_episodes=None):

        sonic_actions = []
        processes = []

        if game_state is None:
            game_state = self._actions_map.keys()

        if max_episodes is None:
            max_episodes = float('inf')

        for gs in set(game_state):
            if gs not in self._actions_map:
                raise KeyError('{} not found'.format(gs))
            else:
                for i, ep in enumerate(self._actions_map[gs]):
                    if i < max_episodes:
                        sonic_actions.append(self._actions_map[gs][ep])
                        processes.append(gs)

        num_processes = len(processes)

        envs =  [_make_env(game = processes[i][0], state = processes[i][1], seed = 1, rank = i, 
                    log_dir = log_dir, scenario = self.scenario, action_set = self.action_set, 
                        actions = sonic_actions[i], human_prob = human_prob, record_dir = record_dir, record_interval = record_interval)
                            for i in range(num_processes)]

        return _make_vec_env(envs)

def _make_env(game, state, seed, rank, log_dir=None, scenario='contest', action_set=None, 
        actions=None, human_prob=0.1, record_dir=None, record_interval=10):

    def _thunk():

        if actions is None:
            env = contest_make(game,state)
            env = RetroWrapper(env, pid = rank)
        else:
            env = make(game, state, scenario=scenario)
            env = RetroWrapper(env, pid = rank)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)
            henv = HumanPlay(env, actions)
            senv = StochasticFrameSkip(env, n=4, stickprob=0.25)
            env = StochasticHumanPlay(senv, henv, humanprob=human_prob)

        env.seed(seed + rank)

        if record_dir is not None:
            record_path = os.path.join(record_dir,str(rank))
            os.makedirs(record_path, exist_ok=True)
            env = EnvRecorder(env, record_path, record_interval)

        if log_dir is not None:
            log_path = os.path.join(log_dir, state)
            os.makedirs(log_path, exist_ok=True)
            env = bench.Monitor(env, os.path.join(log_path, str(rank)))

        env = SonicObsWrapper(env)
        env = AllowBacktracking(env)
        env = SonicActDiscretizer(env, action_set)

        return env

    return _thunk

def _make_vec_env(envs):

    num_processes = len(envs)

    if num_processes > 1:
        envs = SubprocVecEnvWrapper(envs)
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

    maker = EnvManager(supervised_levels=[('SonicTheHedgehog-Genesis','GreenHillZone.Act1'),
        ('SonicTheHedgehog-Genesis','GreenHillZone.Act3')], play_path='../../glf/play/human', scenario='contest')

    envs = maker.make_human_vec_env(log_dir=None,record_dir='../../test_bk2s',record_interval=2, max_episodes=2)

    envs.reset()
    while True:
        _obs, _rew, done, _info = envs.step(np.random.randint(0, len(maker.action_set), envs.num_envs))
        envs.render()
