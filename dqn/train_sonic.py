from collections import Mapping
import numpy as np
import logging
import csv
import os
from sonic_util import SonicEnvWrapper
from model import Model

class Actions(Mapping):
    
    def __init__(self, actions):
        self._actions = actions
    
    def __getitem__(self,item):
        return self._actions[item]
    
    def __iter__(self):
        return iter(self._actions)
    
    def __len__(self):
        return len(self._actions)

    def sample(self):
        ind = np.random.randint(len(self))
        return self[ind]

class Agent(object):

    def __init__(self, actions, n_episodes=2500, env=None, game=None, state=None, record=None, render=False):

        self.actions = actions
        self.n_episodes = n_episodes
        self.game = game
        self.state = state
        self.render = render
        
        if record is not None:
            self.record = True
        else:
            self.record = False
        self.path = record

        if env is not None:
            self.env = SonicEnvWrapper(env)
        else:
            self.env = env           

    def make(self):
        if self.env is None:
            from retro_contest.local import make
            if self.record:
                self.env = SonicEnvWrapper(make(game=self.game, state=self.state, bk2dir=self.path))
            else:
                self.env = SonicEnvWrapper(make(game=self.game, state=self.state))
    
if __name__ == '__main__':

    """
    Each action is an array of 12 items. Each item consist of a sub-action (i.e. left, jump etc..).
    The activation of the sub-action is determined by the value of the item (0 or 1).
    Therefore, there are 2^12 (4096) different possible actions.
    
    However, some actions are redundant. Only 7 unique actions are used by this object.
    """
    
    actions = Actions({
        0: [0,0,0,0,0,1,0,0,0,0,0,0], # DOWN
        1: [0,0,0,0,0,0,1,0,0,0,0,0], # LEFT
        2: [0,0,0,0,0,0,0,1,0,0,0,0], # RIGHT
        3: [0,0,0,0,0,1,1,0,0,0,0,0], # LEFT DOWN
        4: [0,0,0,0,0,1,0,1,0,0,0,0], # RIGHT DOWN
        5: [1,0,0,0,0,0,0,0,0,0,0,0], # JUMP
        6: [1,0,0,0,0,1,0,0,0,0,0,0], # SPIN DASH (>= SONIC 2)
        })

    agents = []

    dictReader = csv.DictReader(open('sonic-train.csv', 'r'),
                                fieldnames = ['game', 'state'],
                                delimiter = ',', quotechar = '"')
    
    for i,row in enumerate(dictReader):
        if i != 0:
            #skip header
            directory = "recordings"
            if not os.path.exists(directory):
                os.makedirs(directory)
            ag = Agent(actions,game=row['game'],state=row['state'],record=directory)
            agents.append(ag)

    logging.basicConfig(filename='log.txt',level=logging.DEBUG)
    m = Model(log = logging)  
    m.train(agents)
    m.save_policy('policy_model.p')

