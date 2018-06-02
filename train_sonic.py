from retro_contest.local import make
from collections import deque, Mapping
from itertools import count
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
from sonic_util import SonicEnvWrapper
import csv
import os

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

class ReplayMemory(object):
    
    def __init__(self, capacity, memory=None):
        self.capacity = capacity
        if memory is None:
            self.memory = deque(maxlen=capacity)
        else:
            self.memory = deque(memory, maxlen=capacity)
    
    def push(self, **kwargs):
        self.memory.append(kwargs)
    
    def sample(self, batch_size):
        memory = random.sample(self.memory, batch_size)
        return ReplayMemory(self.capacity,memory)
    
    def __len__(self):
        return len(self.memory)

    def as_tensor_cat(self,key):
        return torch.cat([torch.tensor(m[key]) for m in self.memory])

class ConvNet(nn.Module):
    """
    The model is a convolutional neural network. The input is the state/observation and
    the output is the quality of each action in the given state.
    """
    
    def __init__(self, n_in, n_out):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(n_in, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(29600, 512)
        self.fc2 = nn.Linear(512, n_out)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(object):
    
    """
    This algorithms uses Deep Q Learning (epsilon greedy).
        
    Neural networks are used to map x->y, where x is the state/observation and y is the reward for all the actions in the given state
    """
    
    def __init__(self, agents, capacity=20000000,
                 batch_size = 128,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 200,
                 target_update = 10,
                 n_cat_states = 5,
                 log = None):
        
        self.agents = agents

        if not all([len(a.actions)==len(agents[0].actions) for a in agents]):
            raise ValueError("All agents must have the same number of actions")
        
        policy_model = ConvNet(n_cat_states * 3, len(agents[0].actions))
        target_model = ConvNet(n_cat_states * 3, len(agents[0].actions))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:

            policy_model = nn.DataParallel(policy_model)
            target_model = nn.DataParallel(target_model)

        self.policy_net = policy_model.to(self.device)
        
        target_net = target_model.to(self.device)
        target_net.load_state_dict(self.policy_net.state_dict())
        target_net.eval()
        self.target_net = target_net
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(capacity)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_step = 0
        
        self.target_update = target_update

        self.n_cat_states = n_cat_states

        self.log = log
        if self.log is not None:
            self.log.info("running on: {device:<5}".format(device=str(self.device)))
            self.log.info("cuda device count: {count:<5}".format(count=str(torch.cuda.device_count())))

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)

        # Concatenate the batch elements
        next_state_batch = batch.as_tensor_cat('next_state')
        state_batch = batch.as_tensor_cat('state')
        action_batch = batch.as_tensor_cat('action')
        reward_batch = batch.as_tensor_cat('reward')

        next_state_batch = next_state_batch.to(self.device).type('torch.FloatTensor')
        state_batch = state_batch.to(self.device).type('torch.FloatTensor')
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of Actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Where V(s_{t+1}) = max_a(Q(s_{t+1},a))
        next_state_values = self.target_net(next_state_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
           param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        for agent in self.agents:
 #           env = agent.env
 #           self.eps_step = 0
            if agent.record:
                env = SonicEnvWrapper(make(game=agent.game, state=agent.state, bk2dir=agent.path))
            else:
                env = SonicEnvWrapper(make(game=agent.game, state=agent.state))
                
            for i_episode in range(agent.n_episodes):
                if self.log is not None:
                    self.log.info("-->game: {game:<30}".format(game=agent.game))
                    self.log.info("-->state: {state:<30}".format(state=agent.state))
                    self.log.info("-->episode: {episode:<4}".format(episode=i_episode))
                    
                # Initialize the environment and state
                state = env.reset()
                # Stacked states
                states = deque([state]*self.n_cat_states,maxlen=self.n_cat_states)
                state_cat=np.concatenate(states, axis=1)
                
                next_states = deque(maxlen=self.n_cat_states)
                rewards = deque(maxlen=20)
                
                for t in count():
                    # Select and perform an action
                    action = self.select_action(state_cat, agent)
                    action_id = action.item()
                    next_state, reward, done, info = env.step(agent.actions[action_id])

                    if (self.log is not None) and (t%10 == 0):
                        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.log.info(
                            "---->step: {step:>7}; action: {action:>1}; xpos: {xpos:>5}; reward: {reward:>5}; time: {time}".format(
                                step=t, action=str(action_id), xpos=str(info['x']), reward="{0:.2f}".format(reward), time=time))

                    rewards.append(reward)
                    
                    if len(rewards)==20:
                        if all([rew<0.01 for rew in rewards]):
                            # if rewards are bad for 20 steps then take more random guesses
                            if self.eps_start < 0.20:
                                self.eps_start = 0.20
                            self.eps_step = 0
                            
                    reward = torch.tensor([reward], device=self.device)

                    # States to be stacked, continue to next loop if didn't reach target size

                    next_states.append(next_state)
                    
                    if len(next_states) == self.n_cat_states:

                        next_state_cat=np.concatenate(next_states, axis=1)
                        
                        # Store the transition in memory
                        self.memory.push(state=state_cat, action=action,
                                            next_state=next_state_cat, reward=reward)

                        # Move to the next state
                        state_cat = next_state_cat

                        # Render if not being recorded
                        if not agent.record:
                            env.render()
                        
                        # Perform one step of the optimization
                        self.optimize()
                        if done:
                            self.save_policy("{0}_{1}.p".format(agent.game,agent.state))
                            break

                    else:
                        states.append(next_state)
                        state_cat=np.concatenate(states, axis=1)
                        
                # Update the target network
                if i_episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    
            del env

    def select_action(self, state, agent):
        state = torch.from_numpy(state).to(self.device).type('torch.FloatTensor')
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.eps_step / self.eps_decay)
        self.eps_step = self.eps_step + 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(
                        len(agent.actions))]], device=self.device, dtype=torch.long)

    def save_policy(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_policy(self, path):
        self.policy_net.load_state_dict(torch.load(path))

class Agent(object):

    def __init__(self, actions, n_episodes=2, game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', record=None):

        self.actions = actions
        self.n_episodes = n_episodes
        self.game = game
        self.state = state
        if record is not None:
            self.record = True
        self.path = record
        #if self.record:
        #    self.env = SonicEnvWrapper(make(game=game, state=state, bk2dir=record))
        #else:
        #    self.env = SonicEnvWrapper(make(game=game, state=state))
    
#    def run(self):
#        for _ in range(self.n_episodes):
#            done = False
#            obs = self.env.reset()
#            while not done:
#                next_obs, reward, done, info = self.env.step(self.actions.sample())
#                if not self.record:
#                    self.env.render()

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

    dictReader = csv.DictReader(open('sonic-train.csv', 'r'), fieldnames = ['game', 'state'], delimiter = ',', quotechar = '"')
    for i,row in enumerate(dictReader):
        if i != 0:
            #skip header
            directory = "recordings_{0}_{1}".format(row['game'],row['state'])
            if not os.path.exists(directory):
                os.makedirs(directory)
            ag = Agent(actions,game=row['game'],state=row['state'],record=directory)
            agents.append(ag)

    logging.basicConfig(filename='log.txt',level=logging.DEBUG)
    m = Model(agents,log = logging)  
    m.run()
    m.save_policy('policy_model.p')

