from retro_contest.local import make
import retro
from collections import deque, Mapping
from itertools import count
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

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
    
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(29600, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Model(object):
    
    """
    This algorithms uses Deep Q Learning (epsilon greedy).
        
    Neural networks are used to map x->y, where x is the state/observation and y is the reward for all the actions in the given state
    """
    
    def __init__(self, agents, capacity=10000,
                 batch_size = 128,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 200,
                 target_update = 10):
        
        self.agents = agents
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = ConvNet().to(self.device)
        target_net = ConvNet().to(self.device)
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
        self.target_update = target_update

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

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
           param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def run(self):
        for agent in self.agents:
            for i_episode in range(agent.n_episodes):
                # Initialize the environment and state
                env = agent.env
                state = env.reset()
                state = agent.process_obs(state)
                for t in count():
                    # Select and perform an action
                    action = self.select_action(state, t, agent)
                    action_id = action.item()
                    #action_id = 0
                    next_state, reward, done, info = env.step(agent.actions[action_id])
                    next_state = agent.process_obs(next_state)
                    reward = torch.tensor([reward], device=self.device)
                    
                    # Store the transition in memory
                    self.memory.push(state=state, action=action,
                                        next_state=next_state, reward=reward)
                    
                    # Move to the next state
                    state = next_state
                    
                    if not agent.record:
                        env.render()
                    
                    # Perform one step of the optimization
                    self.optimize()
                    if done:
                        #episode_durations.append(t + 1)
                        #plot_durations()
                        break
                # Update the target network
                if i_episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, step, agent):
        state = torch.from_numpy(state).to(self.device).type('torch.FloatTensor')
        sample = np.random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * step / self.eps_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(
                        len(agent.actions))]], device=self.device, dtype=torch.long)

class Agent(object):

    def __init__(self, actions, n_episodes=30, game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1', record=False):

        self.actions = actions
        self.n_episodes = n_episodes
        self.record = record
        if self.record:
            self.env = retro.make(game=game, state=state, record='.')
        else:
            self.env = make(game=game, state=state)

    def process_obs(self, obs):
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
    
    def run(self):
        for _ in range(self.n_episodes):
            done = False
            obs = self.env.reset()
            while not done:
                next_obs, reward, done, info = self.env.step(self.actions.sample())
                if not self.record:
                    self.env.render()

if __name__ == '__main__':

    """
    Each action is an array of 12 items. Each item consist of a sub-action (i.e. left, jump etc..).
    The activation of the sub-action is determined by the value of the item (0 or 1).
    Therefore, there are 2^12 (4096) different possible actions.
    
    However, some actions are redundant. Only 10 unique actions are used by this object.
    """
    actions = Actions({
        0: [0,0,0,0,1,0,0,0,0,0,0,0], # UP
        1: [0,0,0,0,0,1,0,0,0,0,0,0], # DOWN
        2: [0,0,0,0,0,0,1,0,0,0,0,0], # LEFT
        3: [0,0,0,0,0,0,0,1,0,0,0,0], # RIGHT
        4: [0,0,0,0,0,1,1,0,0,0,0,0], # LEFT DOWN
        5: [0,0,0,0,0,1,0,1,0,0,0,0], # RIGHT DOWN
        6: [1,0,0,0,0,0,0,0,0,0,0,0], # JUMP
        7: [1,0,0,0,0,0,1,0,0,0,0,0], # JUMP LEFT
        8: [1,0,0,0,0,0,0,1,0,0,0,0], # JUMP RIGHT
        9: [1,0,0,0,0,1,0,0,0,0,0,0], # SPIN DASH (>= SONIC 2)
        })

    agent = Agent(actions,record=True)
    m = Model([agent])
    m.run()

