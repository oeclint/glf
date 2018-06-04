from collections import deque, OrderedDict
from itertools import count
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime

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
    
    def __init__(self, capacity=20000000,
                 batch_size = 128,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 200,
                 target_update = 10,
                 n_cat_states = 5,
                 n_actions = 7,
                 device = None,
                 policy = None,
                 log = None):
        
        policy_model = ConvNet(n_cat_states * 3, n_actions)
        target_model = ConvNet(n_cat_states * 3, n_actions)

        if policy is not None:
            # If policy exists only optimize last layer
            state_dict = torch.load(policy)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.','') # remove `module.`
                new_state_dict[name] = v
            policy_model.load_state_dict(new_state_dict)
            # Need to set requires_grad = False to freeze the paramenters so that the
            # gradients are not computed in backward()
            for param in policy_model.parameters():
                param.required_grad = False
            num_ftrs = policy_model.fc2.in_features
            # Parameters of newly constructed modules have requires_grad=True by default
            policy_model.fc2 = nn.Linear(num_ftrs, n_actions)
            # Set parameters to only last
            self.policy_net_parameters = policy_model.fc2.parameters()
        else:
            self.policy_net_parameters = policy_model.parameters()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if torch.cuda.device_count() > 1:

            policy_model = nn.DataParallel(policy_model)
            target_model = nn.DataParallel(target_model)

        self.policy_net = policy_model.to(self.device)
        
        target_net = target_model.to(self.device)
        target_net.load_state_dict(self.policy_net.state_dict())
        target_net.eval()
        self.target_net = target_net
        
        self.optimizer = optim.RMSprop(self.policy_net_parameters)
        self.memory = ReplayMemory(capacity)
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps_step = 0
        
        self.target_update = target_update

        self.n_cat_states = n_cat_states

        self._n_actions = n_actions

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
        for param in self.policy_net_parameters:
           param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, agents):
        
        for agent in agents:
            
            if not len(agent.actions)==self._n_actions:
                raise ValueError("Expected agent to have {0} action(s)".format(str(self._n_actions)))

            agent.make()
            env = agent.env
                            
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

                        if agent.render:
                            env.render()
                        
                        # Perform one step of the optimization
                        self.optimize()
                        if done:
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
