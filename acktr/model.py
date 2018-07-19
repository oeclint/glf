import torch
import torch.nn as nn
import torch.nn.functional as F
from glf.acktr.distributions import Categorical, DiagGaussian
from glf.acktr.utils import init, init_normc_
from glf.acktr.gmodel import G


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, n_channel, n_action, recurrent_policy, games=None, temp = 1):
        super(Policy, self).__init__()
        if games is None:
            self.base = CNNBase(n_channel, recurrent_policy)
            self.base_target = CNNBase(n_channel, recurrent_policy)
        else:
            self.base = G(CNNBase(n_channel, recurrent_policy),games)
            self.base_target = G(CNNBase(n_channel, recurrent_policy),games)

        self.dist = Categorical(self.base.output_size, n_action, temp = temp)
        self.dist_target = Categorical(self.base.output_size, n_action, temp = temp)

        self.state_size = self.base.state_size

        self.step = 0
        self.base_target.load_state_dict(self.base.state_dict())
        self.dist_target.load_state_dict(self.dist.state_dict())

        self.base_target.eval()
        self.dist_target.eval()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, actor_features, states = self.base_target(inputs, states, masks)
        dist = self.dist_target(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        self.step+=1

        return value, action, action_log_probs, states

    def get_value(self, inputs, states, masks):
        value, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action, update_step = 1):
        value, actor_features, states = self.base(inputs, states, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if self.step%update_step == 0:
            self.base_target.load_state_dict(self.base.state_dict())
            self.dist_target.load_state_dict(self.dist.state_dict())

        return value, action_log_probs, dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(27648, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states
