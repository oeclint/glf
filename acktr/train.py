
import torch
import numpy as np

from glf.common.sonic_util import make_envs, SonicActionsVec, update_current_obs, actions_from_human_data
from glf.common.containers import OrderedMapping
from glf.acktr.a2c_acktr import A2C_ACKTR
from glf.acktr.model import Policy
from glf.acktr.storage import RolloutStorage

import time
from baselines.logger import CSVOutputFormat


class Trainer(object):
    def __init__(self, num_stack = 4,
        recurrent_policy = True,
        vis = False,
        value_loss_coef = 0.5,
        entropy_coef = 0.01,
        alpha = 0.99,
        eps = 1e-5,
        max_grad_norm = 0.5,
        num_steps = 4,
        num_processes = 16,
        cuda = None,
        num_frames = 10e6,
        log_dir = 'log',
        seed = 1,
        use_gae = False,
        gamma = 0.99,
        tau = 0.95,
        vis_interval = 100,
        save_interval = 100,
        save_dir = 'trained_models',
        algo = 'acktr',
        port = 8097):
        
        torch.manual_seed(seed)

        if cuda is None:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = cuda

        self.recurrent_policy = recurrent_policy
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef 
        self.alpha = alpha
        self.eps = eps
        self.max_grad_norm = max_grad_norm

        self.num_steps = num_steps
        self.num_stack = num_stack
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        
        if self.cuda:
            torch.cuda.manual_seed(seed)
            
        torch.set_num_threads(1)

        if vis:
            from visdom import Visdom
            viz = Visdom(port=port)
            win = None

        self.agent = None
        self.actions = None

    def make_agent(self, lr, obs_shape, action_space):

        actor_critic = Policy(obs_shape, action_space, self.recurrent_policy, self.cuda)

        self.agent = A2C_ACKTR(
            actor_critic,
            self.value_loss_coef,
            self.entropy_coef,
            lr = lr,
            alpha = self.alpha,
            eps = self.eps,
            max_grad_norm = self.max_grad_norm)

    def set_lr(self, lr)

        if self.agent is not None:

            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = lr

        else:

            raise RuntimeError('agent is not set')
   
    def train(self,game,state,lr=7e-5,num_frames=10e6,num_processes=16,log_dir='log',log_interval=10,record_dir='bk2s',record_interval=100):

        processes = [(game,state)]*num_processes

        envs = make_envs(processes,log_dir=log_dir,actions=self.actions,record_dir=record_dir, record_interval=record_interval)

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * self.num_stack, *obs_shape[1:])

        if self.agent is None:
            self.make_agent(lr,obs_shape, envs.action_space)
        else:
            self.set_lr(lr)

        actor_critic = self.agent.actor_critic

        rollouts = RolloutStorage(self.num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(num_processes, *obs_shape)

        obs = envs.reset()
        update_current_obs(current_obs, obs, envs, self.num_stack)
        rollouts.observations[0].copy_(current_obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])

        if self.cuda:
            current_obs = current_obs.cuda()
            rollouts.cuda()

        num_updates = int(num_frames) // self.num_steps // num_processes

        csvwriter = CSVOutputFormat('rewards.csv')

        start = time.time()
        for j in range(num_updates):
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                            rollouts.observations[step],
                            rollouts.states[step],
                            rollouts.masks[step])
                cpu_actions = action.squeeze(1).cpu().numpy()
                #print(cpu_actions,'machine')
                # Obser reward and next obs
                obs, reward, done, info = envs.step(cpu_actions)
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                episode_rewards += reward

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                if self.cuda:
                    masks = masks.cuda()

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(current_obs, obs, envs, self.num_stack)
                rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)


            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

            value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

            rollouts.after_update()


            if j % log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * num_processes * self.num_steps

                kv = OrderedMapping([("updates", j),
                                  ("num timesteps", total_num_steps),
                                  ("FPS", int(total_num_steps / (end - start))),
                                  ("mean reward", final_rewards.mean().tolist()),
                                  ("median reward", final_rewards.median().tolist()),
                                  ("min reward", final_rewards.min().tolist()),
                                  ("max reward", final_rewards.max().tolist()),
                                  ("entropy", dist_entropy),
                                  ("value_loss", value_loss),
                                  ("action_loss", action_loss)])

                csvwriter.writekvs(kv)

    def train_from_human(self,game_state,lr=7e-6,num_repeat=30,log_dir='log_human',play_path='human',scenario='contest'):

        unique_actions, game_state_actions = actions_from_human_data(game_state, scenario, play_path)
        self.actions = unique_actions

        sonic_actions = []
        processes = []

        for k in game_state_actions:
            for ep in game_state_actions[k]:
                sonic_actions.append(game_state_actions[k][ep])
                processes.append(k)
 
        num_processes = len(processes)
        envs = make_envs(processes, log_dir=log_dir, scenario=scenario, actions=self.actions)

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * self.num_stack, *obs_shape[1:])

        if self.agent is None:
            self.make_agent(lr,obs_shape, envs.action_space)
        else:
            self.set_lr(lr)

        actor_critic = self.agent.actor_critic

        sonic_actions = SonicActionsVec(sonic_actions)
        sonic_actions = sonic_actions.discretize(self.actions)

        #f = open('acc.txt', 'w')

        for s in range(num_repeat):

            rollouts = RolloutStorage(self.num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
            current_obs = torch.zeros(num_processes, *obs_shape)

            obs = envs.reset()
            update_current_obs(current_obs, obs, envs, self.num_stack)
            rollouts.observations[0].copy_(current_obs)

            supervised_losses = torch.zeros(self.num_steps, num_processes, 1)
            # These variables are used to compute average rewards for all processes.
            episode_rewards = torch.zeros([num_processes, 1])
            final_rewards = torch.zeros([num_processes, 1])

            if self.cuda:
                current_obs = current_obs.cuda()
                supervised_losses = supervised_losses.cuda()
                rollouts.cuda()

            start = time.time()

            for action_group in sonic_actions.stack_by(self.num_steps):
                for step, actions in enumerate(action_group):

                    with torch.no_grad():
                        value, critic_actions , action_log_prob, states = actor_critic.act(
                                rollouts.observations[step],
                                rollouts.states[step],
                                rollouts.masks[step])
                   
                    actions = torch.tensor(actions)
                    if self.cuda:
                        actions.cuda()

                    cpu_actions = actions.squeeze(1).cpu().numpy()
                    #weighted loss calculation
                    for i,(act, true_act) in enumerate(zip(critic_actions,cpu_actions)):
                        if int(act) == int(true_act):
                            l = 0
                        else:
                            if true_act == 1:
                                l = 1
                            else:
                                l = 20
                        supervised_losses[step][i] = l                        
                     # Obser reward and next obs
                    obs, reward, done, info = envs.step(cpu_actions)
                    reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                    episode_rewards += reward

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    final_rewards *= masks
                    final_rewards += (1 - masks) * episode_rewards
                    episode_rewards *= masks

                    if self.cuda:
                        masks = masks.cuda()

                    if current_obs.dim() == 4:
                        current_obs *= masks.unsqueeze(2).unsqueeze(2)
                    else:
                        current_obs *= masks

                    update_current_obs(current_obs, obs, envs, self.num_stack)
                    rollouts.insert(current_obs, states, actions, action_log_prob, value, reward, masks)

                    critic_actions = critic_actions.squeeze(1).cpu().numpy()
                    expected_actions = cpu_actions

                    #f.write('step ' + str(s) + '>>' + ','.join([str(a1) for a1 in critic_actions]) + '>>' + ','.join([str(a2) for a2 in expected_actions]) + '\n')
              
                with torch.no_grad():
                    next_value = actor_critic.get_value(rollouts.observations[-1],
                                                        rollouts.states[-1],
                                                        rollouts.masks[-1]).detach()

                rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)
                supervised_loss = supervised_losses.mean()
                
                #f.write('loss >>' + str(supervised_loss) + '\n')

                value_loss, action_loss, dist_entropy = self.agent.update(rollouts,supervised_loss)

                rollouts.after_update()

        #f.close()

if __name__ == '__main__':
    t = Trainer()
    #t.train_from_human([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')],play_path='../play/human')
    t.train('SonicTheHedgehog-Genesis','GreenHillZone.Act3')
