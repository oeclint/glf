
import torch
import numpy as np

from glf.common.sonic_util import update_current_obs, EnvMaker
from glf.common.containers import OrderedMapping
from glf.acktr.a2c_acktr import A2C_ACKTR
from glf.acktr.model import Policy
from glf.acktr.storage import RolloutStorage

import time
from baselines.logger import CSVOutputFormat


class Trainer(object):
    def __init__(self, 
        num_stack = 4,
        recurrent_policy = False,
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
        port = 8097,
        gmat = None):
        
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
        self.gmat = gmat
        
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

        if self.gmat is not None:

            actor_critic = Policy(obs_shape, action_space, self.recurrent_policy, self.cuda, self.gmat)

        else:
            actor_critic = Policy(obs_shape, action_space, self.recurrent_policy, self.cuda)

        self.agent = A2C_ACKTR(
            actor_critic = actor_critic,
            value_loss_coef = self.value_loss_coef,
            entropy_coef = self.entropy_coef,
            lr = lr,
            alpha = self.alpha,
            eps = self.eps,
            max_grad_norm = self.max_grad_norm)

    def set_lr(self, lr):

        if self.agent is not None:

            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] = lr

        else:

            raise RuntimeError('agent is not set')
   
    def train(self,game_state,lr=1e-4,num_frames=10e6,num_processes=16,log_dir='log',log_interval=10,record_dir='bk2s',record_interval=100):

        maker = EnvMaker(game_state=game_state, num_processes=num_processes, log_dir=log_dir,actions=self.actions,
            record_dir=record_dir, record_interval=record_interval)

        envs = maker.vec_env

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * self.num_stack, *obs_shape[1:])

        if self.agent is None:
            self.make_agent(lr,obs_shape, envs.action_space)
        else:
            self.set_lr(lr)

        actor_critic = self.agent.actor_critic

        if self.gmat is not None:
            actor_critic.base.set_batches(processes)

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
                                  ("mean reward", final_rewards.mean().item()),
                                  ("median reward", final_rewards.median().item()),
                                  ("min reward", final_rewards.min().item()),
                                  ("max reward", final_rewards.max().item()),
                                  ("value loss", value_loss * self.agent.value_loss_coef),
                                  ("action loss", action_loss),
                                  ("dist entropy", dist_entropy * self.agent.entropy_coef)])

                csvwriter.writekvs(kv)

    def train_from_human(self,game_state,lr=1e-3,num_frames=10e6,log_dir='log_human',log_interval=10,
        record_dir='supervised_bk2s',record_interval=10,play_path='human',scenario='contest',p_correct=0.95):

        maker = EnvMaker.from_human_play(game_state=game_state, play_path=play_path, scenario=scenario, log_dir=log_dir,
            record_dir=record_dir,record_interval=record_interval)
        envs = maker.vec_env
        self.actions = maker.action_set

        num_processes = envs.num_envs

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * self.num_stack, *obs_shape[1:])

        if self.agent is None:
            self.make_agent(lr,obs_shape, envs.action_space)
        else:
            self.set_lr(lr)

        actor_critic = self.agent.actor_critic

        if self.gmat is not None:
            actor_critic.base.set_batches(processes)

        csvwriter = CSVOutputFormat('supervised_loss.csv')

        rollouts = RolloutStorage(self.num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(num_processes, *obs_shape)

        obs = envs.reset()
        update_current_obs(current_obs, obs, envs, self.num_stack)
        rollouts.observations[0].copy_(current_obs)

        #supervised_prob = torch.zeros(self.num_steps, num_processes, 1)
        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])

        if self.cuda:
            current_obs = current_obs.cuda()
           # supervised_prob = supervised_prob.cuda()
            rollouts.cuda()

        num_updates = int(num_frames) // self.num_steps // num_processes

        start = time.time()
        for j in range(num_updates):
            rr = np.random.random()
            supervised_prob = torch.zeros(self.num_steps, num_processes, 1)
            if self.cuda:
                supervised_prob = supervised_prob.cuda()
            for step in range(self.num_steps):

                with torch.no_grad():
                    value, critic_actions, action_log_prob, states = actor_critic.act(
                            rollouts.observations[step],
                            rollouts.states[step],
                            rollouts.masks[step])

                if rr<=np.exp(-1*(j * num_processes * self.num_steps)/500000):
                 #   supervised log prob calculation
                    if None not in envs.actions: 
                        n_actions = len(self.actions)
                        for i,(act, true_act) in enumerate(zip(critic_actions,envs.actions)):
                           if int(act) == int(true_act):
                               # probability it is correct
                               prob = p_correct
                           else:
                               # assume all wrong choices are uniformily distributed
                               prob = (1 - p_correct)/(n_actions-1)

                           supervised_prob[step][i] = prob
                           #_, _, _ = self.agent.update(rollouts, supervised_prob)
                    # fused = [int(a2) if a1 is None else int(a1) for a1, a2 in zip(
                    #     envs.actions, critic_actions)]

                    # critic_actions = torch.tensor(fused).unsqueeze(1)

                    # if self.cuda:
                    #     critic_actions = critic_actions.cuda()
                
                cpu_actions = critic_actions.squeeze(1).cpu().numpy()
                # Obser reward and next obs
                obs, reward, done, info = envs.step(cpu_actions)
                #actions = [i['action'] for i in info]
                #supervised log prob calculation
                #n_actions = len(self.actions)
                #for i,(act, true_act) in enumerate(zip(critic_actions,actions)):
                #    if int(act) == int(true_act):
                #        # probability it is correct
                #        prob = p_correct
                #    else:
                #        # assume all wrong choices are uniformily distributed
                #        prob = (1 - p_correct)/(n_actions-1)

                #    supervised_prob[step][i] = prob
  
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

                rollouts.insert(current_obs, states, critic_actions, action_log_prob, value, reward, masks)
          
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, True, self.gamma, self.tau)
            if np.any(supervised_prob.cpu().numpy()):
                value_loss, action_loss, dist_entropy = self.agent.update(rollouts, supervised_prob)
            else:
                value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

            rollouts.after_update()
 
            if j % log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * num_processes * self.num_steps

                kv = OrderedMapping([("updates", j),
                                  ("num timesteps", total_num_steps),
                                  ("FPS", int(total_num_steps / (end - start))),
                                  ("mean reward", final_rewards.mean().item()),
                                  ("median reward", final_rewards.median().item()),
                                  ("min reward", final_rewards.min().item()),
                                  ("max reward", final_rewards.max().item()),
                                  ("value loss", value_loss * self.agent.value_loss_coef),
                                  ("action loss", action_loss),
                                  ("dist entropy", dist_entropy * self.agent.entropy_coef)])

                csvwriter.writekvs(kv)

if __name__ == '__main__':
    t = Trainer()
    #t.train_from_human([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')],play_path='../play/human')
    t.train([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')])
