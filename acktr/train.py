
import torch
import numpy as np

from glf.common.sonic_util import update_current_obs, EnvManager
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
        lr = 1e-4,
        alpha = 0.99,
        eps = 1e-5,
        max_grad_norm = 0.5,
        num_steps = 4,
        cuda = None,
        seed = 1,
        use_gae = False,
        gamma = 0.99,
        tau = 0.95,
        vis_interval = 100,
        save_interval = 100,
        save_dir = 'trained_models',
        algo = 'acktr',
        port = 8097,
        gmat = None,
        supervised_levels=None,
        play_path='human',
        scenario='contest'):
        
        torch.manual_seed(seed)

        if cuda is None:
            self.cuda = torch.cuda.is_available()
        else:
            self.cuda = cuda

        self.recurrent_policy = recurrent_policy
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.lr = lr 
        self.alpha = alpha
        self.eps = eps
        self.max_grad_norm = max_grad_norm

        self.num_steps = num_steps
        self.num_stack = num_stack
        self.use_gae = use_gae
        self.gamma = gamma
        self.tau = tau
        self.gmat = gmat

        torch.set_num_threads(1)

        self.em = EnvManager(supervised_levels, num_stack, play_path, scenario)

        actor_critic = Policy(self.em.obs_shape[0], self.em.n_action, self.recurrent_policy, self.gmat)

        if self.cuda:
            torch.cuda.manual_seed(seed)
            actor_critic.cuda()

        self.agent = A2C_ACKTR(
            actor_critic = actor_critic,
            value_loss_coef = self.value_loss_coef,
            entropy_coef = self.entropy_coef,
            lr = self.lr,
            alpha = self.alpha,
            eps = self.eps,
            max_grad_norm = self.max_grad_norm)
   
    def train(self,game_state,num_frames=10e6,num_processes=16,log_dir='log',log_interval=10, log_name='rewards.csv',
        record_dir='bk2s',record_interval=10):

        envs = self.em.make_vec_env(game_state=game_state, num_processes=num_processes, log_dir=log_dir,
            record_dir=record_dir, record_interval=record_interval)

        self._train(envs, num_frames, num_processes, log_interval, log_name)

    def train_from_human(self,game_state=None,num_frames=10e6,log_dir='log_human',log_interval=10,
        log_name='supervised_rewards.csv',record_dir='supervised_bk2s',record_interval=10,
        human_prob=0.1,max_episodes=None):

        envs = self.em.make_human_vec_env(game_state=game_state, log_dir=log_dir, record_dir=record_dir, 
            record_interval=record_interval, human_prob=human_prob, max_episodes=max_episodes)

        num_processes = envs.num_envs

        self._train(envs, num_frames, num_processes, log_interval, log_name)

    def _train(self, envs, num_frames, num_processes, log_interval, log_name):

        actor_critic = self.agent.actor_critic
        obs_shape = self.em.obs_shape

        if self.gmat is not None:
            actor_critic.base.set_batches(processes)

        csvwriter = CSVOutputFormat(log_name)

        rollouts = RolloutStorage(self.num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(num_processes, *obs_shape)

        obs = envs.reset()
        update_current_obs(current_obs, obs, envs, self.num_stack)
        rollouts.observations[0].copy_(current_obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])
        is_human = np.array([[False]*num_processes]*self.num_steps)

        if self.cuda:
            current_obs = current_obs.cuda()
            rollouts.cuda()

        num_updates = int(num_frames) // self.num_steps // num_processes

        start = time.time()
        for j in range(num_updates):
            for step in range(self.num_steps):

                with torch.no_grad():
                    value, actor_actions, action_log_prob, states = actor_critic.act(
                            rollouts.observations[step],
                            rollouts.states[step],
                            rollouts.masks[step])

                is_human[step] = envs.is_human
                
                cpu_actions = actor_actions.squeeze(1).cpu().numpy()

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
                rollouts.insert(current_obs, states, actor_actions, action_log_prob, value, reward, masks)
          
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.tau)

            human_proc = np.any(is_human,axis=0)
            if np.any(human_proc):
                value_loss, action_loss, dist_entropy = self.agent.update(rollouts, human_proc)
            else:
                value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

            rollouts.after_update()
 
            if j % log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * num_processes * self.num_steps

                kv = OrderedMapping([("updates", j),
                                  ("num_timesteps", total_num_steps),
                                  ("FPS", int(total_num_steps / (end - start))),
                                  ("mean_reward", final_rewards.mean().item()),
                                  ("median_reward", final_rewards.median().item()),
                                  ("min_reward", final_rewards.min().item()),
                                  ("max_reward", final_rewards.max().item()),
                                  ("value_loss", value_loss * self.agent.value_loss_coef),
                                  ("action_loss", action_loss),
                                  ("dist_entropy", dist_entropy * self.agent.entropy_coef)])

                csvwriter.writekvs(kv)

if __name__ == '__main__':
    t = Trainer()
    #t.train_from_human([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')],play_path='../play/human')
    t.train([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')])
