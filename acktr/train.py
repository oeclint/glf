
import torch
import numpy as np

from glf.common.sonic_util import EnvManager
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
        gmat = None,
        supervised_levels=None,
        play_path='human',
        scenario='contest'):

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
            torch.cuda.manual_seed_all(seed)
            actor_critic.cuda()
        else:
            torch.manual_seed(seed)

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

        runner = Runner(self, envs, num_frames, num_processes, log_interval, log_name)
        runner.run()

    def train_from_human(self,game_state=None,num_frames=10e6,log_dir='log_human',log_interval=10,
        log_name='supervised_rewards.csv',record_dir='supervised_bk2s',record_interval=10,
        human_prob=0.1,max_episodes=None):

        envs = self.em.make_human_vec_env(game_state=game_state, log_dir=log_dir, record_dir=record_dir, 
            record_interval=record_interval, human_prob=human_prob, max_episodes=max_episodes)

        num_processes = envs.num_envs

        runner = Runner(self, envs, num_frames, num_processes, log_interval, log_name)
        runner.run()

class Runner(object):

    def __init__(self, trainer, envs, num_frames, num_processes, log_interval, log_name):
        self.trainer = trainer
        self.envs = envs
        self.num_frames = num_frames
        self.num_processes = num_processes
        self.log_interval = log_interval
        self.log_name = log_name

    def run(self):

        num_frames = self.num_frames
        num_processes = self.num_processes
        
        agent = self.trainer.agent
        actor_critic = self.trainer.agent.actor_critic
        cuda = self.trainer.cuda
        num_steps = self.trainer.num_steps
        use_gae = self.trainer.use_gae
        gamma = self.trainer.gamma
        tau = self.trainer.tau

        envs = self.envs
        obs_shape = envs.observation_space.shape

        if self.trainer.gmat is not None:
            actor_critic.base.set_batches(processes)

        csvwriter = CSVOutputFormat(self.log_name)

        rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)

        if cuda:
            rollouts.cuda()
            envs.cuda()
        
        obs = envs.reset()

        rollouts.observations[0].copy_(obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])

        is_human = np.array([[False]*num_processes]*num_steps)

        num_updates = int(num_frames) // num_steps // num_processes

        start = time.time()
        for j in range(num_updates):
            for step in range(num_steps):

                with torch.no_grad():
                    value, actor_actions, action_log_prob, states = actor_critic.act(
                            rollouts.observations[step],
                            rollouts.states[step],
                            rollouts.masks[step])

                is_human[step] = envs.is_human
                
                cpu_actions = actor_actions.squeeze(1).cpu().numpy()

                obs, reward, done, info = envs.step(cpu_actions)
                reward = torch.from_numpy(reward).unsqueeze(1).float()
                episode_rewards += reward

                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                rollouts.insert(obs, states, actor_actions, action_log_prob, value, reward, masks)
          
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, use_gae, gamma, tau)

            human_proc = np.any(is_human,axis=0)
            if np.any(human_proc):
                value_loss, action_loss, dist_entropy = agent.update(rollouts, human_proc)
            else:
                value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()
 
            if j % self.log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * num_processes * num_steps

                kv = OrderedMapping([("updates", j),
                                  ("num_timesteps", total_num_steps),
                                  ("FPS", int(total_num_steps / (end - start))),
                                  ("mean_reward", final_rewards.mean().item()),
                                  ("median_reward", final_rewards.median().item()),
                                  ("min_reward", final_rewards.min().item()),
                                  ("max_reward", final_rewards.max().item()),
                                  ("value_loss", value_loss * agent.value_loss_coef),
                                  ("action_loss", action_loss),
                                  ("dist_entropy", dist_entropy * agent.entropy_coef)])

                csvwriter.writekvs(kv)

if __name__ == '__main__':
    t = Trainer()
    #t.train_from_human([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')],play_path='../play/human')
    t.train([('SonicTheHedgehog-Genesis','GreenHillZone.Act3')])
