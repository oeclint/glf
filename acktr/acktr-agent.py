import copy
import time
import os

from retro_contest.local import make
from glf.common.sonic_util import SonicActDiscretizer, SonicObsWrapper, AllowBacktracking

import torch
import numpy as np

from glf.acktr.a2c_acktr import A2C_ACKTR
from glf.acktr.model import Policy
from glf.acktr.storage import RolloutStorage

from baselines import bench
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv



if __name__ == '__main__':

    num_stack = 4
    recurrent_policy = True
    vis = False
    value_loss_coef = 0.5
    entropy_coef = 0.01
    num_steps = 20
    num_processes = 16
    cuda = torch.cuda.is_available()
    num_frames = 10e6
    log_dir = 'log'
    seed = 1
    use_gae = False
    gamma = 0.99
    tau = 0.95
    vis_interval = 100
    log_interval = 10
    save_interval = 100
    save_dir = 'trained_models'
    algo = 'acktr'
    port = 8097
    
    game='SonicTheHedgehog-Genesis'
    state='LabyrinthZone.Act2'
    
    num_updates = int(num_frames) // num_steps // num_processes

    torch.manual_seed(seed)
    
    if cuda:
        torch.cuda.manual_seed(seed)
        
    torch.set_num_threads(1)

    if vis:
        from visdom import Visdom
        viz = Visdom(port=port)
        win = None

    def make_env(game, state, seed, rank, log_dir):

        def _thunk():

            env = make(game,state)

            env.seed(seed + rank)

            if log_dir is not None:
                env = bench.Monitor(env, os.path.join(log_dir, str(rank)))

            env = SonicActDiscretizer(env)
            env = SonicObsWrapper(env)
            env = AllowBacktracking(env)

            return env
        
        return _thunk

    envs = [make_env(game, state, seed, i, log_dir)
                for i in range(num_processes)]

    if num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])

    actor_critic = Policy(obs_shape, envs.action_space, recurrent_policy, cuda=cuda)

    agent = A2C_ACKTR(
        actor_critic,
        value_loss_coef,
        entropy_coef,
        lr = 7e-4,
        alpha = 0.99,
        eps = 1e-5,
        max_grad_norm = 0.5 )

    rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
    current_obs = torch.zeros(num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([num_processes, 1])
    final_rewards = torch.zeros([num_processes, 1])

    if cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
            cpu_actions = action.squeeze(1).cpu().numpy()

            # Obser reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)
            rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(rollouts.observations[-1],
                                                rollouts.states[-1],
                                                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma, tau)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % save_interval == 0 and save_dir != "":
            save_path = os.path.join(save_dir, algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                            hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, game + "-" + state + ".pt"))

        if j % log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * num_processes * num_steps
            print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy,
                       value_loss, action_loss))
        if vis and j % vis_interval == 0:
            try:
                # Sometimes monitor doesn't properly flush the outputs
                win = visdom_plot(viz, win, log_dir, game,
                                  algo, num_frames)
            except IOError:
                pass
