import glob
import os
import json
from retro_contest.local import make as contest_make
from retro import make
import retro
import argparse
from glf.common.sonic_util import SonicActions
from glf.common.parse import to_json
import gym

class PlayBack(object):

    def __init__(self, game, state, scenario, root='human',maker=None):
        self.path = os.path.join(root,game,scenario)
        self.game = game
        self.state = state
        self.scenario = scenario
        if maker is None:
            self.make = make
        else:
            self.make = maker

    def play_bk2(self, render=False, write=True):

        keysdict = {}

        for f in glob.glob(os.path.join(self.path,'{}-{}*.bk2'.format(self.game,self.state))):

            movie = retro.Movie(f)
            movie.step()

            env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
            env.initial_state = movie.get_state()
            env.reset()

            base = os.path.basename(f)
            base = os.path.splitext(base)[0]
            ep = int(base.split('-')[-1])

            keysarr = []

            while movie.step():
                keys = []
                for i in range(env.NUM_BUTTONS):
                    keys.append(movie.get_key(i))
                keysarr.append([int(k) for k in keys])
                _obs, _rew, _done, _info = env.step(keys)
                if _done:
                    env.close()
                else:
                    if render:
                        env.render()

            keysdict[ep] = keysarr
            
        if write:
            with open(os.path.join(self.path,'{}-{}.json'.format(self.game,self.state)), 'w') as outfile:
                 outfile.write(to_json(keysdict))

    def play_json(self, render=True):

        with open(os.path.join(self.path,'{}-{}.json'.format(self.game,self.state))) as f:
            data = json.load(f)

        env = self.make(game=self.game, state=self.state, scenario=self.scenario)
        for ep in data:
            obs = env.reset()
            actions = SonicActions.from_sonic_config(data[ep])
            for action in actions.data:
                obs, rew, done, info = env.step(action)
                env.render()
                if done:
                    break

    def filter_json(self, min_rew=None):

        for fi in glob.glob(os.path.join(self.path,'{}-{}.json'.format(self.game,self.state))):

            base = os.path.basename(fi)
            base = os.path.splitext(base)[0]
            splits = base.split('-')
            state = splits[-1]
            game = '-'.join([splits[0],splits[1]])

            filtered_data = {}

            with open(fi) as f:
                data = json.load(f)

            env = self.make(game=game, state=state, scenario=self.scenario)
            env = gym.wrappers.TimeLimit(env, max_episode_seconds=390)

            for ep in data:
                filtered_actions = []
                obs = env.reset()
                reward = 0
                for action in data[ep]:
                    filtered_actions.append(action)
                    obs, rew, done, info = env.step(action)
                    reward += rew
                    if done:
                        if min_rew is not None:
                            if reward>=min_rew:
                                l = len(filtered_data)
                                filtered_data[l] = filtered_actions                     
                        else:
                            l = len(filtered_data)
                            filtered_data[l] = filtered_actions
                        break

            with open(fi,'w') as outfile:
                outfile.write(to_json(filtered_data))

            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='retro game to use')
    parser.add_argument('--state', help='retro state to start from')
    parser.add_argument('--scenario', help='scenario to use', default='contest')
    parser.add_argument('--mode', help='mode (play or filter)', default='play')
    args = parser.parse_args()
    p = PlayBack(game=args.game,state=args.state,scenario=args.scenario)
    if args.mode == 'play':
        p.play_json()
    elif args.mode == 'filter':
        p.filter_json(min_rew=9000)
