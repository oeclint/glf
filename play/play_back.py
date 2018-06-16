import glob
import os
import json
from retro_contest.local import make as contest_make
from retro import make
import argparse
from glf.common.sonic_util import SonicActions

class PlayBack(object):

    def __init__(self, game, state, scenario, root='human'):
        self.path = os.path.join(root,game,scenario)
        self.game = game
        self.state = state
        self.scenario = scenario

    def play_bk2(self, render=False, write=True):

        keysdict = {}

        for f in glob.glob(os.path.join(self.path,'{}-{}*.bk2'.format(game,state))):

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
            with open('{}-{}.json'.format(self.game,self.state), 'w') as outfile:
                 outfile.write(to_json(keysdict))

    def play_json(self, maker, render=True):

        with open(os.path.join(self.path,'{}-{}.json'.format(self.game,self.state))) as f:
            data = json.load(f)

        env = maker(game=self.game, state=self.state, scenario=self.scenario)
        for ep in data:
            obs = env.reset()
            actions = SonicActions(data[ep])
            for action in actions.data:
                obs, rew, done, info = env.step(action)
                env.render()
                if done:
                    break
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='retro game to use')
    parser.add_argument('--state', help='retro state to start from')
    parser.add_argument('--scenario', help='scenario to use', default='contest')
    args = parser.parse_args()
    p = PlayBack(game=args.game,state=args.state,scenario=args.scenario)
  #  p.play_bk2()
    p.play_json(make)
