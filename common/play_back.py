import glob
import os
import json
from retro_contest.local import make

def to_json(o):
    str_out = []
    s4 = '    '
    s8 = s4*2
    for k, v in o.items():
        kv=''
        kv += '{}"{}":\n'.format(s4,k)
        f = (s8+'[{}]\n').format( "{},\n " + (s8 + "{},\n ")*(len(v)-2) + s8 + "{}")
        kv += f.format(*v)
        str_out.append(kv)

    return "{{\n{}}}".format(','.join(str_out))

class PlayBack(object):

    def __init__(self, game, state, root='../play_sonic/human', scenario='contest'):
        self.bk2_path = os.path.join(root,game,scenario,'-'.join([game,state]))
        self.game = game
        self.state = state

    def play_bk2(self, render=False, write=True):

        keysdict = {}

        for f in glob.glob(self.bk2_path+'*.bk2'):

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

    def play_json(self, render=True):

        with open('{}-{}.json'.format(self.game,self.state)) as f:
            data = json.load(f)

        env = make(game=self.game, state=self.state)
        obs = env.reset()
        for ep in data:
            for action in data[ep]:
                obs, rew, done, info = env.step(action)
                env.render()
                if done:
                    obs = env.reset()
                    break
        

if __name__ == '__main__':
    p = PlayBack('SonicTheHedgehog-Genesis','SpringYardZone.Act3')
    p.play_bk2()
  #  p.play_json()
