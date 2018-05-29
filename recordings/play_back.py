import retro
import glob

for f in glob.glob('.*bk2'):
    movie = retro.Movie(f)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.STATE_NONE, use_restricted_actions=retro.ACTIONS_ALL)
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for i in range(env.NUM_BUTTONS):
            keys.append(movie.get_key(i))
        _obs, _rew, _done, _info = env.step(keys)
        env.render()
