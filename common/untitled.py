def actions_from_human_data(game_state, scenario='scenario', play_path='../play/human', order=True):

    all_actions = []
    buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
    actions = {}

    for game, state in set(game_state):

        path = os.path.join(play_path,game,scenario)

        fi = os.path.join(path,'{}-{}.json'.format(game,state))

        with open(fi) as f:
            actions[('game','state')] = json.load(f)

        for ep in actions[('game','state')]:

            arr = SonicActions.from_sonic_config(actions[ep])
            actions[('game','state')][ep] = arr

            all_actions+=arr.actions.tolist()

    unique_actions = [np.array(a, 'int') for a in list(set(map(tuple, all_actions)))]

    if order:

        left = [0] * 12
        left[buttons.index("LEFT")] = 1
        left = tuple(left)
        right = [0] * 12
        right[buttons.index("RIGHT")] = 1
        right = tuple(right)
        jump = [0] * 12
        jump[buttons.index("B")] = 1
        jump = tuple(jump)

        core_actions = tuple([left, right, jump])
        actions_tup = tuple(map(tuple, unique_actions))

        for core_action in core_actions:
            if core_action not in actions_tup:
                raise ValueError("core actions missing in action set")

        sort_index = []

        for act in actions_tup:
            if act in core_actions:
                sort_index.append(core_actions.index(act))
            else:
                sort_index.append(len(sort_index)+len(core_actions))

        unique_actions = [np.array(a) for _,a in sorted(zip(sort_index,actions_tup))]

    return unique_actions, actions