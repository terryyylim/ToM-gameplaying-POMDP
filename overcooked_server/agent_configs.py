"""
Actions
-------
TO-DO: Actions that can be performed.
"""
ACTIONS = {
    0: 'MOVE_LEFT',
    1: 'MOVE_RIGHT',
    2: 'MOVE_UP',
    3: 'MOVE_DOWN',
    4: 'MOVE_DIAGONAL_LEFT_UP',
    5: 'MOVE_DIAGONAL_RIGHT_UP',
    6: 'MOVE_DIAGONAL_LEFT_DOWN',
    7: 'MOVE_DIAGONAL_RIGHT_DOWN',
    8: 'STAY',
    9: 'PICK',
    10: 'CHOP',
    11: 'COOK',
    12: 'SCOOP',
    13: 'SERVE',
    14: 'DROP'
}


"""
Rewards
-------
TO-DO: Rewards that have floating values.

* WHEN DIAGONAL MOVEMENT COSTS -1, AGENT WILL GET STUCK.
"""
REWARDS_1 = {
    'MOVE_LEFT': -1,
    'MOVE_RIGHT': -1,
    'MOVE_UP': -1,
    'MOVE_DOWN': -1,
    'MOVE_DIAGONAL_LEFT_UP': -2,
    'MOVE_DIAGONAL_RIGHT_UP': -2,
    'MOVE_DIAGONAL_LEFT_DOWN': -2,
    'MOVE_DIAGONAL_RIGHT_DOWN': -2,
    'STAY': -2,
    'PICK': 10, # consider changing to 'create'
    'CHOP': 20,
    'COOK': 40,
    'SCOOP': 50,
    'SERVE': 100,
    'DROP': 0
}

REWARDS = REWARDS_1