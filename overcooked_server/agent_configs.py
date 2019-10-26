BARRIERS_1 = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
    (1,0), (1,12),
    (2,0), (2,12),
    (3,0), (3,12),
    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,12),
    (5,0), (5,12),
    (6,0), (6,12),
    (7,0), (7,12),
    (8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12)
]

ITEMS_INITIALIZATION_1 = {
    'chopping_board': [(8,3), (8,5)],
    'extinguisher': [(1,12)],
    'plate': [(8,9), (8,10)],
    'pot': [(0,8)],
    'stove': [(0,8)],
}

INGREDIENTS_INITIALIZATION_1 = {
    # properties of ingredients
    'onion': {
        'location': [(0,3)],
        'state': 'fresh',
        'category': 'ingredient',
        'is_raw': True
    }
}

RECIPES_INGREDIENTS_COUNT_1 = {
    'onion_soup': {
        'onion': 3
    }
}

RECIPES_SERVE_TASK_1 = {
    'onion_soup': [('plate', 'cooked'), ('scoop', 'cooked'), ('serve', 'cooked')]
}

# As of now: DO NOT ALLOW DROPPING ITEMS INFRONT OF TASK_PERFORMING CELL
WORLD_STATE_1 = {
    'valid_optimal_table_tops': [
        (1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4)
    ],
    'invalid_optimal_table_tops': [],
    'valid_item_cells': [
        (0,1), (0,2), (0,4), (0,5), (0,6), (0,7), (0,9), (0,10), (0,11),
        (1,0), (1,1), (1,2), (1,4), (1,5), (1,6), (1,7), (1,9), (1,10), (1,11),
        (2,0), (2,1), (2,2), (2,3), (2,5), (2,6), (2,7), (2,9), (2,10), (2,11), (2,12),
        (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10),
        (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10),
        (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12),
        (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12),
        (7,2), (7,4), (7,6), (7,7), (7,8), (7,9), (7,10),
        (8,2), (8,4), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11)
    ],
    'temporary_valid_item_cells': [
        (0,1), (0,2), (0,4), (0,5), (0,6), (0,7), (0,9), (0,10), (0,11),
        (1,0),
        (2,0), (2,12),
        (3,0), 
        (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8),
        (5,12),
        (6,12),
        (8,1), (8,2), (8,4), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11)
    ],
    'valid_movement_cells': [
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11),
        (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11),
        (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11),
        (4,10), (4,11),
        (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11),
        (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11),
        (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10), (7,11),
    ],
    'ingredient_onion': [(0,3)],
    'service_counter': [(6,0), (7,0)],
    'return_counter': [(5,0)],
}

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
    'CHOP': 30,
    'COOK': 45,
    'SCOOP': 50,
    'SERVE': 100,
    'DROP': 0
}

INGREDIENTS_STATION_1 = {
    'onion': (0,3)
}

# Current World State
WORLD_STATE = WORLD_STATE_1
BARRIERS = BARRIERS_1
ITEMS_INITIALIZATION = ITEMS_INITIALIZATION_1
INGREDIENTS_STATION = INGREDIENTS_STATION_1
INGREDIENTS_INITIALIZATION = INGREDIENTS_INITIALIZATION_1
RECIPES_INGREDIENTS_COUNT = RECIPES_INGREDIENTS_COUNT_1
RECIPES_SERVE_TASK = RECIPES_SERVE_TASK_1
REWARDS = REWARDS_1