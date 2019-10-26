AGENTS_INITIALIZATION_1 = {
    '1': {
        'coords': (2,8),
        'ToM': False
    }
}
RECIPES_1 = ['onion_soup']
RECIPES_INGREDIENTS_TASK_1 = {
    'onion_soup': {
        'onion': [('pick', 'fresh'), ('chop', 'unchopped'), ('cook', 'chopped')]
    }
}

MAP_ACTIONS_1 = {
    'MOVE_LEFT': [0, -1],
    'MOVE_RIGHT': [0, 1],
    'MOVE_UP': [-1, 0],
    'MOVE_DOWN': [1, 0],
    'MOVE_DIAGONAL_LEFT_UP': [-1, -1],
    'MOVE_DIAGONAL_RIGHT_UP': [-1, 1],
    'MOVE_DIAGONAL_LEFT_DOWN': [1, -1],
    'MOVE_DIAGONAL_RIGHT_DOWN': [1, 1],
    'STAY': [0,0]
}

DEFAULT_COLOURS_1 = {
    ' ': [0, 0, 0],
    '@': [245, 245, 220],
    'B': [0,0,255],
    'C': [210,180,140],
    'D': [218,165,32],
    'E': [216, 30, 54],
    'O': [244,164,96],
    'Z': [205,133,63], # Taken Unchopped Onion
    'X': [188,143,143], # Taken Chopped Onion
    'P': [255,255,255],
    'S': [211,211,211],
    'T': [139,69,19],
    'W': [119,136,153],

    # Agent colours
    '1': [34,139,34],
    '2': [60,179,113],
    '3': [0,255,127],
    '4': [0,255,0]
}

"""
  - Floor 
@ - Table Top
A - Agent Spawn Point
B - Basin
C - Chopping Board
D - Dirty Plate Return Point
E - Extinguisher
O - Onion
P - Plate
S - Serving Point
T - Trash Bin
W - Cooking Ware
"""
OVERCOOKED_MAP_1 = [
    '@@@O@@@@W@@@@',
    '@           E',
    '@   A   A   @',
    '@           B',
    '@@@@@@@@@@  @',
    'D           @',
    'S           @',
    'S           T',
    '@@@C@C@@@PP@@'
]

OVERCOOKED_MAP = OVERCOOKED_MAP_1
AGENTS_INITIALIZATION = AGENTS_INITIALIZATION_1
MAP_ACTIONS = MAP_ACTIONS_1
RECIPES_INGREDIENTS_TASK = RECIPES_INGREDIENTS_TASK_1
RECIPES = RECIPES_1
DEFAULT_COLOURS = DEFAULT_COLOURS_1