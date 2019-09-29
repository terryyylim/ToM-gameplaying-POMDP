AGENTS_INITIALIZATION_1 = [(2,4), (2,8)] # (8,6), (6,6)
RECIPES_1 = ['onion_soup']
RECIPES_INGREDIENTS_TASK_1 = {
    'onion_soup': {
        'onion': [('pick', 'fresh'), ('slice', 'unchopped'), ('cook', 'chopped')]
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
    'P': [255,255,255],
    'S': [211,211,211],
    'T': [139,69,19],

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
P - Plate
S - Serving Point
T - Trash Bin
"""
OVERCOOKED_MAP_1 = [
    '@@@@@@@@@@@@@',
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