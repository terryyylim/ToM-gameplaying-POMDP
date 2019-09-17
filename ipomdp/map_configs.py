ACTIONS = {
    'MOVE_LEFT': [-1, 0],
    'MOVE_RIGHT': [0, 1],
    'MOVE_UP': [0, -1],
    'MOVE_DOWN': [0, 1],
    'MOVE_DIAGONAL_LEFT_UP': [-1, -1],
    'MOVE_DIAGONAL_RIGHT_UP': [1, -1],
    'MOVE_DIAGONAL_LEFT_DOWN': [-1, 1],
    'MOVE_DIAGONAL_RIGHT_DOWN': [1, 1],
    'STAY': [0, 0],
    'PICK': [0, 0],
    'DROP': [0, 0]
}

DEFAULT_COLOURS = {
    ' ': [0, 0, 0],
    '@': [245, 245, 220],
    'B': [240,248,255],
    'C': [245,255,250],
    'D': [218,165,32],
    'E': [216, 30, 54],
    'P': [255,255,255],
    'S': [211,211,211],
    'T': [210,180,140],

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