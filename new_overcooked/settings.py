# ==================== Colour definition ====================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BACKGROUND_BLUE = (213, 226, 237)
BROWN = (160, 82, 4)

# ======================= Game Settings =======================
WIDTH = 416 # 13x32  # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 288 # 9x32 # 16 * 48 or 32 * 24 or 64 * 12
FPS = 60
TITLE = "Overcooked Simulation"
BGCOLOR = BACKGROUND_BLUE

TILESIZE = 32
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# ======================= World State =======================
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

# ========================= Recipes ===========================
RECIPES_1 = ['onion_soup'] #'tomato_soup'
RECIPES_INFO_1 = {
    'onion_soup': {
        'ingredient': 'onion',
        'count': 3
    },
    'tomato_soup': {
        'ingredient': 'tomato',
        'count': 3
    }
}
RECIPES_INGREDIENTS_TASK_1 = {
    'onion_soup': {
        'onion': [('pick', 'fresh'), ('chop', 'unchopped'), ('cook', 'chopped')]
    }
}
RECIPES_ACTION_MAPPING_1 = {
    'onion_soup': {
        'PICK': 0,
        'CHOP': 1,
        'COOK': 2,
        'SCOOP': 3,
        'SERVE': 4
    },
    'tomato_soup': {
        'PICK': 5,
        'CHOP': 6,
        'COOK': 7,
        'SCOOP': 8,
        'SERVE': 9
    }
}

# ================== Game Background Initialization ==================
PLAYERS_1 = {
    1: {
        'holding': None,
        'coords': (2,5)
    },
    2: {
        'holding': None,
        'coords': (2,8)
    }
}

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

WALLS_1 = [
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

TABLE_TOPS_1 = [
    (0,0), (0,1), (0,2), (0,4), (0,5), (0,6), (0,7), (0,9), (0,10), (0,11), (0,12),
    (1,0),
    (2,0), (2,12),
    (3,0), (3,12),
    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,12),
    (5,12),
    (6,12),
    (8,0), (8,1), (8,2), (8,4), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12)
]

INGREDIENTS_1 = []

CHOPPING_BOARDS_1 = {
    1: {
        'state': 'empty',
        'coords': (8,3)
    },
    2: {
        'state': 'empty',
        'coords': (8,5)
    }
}

PLATES_1 = {
    1: {
        'state': 'empty', 
        'coords': (8,9)
    },
    2: {
        'state': 'empty',
        'coords': (8,10)
    }
}

POTS_1 = {
    1: {
        'ingredient': None,
        'ingredient_count': 0,
        'coords': (0,8)
    }
}

INGREDIENTS_STATION_1 = {
    'onion': (0,3)
}

SERVING_STATION_1 = [(6,0), (7,0)]
RETURN_STATION_1 = {
    'state': 'empty',
    'coords': (5,0)
}
EXTINGUISHER_1 = (1,12)
TRASH_BIN_1 = (7,12)

HUMAN_AGENTS_1 = {
    '1': {
        'coords': (2,5),
    }
}

AI_AGENTS_1 = {
    '1': {
        'coords': (2,8),
        'ToM': False
    }
}

# ====================== Actions ======================= 
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


RECIPES = RECIPES_1
RECIPES_INFO = RECIPES_INFO_1
RECIPES_INGREDIENTS_TASK = RECIPES_INGREDIENTS_TASK_1
RECIPES_ACTION_MAPPING = RECIPES_ACTION_MAPPING_1
HUMAN_AGENTS = HUMAN_AGENTS_1
AI_AGENTS = AI_AGENTS_1
TABLE_TOPS = TABLE_TOPS_1
# PLAYERS = PLAYERS_1
CHOPPING_BOARDS = CHOPPING_BOARDS_1
ITEMS_INITIALIZATION = ITEMS_INITIALIZATION_1
INGREDIENTS_INITIALIZATION = INGREDIENTS_INITIALIZATION_1
INGREDIENTS_STATION = INGREDIENTS_STATION_1
SERVING_STATION = SERVING_STATION_1
# RETURN_STATION = RETURN_STATION_1
EXTINGUISHER = EXTINGUISHER_1
TRASH_BIN = TRASH_BIN_1
WALLS = WALLS_1
MAP_ACTIONS = MAP_ACTIONS_1
WORLD_STATE = WORLD_STATE_1