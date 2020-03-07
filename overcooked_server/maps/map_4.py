# ======================= World State =======================
# As of now: DO NOT ALLOW DROPPING ITEMS INFRONT OF TASK_PERFORMING CELL
MAP = 'map4'
COMPLEX_RECIPE = True

WORLD_STATE = {    
    'invalid_stay_cells': [(1,1), (2,1), (3,1), (1,10), (1,11)],
    'invalid_movement_cells': [],
    'valid_item_cells': [
        (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9),
        (1,12),
        (2,12),
        (3,12),
        (4,0), (4,12),
        (5,0), (5,12),
        (6,0), (6,12),
        (7,0),
        (8,1), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9),
    ],
    'valid_movement_cells': [
        (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11),
        (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11),
        (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11),
        (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (4,11),
        (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11),
        (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11),
        (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10), (7,11),
    ],
    'ingredient_carrot': [(1,0)],
    'ingredient_onion': [(2,0)],
    'ingredient_spinach': [(3,0)],
    'service_counter': [(8,2)],
    'return_counter': [(8,3)],
}

# ========================= Recipes ===========================
RECIPES = ['guillotine_soup']
RECIPES_INFO = {
   'guillotine_soup': {
       'carrot': 1,
       'onion': 1,
       'spinach': 1
    }
}
RECIPES_ACTION_MAPPING = {
    'guillotine_soup': {
        'carrot': {
            'PICK': 0,
            'CHOP': 1,
            'COOK': 2,
        },
        'onion': {
            'PICK': 3,
            'CHOP': 4,
            'COOK': 5,
        },
        'spinach': {
            'PICK': 6,
            'CHOP': 7,
            'COOK': 8,
        },
        'general': {
            'SCOOP': 9,
            'SERVE': 10
        }
    }
}
RECIPE_ACTION_NAME = {
    'guillotine_soup': [0,1,2,3,4,5,6,7,8,9,10],
}
INGREDIENT_ACTION_NAME = {
    'carrot': [0,1,2],
    'onion': [3,4,5],
    'spinach': [6,7,8]
}
FLATTENED_RECIPES_ACTION_MAPPING = {
    'PICK': [0,3,6],
    'CHOP': [1,4,7],
    'COOK': [2,5,8],
    'SCOOP': [9],
    'SERVE': [10]
}

# ================== Game Background Initialization ==================
ITEMS_INITIALIZATION = {
    'chopping_board': [(8,10),(8,11)],
    'plate': [(5,0),(6,0)],
    'pot': [(0,10),(0,11)]
}

INGREDIENTS_INITIALIZATION = {
    # properties of ingredients
    'carrot': {
        'location': [(1,0)]
    },
    'onion': {
        'location': [(2,0)]
    },
    'spinach': {
        'location': [(3,0)]
    }
}

WALLS = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12),
    (1,0), (1,12),
    (2,0), (2,12),
    (3,0), (3,12),
    (4,0), (4,12),
    (5,0), (5,12),
    (6,0), (6,12),
    (7,0), (7,12),
    (8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12)
]

TABLE_TOPS = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,12),
    (1,12),
    (2,12),
    (3,12),
    (4,0), (4,12),
    (5,0), (5,12),
    (6,0), (6,12),
    (7,0), (7,12),
    (8,0), (8,1), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,12)
]

CHOPPING_BOARDS = {
    1: {
        'state': 'empty',
        'coords': (8,10)
    },
    2: {
        'state': 'empty',
        'coords': (8,11)
    }
}

PLATES = {
    1: {
        'state': 'empty', 
        'coords': (5,0)
    },
    2: {
        'state': 'empty',
        'coords': (6,0)
    }
}

INGREDIENTS_STATION = {
    'carrot': [(1,0)],
    'onion': [(2,0)],
    'spinach': [(3,0)]
}

SERVING_STATION = [(8,2)]
RETURN_STATION = {
    'state': 'empty',
    'coords': (8,3)
}

HUMAN_AGENTS = {
    '1': {
        'coords': (2,3),
    },
    '2': {
        'coords': (2,4),
    }
}

AI_AGENTS = {
    '1': {
        'coords': (2,4),
        'ToM': False
    },
    '2': {
        'coords': (2,3),
        'ToM': False
    },
    '3': {
        'coords': (3,3),
        'ToM': False
    },
    '4': {
        'coords': (3,4),
        'ToM': False
    }
}

# ================== Game Scoreboard Initialization ==================
SCOREBOARD_SCORE = (9,0)
SCOREBOARD_ORDERS = (9,4)
SCOREBOARD = [
    (9,1), (9,2), (9,3), (9,5), (9,6), (9,7), (9,8), (9,9), (9,10), (9,11), (9,12)
]
QUEUE_EPISODES = 50