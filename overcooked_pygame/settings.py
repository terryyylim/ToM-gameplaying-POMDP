# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BACKGROUND_BLUE = (213, 226, 237)
BROWN = (160, 82, 4)

# game settings
WIDTH = 416 # 13x32  # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 288 # 9x32 # 16 * 48 or 32 * 24 or 64 * 12
FPS = 60
TITLE = "Overcooked Simulation"
BGCOLOR = BACKGROUND_BLUE

TILESIZE = 32
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE


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


TABLE_TOPS = TABLE_TOPS_1
PLAYERS = PLAYERS_1
CHOPPING_BOARDS = CHOPPING_BOARDS_1
PLATES = PLATES_1
POTS = POTS_1
INGREDIENTS = INGREDIENTS_1
INGREDIENTS_STATION = INGREDIENTS_STATION_1
SERVING_STATION = SERVING_STATION_1
RETURN_STATION = RETURN_STATION_1
EXTINGUISHER = EXTINGUISHER_1
TRASH_BIN = TRASH_BIN_1