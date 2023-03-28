# Choose the map
import maps.modeling_no_line_1 as selected_map

# ==================== Colour definition ====================
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BACKGROUND_SAND = (194,178,128)
BROWN = (160, 82, 4)
SCOREBOARD_BG = (143, 186, 200)

# ======================= Game Settings =======================
WIDTH = 896 # 13x32  # 16 * 64 (832) or 32 * 32 or 64 * 16
HEIGHT = 960 # 9x32 # 16 * 48 (640) or 32 * 24 or 64 * 12
FPS = 60
TITLE = "Water Collecting"
BGCOLOR = BACKGROUND_SAND

TILESIZE = 64
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

SCOREBOARD_SCORE = (14,0)
SCOREBOARD_TIMER = (14,8)
SCOREBOARD = [
    (14,1), (14,2), (14,3), (14,4), (14,5), (14,6), (14,7), (14,9), (14,10), (14,11), (14,12), (14,13)
]

TERMINATING_EPISODE = 250 

# ====================== Actions ======================= 
# MAP_ACTIONS = {
#     'MOVE_LEFT': [0, -1],
#     'MOVE_RIGHT': [0, 1],
#     'MOVE_UP': [-1, 0],
#     'MOVE_DOWN': [1, 0],
#     'MOVE_DIAGONAL_LEFT_UP': [-1, -1],
#     'MOVE_DIAGONAL_RIGHT_UP': [-1, 1],
#     'MOVE_DIAGONAL_LEFT_DOWN': [1, -1],
#     'MOVE_DIAGONAL_RIGHT_DOWN': [1, 1],
#     'STAY': [0,0]
# }

#map actions irrelevant now

# ====================== Chosen Map ======================
MAP = selected_map.MAP
#AI_AGENTS = selected_map.AI_AGENTS
CACTUS = selected_map.CACTUS
PALM_TREES = selected_map.PALM_TREES
ROCKS = selected_map.ROCKS
RIVER_WATER = selected_map.RIVER_WATER
STILL_WATER = selected_map.STILL_WATER
WELL = selected_map.WELL
BARREL = selected_map.BARREL
EP_DATA = selected_map.EP_DATA
