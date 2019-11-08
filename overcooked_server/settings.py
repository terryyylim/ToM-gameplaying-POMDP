# Choose the map
import maps.map_2 as selected_map

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
SCOREBOARD_BG = (143, 186, 200)

# ======================= Game Settings =======================
WIDTH = 416 # 13x32  # 16 * 64 or 32 * 32 or 64 * 16
HEIGHT = 320 # 9x32 # 16 * 48 or 32 * 24 or 64 * 12
FPS = 60
TITLE = "Overcooked Simulation"
BGCOLOR = BACKGROUND_BLUE

TILESIZE = 32
GRIDWIDTH = WIDTH / TILESIZE
GRIDHEIGHT = HEIGHT / TILESIZE

# ====================== Actions ======================= 
MAP_ACTIONS = {
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


# ====================== Chosen Map ======================
RECIPES = selected_map.RECIPES
RECIPES_INFO = selected_map.RECIPES_INFO
RECIPES_ACTION_MAPPING = selected_map.RECIPES_ACTION_MAPPING
RECIPE_ACTION_NAME = selected_map.RECIPE_ACTION_NAME
INGREDIENT_ACTION_NAME = selected_map.INGREDIENT_ACTION_NAME
FLATTENED_RECIPES_ACTION_MAPPING = selected_map.FLATTENED_RECIPES_ACTION_MAPPING
HUMAN_AGENTS = selected_map.HUMAN_AGENTS
AI_AGENTS = selected_map.AI_AGENTS
TABLE_TOPS = selected_map.TABLE_TOPS
CHOPPING_BOARDS = selected_map.CHOPPING_BOARDS
ITEMS_INITIALIZATION = selected_map.ITEMS_INITIALIZATION
INGREDIENTS_INITIALIZATION = selected_map.INGREDIENTS_INITIALIZATION
INGREDIENTS_STATION = selected_map.INGREDIENTS_STATION
SERVING_STATION = selected_map.SERVING_STATION
WALLS = selected_map.WALLS
WORLD_STATE = selected_map.WORLD_STATE
SCOREBOARD_SCORE = selected_map.SCOREBOARD_SCORE
SCOREBOARD_ORDERS = selected_map.SCOREBOARD_ORDERS
SCOREBOARD = selected_map.SCOREBOARD
QUEUE_EPISODES = selected_map.QUEUE_EPISODES