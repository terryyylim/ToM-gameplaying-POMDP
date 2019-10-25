from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import bz2
import copy
import pickle
from pathlib import Path
import pygame as pg

from overcooked_pygame.settings import *
from overcooked_pygame.sprites import *
from overcooked_pygame.overcooked import Game

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.map_configs import *
from ipomdp.agents.agent_configs import *
from ipomdp.agents.base_agent import OvercookedAgent
from ipomdp.overcooked import *
from ipomdp.helpers import *

from helpers import make_video_from_image_dir


def find_latest_file(path: str) -> str:
    files = Path(path).glob('*.pbz2')
    latest_file = max(files, key=lambda file: file.stat().st_ctime)

    return latest_file.as_posix()

def load_input_file(input_file: str) -> Any:
    if input_file.endswith('pbz2'):
        handler = bz2.open(input_file, 'rb')
    elif input_file.endswith('pkl'):
        handler = open(input_file, 'rb')

    with handler as f:
        data = pickle.load(f)

    return data

def get_deep_copy(obj):
    return copy.deepcopy(obj)

def generate_episodes(latest_episodes_world_state):
    episode_count = 1
    for episode in latest_episodes_world_state:
        # episode = latest_episodes_world_state[episode_count]
        PLAYERS, PLATES, POTS = {}, {}, {}
        INGREDIENTS = []
        TABLE_TOPS_COPY = get_deep_copy(TABLE_TOPS)
        
        RETURN_STATION_OCCUPIED = False
        RETURN_STATION = {
            'state': 'empty',
            'coords': episode['return_counter']
        }

        for agent in episode['agents']:
            PLAYERS[agent.agent_id] = {
                'holding': agent.holding,
                'coords': agent.location
            }
        for plate in episode['plate']:
            if plate.location in TABLE_TOPS_COPY:
                TABLE_TOPS_COPY.remove(plate.location)
            if plate.location == episode['return_counter']:
                RETURN_STATION_OCCUPIED = True
            PLATES[plate.plate_id] = {
                'state': plate.state,
                'coords': plate.location
            }
        for pot in episode['pot']:
            if pot.ingredient_count:
                for k,v in pot.ingredient_count.items():
                    pot_ingredient, pot_ingredient_count = k, v 
            else:
                pot_ingredient, pot_ingredient_count = None, 0
            POTS[pot.pot_id] = {
                'ingredient': pot_ingredient,
                'ingredient_count': pot_ingredient_count,
                'coords': pot.location
            }

        for ingredient in episode['ingredients']:
            if ingredient.location in TABLE_TOPS_COPY:
                TABLE_TOPS_COPY.remove(ingredient.location)
            INGREDIENTS.append([
                ingredient.name,
                ingredient.state,
                ingredient.location
            ])
        
        if RETURN_STATION_OCCUPIED:
            RETURN_STATION = {
                'state': 'filled',
                'coords': episode['return_counter']
            }
        CHOPPING_BOARDS = [chopping_board.location for chopping_board in episode['chopping_board'] if chopping_board.state != 'taken']

        g = Game()

        g.new(
            PLAYERS, TABLE_TOPS_COPY, INGREDIENTS, CHOPPING_BOARDS, PLATES, POTS, INGREDIENTS_STATION,
            SERVING_STATION, RETURN_STATION, EXTINGUISHER, TRASH_BIN
        )
        g.update()
        g.draw()
        pg.image.save(g.screen, f'episodes/episode_{episode_count}.png')
        episode_count += 1

def main():
    latest_episodes_world_state_file = find_latest_file('episodes_cache')
    print(latest_episodes_world_state_file)
    
    latest_episodes_world_state = load_input_file(latest_episodes_world_state_file)

    # Remove the previous images and video
    images_dir = 'episodes'
    videos_dir = 'videos'
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            os.remove(images_dir+'/'+file)
    for file in os.listdir(videos_dir):
        if file.endswith('/trajectory_new.mp4'):
            os.remove(videos_dir+'/trajectory_new.mp4')

    print(f'Starting episode generation')
    generate_episodes(latest_episodes_world_state)
    make_video = True

    if make_video:
        video_path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
        image_path = os.path.abspath(os.path.dirname(__file__)) + '/episodes'
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        fps = 1
        video_name = 'trajectory_new'
        make_video_from_image_dir(
            video_path,
            image_path,
            video_name,
            fps
        )


if __name__ == "__main__":
    main()
