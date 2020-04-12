from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import click
import copy
import sys
import pandas as pd
import pygame as pg
from datetime import datetime

from map_env import MapEnv
from overcooked_env import OvercookedEnv
from sprites import *
from settings import *

import helpers

class Game:
    def __init__(
        self,
        num_ai_agents: int=1,
        is_simulation: bool=False,
        simulation_episodes: int=500,
        is_tom: bool=False,
        experiment_id: str='1'
    ) -> None:
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.is_simulation = is_simulation
        self.is_tom = is_tom
        self.experiment_id = experiment_id

        AI_AGENTS_TO_INITIALIZE = {}
        for idx in range(1, num_ai_agents+1):
            idx = str(idx)
            AI_AGENTS_TO_INITIALIZE[idx] = AI_AGENTS[idx]

        # Logs saving
        game_folder = os.path.dirname(__file__)
        self.experiment_folder = os.path.join(game_folder, experiment_id)
        helpers.check_dir_exist(self.experiment_folder)
        self.images_folder = os.path.join(self.experiment_folder, 'images')
        helpers.check_dir_exist(self.images_folder)

        if self.is_simulation:
            self.env = OvercookedEnv(
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                queue_episodes=QUEUE_EPISODES
            )
            self.load_data()

            self.results_filename = 'results/' + self.env.results_filename + '.csv'
            self.results = defaultdict(int)
            self.results_col = []
            for i in range(TERMINATING_EPISODE+1):
                if i%50 == 0:
                    self.results[str(i)] = 0
                    self.results_col.append(str(i))

            self.run_simulation(simulation_episodes)
        else:
            final_HUMAN_AGENTS = {k:v for k,v in HUMAN_AGENTS.items() if k == '1'}
            self.env = OvercookedEnv(
                human_agents=final_HUMAN_AGENTS,
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                queue_episodes=QUEUE_EPISODES
            )
            self.info_df = pd.DataFrame(
                columns=[
                    'episode', 'player_coords', 'agent_coords', 'player_action', 'agent_action', 'available_orders', 'score', 'total_score'
                ]
            )
            self.load_data()
            self.results_filename = self.experiment_folder + '/' + self.env.results_filename + '.csv'
        self.results = defaultdict(int)
        self.results_col = []
        for i in range(TERMINATING_EPISODE+1):
            if i%50 == 0:
                self.results[str(i)] = 0
                self.results_col.append(str(i))

    def _deep_copy(self, obj):
        return copy.deepcopy(obj)

    def load_data(self):
        episode = self.env.world_state

        self.PLAYERS, self.PLATES, self.POTS = {}, {}, {}
        self.INGREDIENTS = []
        self.TABLE_TOPS = self._deep_copy(TABLE_TOPS)
        
        RETURN_STATION_OCCUPIED = False
        self.RETURN_STATION = {
            'state': 'empty',
            'coords': episode['return_counter']
        }

        for agent in episode['agents']:
            self.PLAYERS[agent.id] = {
                'holding': agent.holding,
                'coords': agent.location
            }
        for plate in episode['plate']:
            if plate.location in self.TABLE_TOPS:
                self.TABLE_TOPS.remove(plate.location)
            if plate.location == episode['return_counter']:
                RETURN_STATION_OCCUPIED = True
            self.PLATES[plate.plate_id] = {
                'state': plate.state,
                'coords': plate.location
            }
        for pot in episode['pot']:
            if pot.ingredient_count:
                pot_ingredient_count = pot.ingredient_count
            else:
                pot_ingredient_count = defaultdict(int)
            self.POTS[pot.pot_id] = {
                'ingredient_count': pot_ingredient_count,
                'coords': pot.location
            }

        for ingredient in episode['ingredients']:
            if ingredient.location in self.TABLE_TOPS:
                self.TABLE_TOPS.remove(ingredient.location)
            self.INGREDIENTS.append([
                ingredient.name,
                ingredient.state,
                ingredient.location
            ])
        
        if RETURN_STATION_OCCUPIED:
            self.RETURN_STATION = {
                'state': 'filled',
                'coords': episode['return_counter']
            }
        self.CHOPPING_BOARDS = [chopping_board.location for chopping_board in episode['chopping_board'] if chopping_board.state != 'taken']
        self.INGREDIENTS_STATION = INGREDIENTS_STATION
        self.SERVING_STATION = SERVING_STATION
        self.WALLS = WALLS

    def new(
        self,
        players: Dict[int, Tuple[int,int]],
        table_tops: List[Tuple[int,int]],
        ingredients,
        chopping_boards: List[Tuple[int,int]],
        plates: Dict[int, List[Any]],
        pots: Dict[int, List[Any]],
        ingredient_stations: Dict[str, Tuple[int,int]],
        serving_stations: List[Tuple[int,int]],
        return_station: List[Tuple[int,int]],
        walls: List[Tuple[int,int]]=WALLS,
        score: Tuple[int,int]=SCOREBOARD_SCORE,
        orders: Tuple[int,int]=SCOREBOARD_ORDERS,
        timer: Tuple[int,int]=SCOREBOARD_TIMER,
        scoreboard: List[Tuple[int,int]]=SCOREBOARD
    ) -> None:
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.table_tops = pg.sprite.Group()
        self.ingredients = pg.sprite.Group()
        self.chopping_boards = pg.sprite.Group()
        self.plates = pg.sprite.Group()
        self.pot_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.serving_stations = pg.sprite.Group()
        self.return_station = pg.sprite.Group()
        self.score = pg.sprite.Group()
        self.orders = pg.sprite.Group()
        self.timer = pg.sprite.Group()
        self.scoreboard = pg.sprite.Group()
        self.walls = walls
        self.player_count = len(players)

        for idx in range(1, self.player_count+1):
            idx = str(idx)
            if idx == '1':
                self.player_1 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '2':
                self.player_2 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '3':
                self.player_3 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '4':
                self.player_4 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
        for table_top_coord in table_tops:
            TableTop(self, table_top_coord[1], table_top_coord[0])
        for ingredient in ingredients:
            ingredient_name = ingredient[0]
            ingredient_state = ingredient[1]
            ingredient_coords = ingredient[2]
            Ingredients(self, ingredient_name, ingredient_state, ingredient_coords[1], ingredient_coords[0])
        for chopping_board_coord in chopping_boards:
            ChoppingBoardStation(self, chopping_board_coord[1], chopping_board_coord[0])
        for key, val in plates.items():
            plate_state = val['state']
            plate_coord = val['coords']
            PlateStation(self, plate_state, plate_coord[1], plate_coord[0])
        for key, val in pots.items():
            pot_ingredient_count = val['ingredient_count']
            pot_coord = val['coords']
            PotStation(self, pot_ingredient_count, pot_coord[1], pot_coord[0])
        for key, val in ingredient_stations.items():
            ingredient = key
            ingredient_station_coord = val
            for coords in ingredient_station_coord:
                IngredientStation(self, ingredient, coords[1], coords[0])
        for serving_station_coord in serving_stations:
            ServingStation(self, serving_station_coord[1], serving_station_coord[0])
            
        ReturnStation(self, return_station['state'], return_station['coords'][1], return_station['coords'][0])
        Score(self, score[1], score[0])
        Orders(self, orders[1], orders[0])
        Timer(self, timer[1], timer[0])
        for scoreboard_coord in scoreboard:
            ScoreBoard(self, scoreboard_coord[1], scoreboard_coord[0])

    def save_results(self):
        try:
            results_df = pd.read_csv(self.results_filename)
            new_row = pd.DataFrame([self.results], columns=self.results.keys())
            results_df = pd.concat([results_df, new_row], axis=0).reset_index()
        except FileNotFoundError:
            results_df = pd.DataFrame([self.results], columns=self.results.keys())
        results_df = results_df[self.results_col]
        results_df.to_csv(self.results_filename, index=False)

        if not self.is_simulation:
            self.info_df.to_csv(self.experiment_folder + '/experiments_' + self.env.results_filename + '.csv', index=False)

            # agent_types = [agent.is_inference_agent for agent in self.env.world_state['agents']]
            video_name_ext = helpers.get_video_name_ext(self.env.world_state['agents'], TERMINATING_EPISODE, MAP)
            # video_name_ext = helpers.get_video_name_ext(agent_types, TERMINATING_EPISODE, MAP)
            helpers.make_video_from_image_dir(
                self.experiment_folder,
                self.images_folder,
                video_name_ext
            )

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        
        while self.playing:
            if self.env.episode%50 == 0:
                self.results[str(self.env.episode)] = self.env.world_state['total_score']
            if self.env.episode >= TERMINATING_EPISODE:
                self.save_results()
                pg.display.quit()
                self.quit()
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.draw_grid()
        self.all_sprites.draw(self.screen)

        # Score Display (On top of sprites)
        font = pg.font.Font('freesansbold.ttf', 20)
        # current_score = self.env.world_state['explicit_rewards']['serve']
        current_score = self.env.world_state['total_score']
        current_order = self.env.world_state['order_count']
        episodes_left = TERMINATING_EPISODE - self.env.episode
        score = font.render(str(current_score), True, GREEN, SCOREBOARD_BG)
        order = font.render(str(current_order), True, GREEN, SCOREBOARD_BG)
        ep_countdown = font.render(str(episodes_left), True, GREEN, SCOREBOARD_BG)
        scoreRect = score.get_rect()
        orderRect = order.get_rect()
        countdownRect = ep_countdown.get_rect()
        scoreRect.center = (
            (SCOREBOARD_SCORE[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2
        )
        orderRect.center = (
            (SCOREBOARD_ORDERS[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2    
        )
        countdownRect.center = (
            (SCOREBOARD_TIMER[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2    
        )
        self.screen.blit(score, scoreRect)
        self.screen.blit(order, orderRect)
        self.screen.blit(ep_countdown, countdownRect)
        pg.display.flip()

    def events(self):    
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYUP:
                if event.key in [
                    pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
                    pg.K_COMMA, pg.K_PERIOD, pg.K_SLASH, pg.K_RSHIFT, pg.K_m,
                    pg.K_z, pg.K_x, pg.K_c, pg.K_v, pg.K_b, pg.K_n
                ]:
                    print(f'\nStart of episode {self.env.episode}')
                    goal_space = self.env.world_state['goal_space']
                    goal_info = self.env.world_state['goal_space_count']
                    print(f'Current goal space: \n{goal_space}\n')
                    print(f'Current goal info: \n{goal_info}\n')
                    print([agent.holding for agent in self.env.world_state['agents']])

                    player_action_validity, action_type, action_task, goal_id = self._check_action_validity(1, event.key)
                    player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
                    best_goals = self.env.find_agents_best_goal()
                    print('found best goals')
                    print(best_goals)

                    if not player_action_validity:
                        best_goals[player_object] = [-1, {'steps': [8], 'rewards': -2}]
                    else:
                        action_mapping, reward_mapping = self._get_action_mapping_info(event.key)
                        if action_type == 'movement':
                            best_goals[player_object] = [-1, {'steps': [action_mapping], 'rewards': reward_mapping}]
                        else:
                            # its a task_action being taken
                            # goal_id = player_object._find_suitable_goal(action_mapping, action_task)
                            best_goals[player_object] = [goal_id, {'steps': action_task, 'rewards': reward_mapping}]

                    print('best goals')
                    print(best_goals)
                    print('before rolling out')
                    print([agent.location for agent in self.env.world_state['agents']])

                    self.rollout(best_goals, self.env.episode)
                    self.load_data()
                    self.update()
                    self.draw()

                    self.new(
                        self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                        self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
                    )
                    if event.key == pg.K_ESCAPE:
                        self.quit()
                    print(f'Just completed episode {self.env.episode}')
                    print([agent.location for agent in self.env.world_state['agents']])
                    print([agent.holding for agent in self.env.world_state['agents']])
                    self.env.update_episode()
                    # pg.image.save(self.screen, f'episodes/episode_{self.env.episode}.png')
                else:
                    print('Not valid key press')

    def _get_pos(self, player_id):
        if player_id == 1:
            return self.player_1.y, self.player_1.x
        elif player_id == 2:
            return self.player_2.y, self.player_2.x

    def _check_action_validity(self, player_id, action):
        action_coords_mapping = {
            pg.K_LEFT: [0, -1],
            pg.K_RIGHT: [0, 1],
            pg.K_UP: [-1, 0],
            pg.K_DOWN: [1, 0],
            pg.K_COMMA: [-1, -1],
            pg.K_PERIOD: [-1, 1],
            pg.K_SLASH: [1, -1],
            pg.K_RSHIFT: [1, 1],
            pg.K_m: [0, 0]
        }
        movement_keys = [
            pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
            pg.K_COMMA, pg.K_PERIOD, pg.K_SLASH, pg.K_RSHIFT, pg.K_m
        ]
        valid_flag = False
        action_type = None

        player_pos = self._get_pos(player_id)
        if action in movement_keys:
            print(f'Its a movement!')
            action_type = 'movement'
            action_task = None
            temp_player_pos = [sum(x) for x in zip(list(player_pos), action_coords_mapping[action])]

            if temp_player_pos not in self.WALLS:
                valid_flag = True
            goal_id = -1
        else:
            # Do actions
            action_type = 'action'
            if action == pg.K_z:
                # Check if there's anything to pick in (UP, DOWN, LEFT, RIGHT)
                valid_flag, action_task, goal_id = self._check_pick_validity(player_id)
            elif action == pg.K_x:
                valid_flag, action_task, goal_id = self._check_chop_validity(player_id)
            elif action == pg.K_c:
                valid_flag, action_task, goal_id = self._check_cook_validity(player_id)
            elif action == pg.K_v:
                valid_flag, action_task, goal_id = self._check_scoop_validity(player_id)
            elif action == pg.K_b:
                valid_flag, action_task, goal_id = self._check_serve_validity(player_id)
            elif action == pg.K_n:
                valid_flag, action_task, goal_id = self._check_drop_validity(player_id)
            else:
                pass

        return valid_flag, action_type, action_task, goal_id

    def _get_action_mapping_info(self, action):
        action_reward_mapping = {
            pg.K_LEFT: [0, -1],
            pg.K_RIGHT: [1, -1],
            pg.K_UP: [2, -1],
            pg.K_DOWN: [3, -1],
            pg.K_COMMA: [4, -2], # DIAG-UP-LEFT
            pg.K_PERIOD: [5, -2], # DIAG-UP-RIGHT
            pg.K_SLASH: [6, -2], # DIAG-DOWN-LEFT
            pg.K_RSHIFT: [7, -2], # DIAG-DOWN-RIGHT
            pg.K_m: [8, -2], # STAY
            pg.K_z: [9, 10],
            pg.K_x: [10, 30],
            pg.K_c: [11, 45],
            pg.K_v: [12, 50],
            pg.K_b: [13, 100],
            pg.K_n: [14, 0], # DROP
        }
        return action_reward_mapping[action][0], action_reward_mapping[action][1]
    
    def _get_ingredient(self, coords):
        ingredient_name = None
        for ingredient in INGREDIENTS_INITIALIZATION:
            if INGREDIENTS_INITIALIZATION[ingredient]['location'][0] == coords:
                ingredient_name = ingredient
        return ingredient_name

    def _get_recipe_ingredient_count(self, ingredient):
        recipe_ingredient_count = None
        for recipe in RECIPES_INFO:
            if ingredient in RECIPES_INFO[recipe]:
                recipe_ingredient_count = RECIPES_INFO[recipe][ingredient]
        
        return recipe_ingredient_count

    def _get_ingredient_dish(self, recipe):
        ingredient = RECIPES_INFO[recipe]['ingredient']
        
        return ingredient

    def _get_goal_id(self, ingredient, action):
        goal_id = None
        for recipe in RECIPES_ACTION_MAPPING:
            if ingredient in RECIPES_ACTION_MAPPING[recipe]:
                goal_id = RECIPES_ACTION_MAPPING[recipe][ingredient][action]
                break
        return goal_id
    
    def _get_general_goal_id(self, recipe, action):
        return RECIPES_ACTION_MAPPING[recipe]['general'][action]

    def _check_pick_validity(self, player_id):
        print('agent@_check_pick_validity')
        pick_validity = False
        player_pos = self._get_pos(player_id)
        all_valid_pick_items = []
        all_valid_pick_items_pos = []
        action_task = []
        goal_id = None

        # Plates, ingredients
        for plate in self.env.world_state['plate']:
            all_valid_pick_items.append(plate)
            all_valid_pick_items_pos.append(plate.location)
        for ingredient in self.env.world_state['ingredients']:
            all_valid_pick_items.append(ingredient)
            all_valid_pick_items_pos.append(ingredient.location)
        for ingredient in INGREDIENTS_STATION:
            for ingredient_coords in INGREDIENTS_STATION[ingredient]:
                all_valid_pick_items_pos.append(ingredient_coords)

        surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
        for surrounding_cell_xy in surrounding_cells_xy:
            surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
            surrounding_cell = tuple(surrounding_cell)
            if surrounding_cell in all_valid_pick_items_pos:
                item = [item for item in all_valid_pick_items if item.location == surrounding_cell]
                if item:
                    item = item[0]
                    if isinstance(item, Plate):
                        action_task.append([
                            'PICK',
                            {
                               'is_new': False,
                                'is_last': True,
                                'pick_type': 'plate',
                                'task_coord': surrounding_cell 
                            },
                            player_pos
                        ])
                        ingredient_name = self._get_ingredient(surrounding_cell)
                        goal_id = self._get_goal_id(ingredient_name, 'PICK')
                    else:
                        action_task.append([
                            'PICK',
                            {
                                'is_new': False,
                                'is_last': False,
                                'pick_type': 'ingredient',
                                'task_coord': surrounding_cell
                            },
                            player_pos
                        ])
                        ingredient_name = self._get_ingredient(surrounding_cell)
                        goal_id = self._get_goal_id(ingredient_name, 'PICK')
                else:
                    # new item from ingredient station
                    action_task.append([
                        'PICK',
                        {
                            'is_new': True,
                            'is_last': True,
                            'pick_type': 'ingredient',
                            'task_coord': surrounding_cell
                        },
                        player_pos
                    ])
                    ingredient_name = self._get_ingredient(surrounding_cell)
                    goal_id = self._get_goal_id(ingredient_name, 'PICK')
        # Have to drop before picking again
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if player_object.holding:
            pick_validity = False

        if action_task and not player_object.holding:
            pick_validity = True
            
        return pick_validity, action_task, goal_id

    def _check_chop_validity(self, player_id):
        chop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = None

        # Have to hold unchopped ingredient before chopping
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            chop_validity = False
        elif not isinstance(player_object.holding, Ingredient):
            chop_validity = False
        elif player_object.holding.state != 'unchopped':
            chop_validity = False
        else:
            # Ensure chopping board is not occupied
            all_valid_chopping_boards_pos = [cb.location for cb in self.env.world_state['chopping_board'] if cb.state != 'taken']

            surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
            for surrounding_cell_xy in surrounding_cells_xy:
                surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
                surrounding_cell = tuple(surrounding_cell)
                if surrounding_cell in all_valid_chopping_boards_pos:
                    action_task.append(
                        ['CHOP', True, surrounding_cell, player_pos]
                    )
                    ingredient_name = player_object.holding.name
                    goal_id = self._get_goal_id(ingredient_name, 'CHOP')
        if action_task:
            chop_validity = True

        return chop_validity, action_task, goal_id

    def _check_cook_validity(self, player_id):
        print(f'human@_check_cook_validity')
        cook_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = None

        # Have to hold chopped ingredient before chopping
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            cook_validity = False
        elif not isinstance(player_object.holding, Ingredient):
            cook_validity = False
        elif player_object.holding.state != 'chopped':
            cook_validity = False
        else:
            # Ensure pot is not full / pot ingredient is same as ingredient in hand
            all_valid_pots = [pot for pot in self.env.world_state['pot']]
            all_valid_pots_pos = [pot.location for pot in self.env.world_state['pot']]

            surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
            for surrounding_cell_xy in surrounding_cells_xy:
                surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
                surrounding_cell = tuple(surrounding_cell)
                if surrounding_cell in all_valid_pots_pos:
                    pot = [pot for pot in all_valid_pots if pot.location == surrounding_cell][0]
                    ingredient_name = player_object.holding.name
                    recipe_ingredient_count = self._get_recipe_ingredient_count(ingredient_name) #assumes no recipe with same ingredient in map
                    # CASE: No ingredient pot yet
                    if pot.is_empty:
                        action_task.append(
                            ['COOK', True, surrounding_cell, player_pos]
                        )
                        goal_id = self._get_goal_id(ingredient_name, 'COOK')
                    # CASE: Already has an ingredient in pot and is same ingredient as hand's ingredient
                    elif pot.ingredient_count[ingredient_name] != recipe_ingredient_count:
                        action_task.append(
                                ['COOK', True, surrounding_cell, player_pos]
                            )
                        goal_id = self._get_goal_id(ingredient_name, 'COOK')
        if action_task:
            cook_validity = True

        return cook_validity, action_task, goal_id
    
    def _check_scoop_validity(self, player_id):
        print('human@_check_scoop_validity')
        scoop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = None

        # Have to hold empty plate before scooping
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            scoop_validity = False
        elif not isinstance(player_object.holding, Plate):
            scoop_validity = False
        elif player_object.holding.state != 'empty':
            scoop_validity = False
        else:
            all_valid_pots = [pot for pot in self.env.world_state['pot']]
            all_valid_pots_pos = [pot.location for pot in self.env.world_state['pot']]

            surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
            for surrounding_cell_xy in surrounding_cells_xy:
                surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
                surrounding_cell = tuple(surrounding_cell)
                if surrounding_cell in all_valid_pots_pos:
                    pot = [pot for pot in all_valid_pots if pot.location == surrounding_cell][0]
                    # Ensure pot is full
                    if pot.dish:
                        action_task.append([
                            'SCOOP',
                            {
                                'is_last': True,
                                'task_coord': surrounding_cell
                            },
                            player_pos
                        ])
                        goal_id = self._get_general_goal_id(pot.dish, 'SCOOP')
        if action_task:
            scoop_validity = True
        return scoop_validity, action_task, goal_id
    
    def _check_serve_validity(self, player_id):
        serve_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = None

        # Have to hold plate with dish before serving
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            serve_validity = False
        elif not isinstance(player_object.holding, Plate):
            serve_validity = False
        elif player_object.holding.state != 'plated':
            serve_validity = False
        else:
            valid_serving_cells = SERVING_STATION

            surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
            for surrounding_cell_xy in surrounding_cells_xy:
                surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
                surrounding_cell = tuple(surrounding_cell)
                if surrounding_cell in valid_serving_cells:
                    action_task.append([
                        'SERVE',
                        {
                            'is_last': True,
                            'task_coord': surrounding_cell
                        },
                        player_pos
                    ])
                    dish_name = player_object.holding.dish.name
                    goal_id = self._get_general_goal_id(dish_name, 'SERVE')
        if action_task:
            serve_validity = True

        return serve_validity, action_task, goal_id

    def _check_drop_validity(self, player_id):
        drop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = -1

        # Have to hold something before dropping
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            drop_validity = False
        else:
            valid_item_cells = self.env.world_state['valid_item_cells']

            surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
            for surrounding_cell_xy in surrounding_cells_xy:
                surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
                surrounding_cell = tuple(surrounding_cell)
                if surrounding_cell in valid_item_cells:
                    if isinstance(player_object.holding, Ingredient):
                        action_task.append([
                            'DROP',
                            {
                                'for_task': 'INGREDIENT'
                            }
                        ])
                    elif isinstance(player_object.holding, Plate):
                        action_task.append([
                            'DROP',
                            {
                                'for_task': 'PLATE'
                            }
                        ])
        if action_task:
            drop_validity = True
        
        return drop_validity, action_task, goal_id

    def update_experiment_results(self, info_df):
        agent_1_info = [(agent.location, agent.last_action) for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        agent_2_info = [(agent.location, agent.last_action) for agent in self.env.world_state['agents'] if agent.id == '2'][0]
        temp_info_df = info_df.append({
            'episode': self.env.episode,
            'player_coords': agent_1_info[0],
            'agent_coords': agent_2_info[0],
            'player_action': agent_1_info[1],
            'agent_action': agent_2_info[1],
            'available_orders': self.env.world_state['order_count'],
            'score': self.env.world_state['score'],
            'timer': TERMINATING_EPISODE - self.env.episode,
            'total_score': self.env.world_state['total_score']
        }, ignore_index=True)
        return temp_info_df

    def rollout(self, best_goals, horizon=50, save_path=None):
        """
        Save deep copy of current world state to be used as previous world state in next timestep.
        Deep copy to be used for inference calculations.

        Deep copy constructs a new compound object and then, recursively, inserts copies into it
        of the objects found in the original.
        This helps prevent any weird occurrences affecting the true world state.
        """
        temp_copy = copy.deepcopy(self.env.world_state)
        # only require historical world_state 1 timestep ago
        temp_copy['historical_world_state'] = {}
        self.env.world_state['historical_world_state'] = temp_copy
        
        reward_mapping = {}
        action_mapping = {}
        for agent in best_goals:
            reward_mapping[agent] = best_goals[agent][1]['rewards']
            action_mapping[agent] = (
                best_goals[agent][0],
                best_goals[agent][1]['steps'][0]
            )
            if type(best_goals[agent][1]['steps'][0]) == int:
                agent.last_action = str(best_goals[agent][1]['steps'][0])
            else:
                agent.last_action = best_goals[agent][1]['steps'][0][0]
        for agent in action_mapping:
            self.env.world_state['historical_actions'][agent.id] = [action_mapping[agent][1]]

        print(action_mapping)
        print('@rollout - Starting step function')
        print([agent.last_action for agent in self.env.world_state['agents']])
        self.env.step(action_mapping, reward_mapping)

        # print(f'Historical World State')
        # print(self.env.world_state['historical_world_state'])

        explicit_chop_rewards = self.env.world_state['explicit_rewards']['chop']
        explicit_cook_rewards = self.env.world_state['explicit_rewards']['cook']
        explicit_serve_rewards = self.env.world_state['explicit_rewards']['serve']
        print(f'Current EXPLICIT chop rewards: {explicit_chop_rewards}')
        print(f'Current EXPLICIT cook rewards: {explicit_cook_rewards}')
        print(f'Current EXPLICIT serve rewards: {explicit_serve_rewards}')

        if not self.is_simulation:
            self.info_df = self.update_experiment_results(self.info_df)
            pg.image.save(self.screen, self.images_folder+f'/episode_{self.env.episode}.png')

    def run_simulation(self, episodes:int=500):
        game_folder = os.path.dirname(__file__)
        simulations_folder = os.path.join(game_folder, 'simulations')
        video_folder = os.path.join(game_folder, 'videos')

        map_folder = os.path.join(*[game_folder, 'videos', MAP])

        helpers.check_dir_exist(simulations_folder)
        helpers.check_dir_exist(video_folder)
        helpers.check_dir_exist(map_folder)

        helpers.clean_dir(simulations_folder)
        
        start_time = datetime.now()
        for episode in range(episodes):
            if episode == 0:
                self.new(
                    self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                    self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
                )

            print(f'================ Episode {episode} best goals ================')
            print(f'\nStart of episode {self.env.episode}')
            best_goals = self.env.find_agents_best_goal()
            goal_space = self.env.world_state['goal_space']
            goal_info = self.env.world_state['goal_space_count']
            print(f'Current goal space: \n{goal_space}\n')
            print(f'Current goal info: \n{goal_info}\n')
            print(f'Best goals')
            print(best_goals)
            print(f'Agent locations')
            print([agent.location for agent in self.env.world_state['agents']])

            self.rollout(best_goals, self.env.episode)
            self.load_data()
            self.update()
            self.draw()

            self.new(
                self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
            )

            print(f'Just completed episode {self.env.episode}')
            goal_space = self.env.world_state['goal_space']
            goal_info = self.env.world_state['goal_space_count']
            print(f'Current goal space: \n{goal_space}\n')
            print(f'Current goal info: \n{goal_info}\n')
            print([agent.location for agent in self.env.world_state['agents']])
            print([agent.holding for agent in self.env.world_state['agents']])
            self.env.update_episode()
            pg.image.save(self.screen, simulations_folder+f'/episode_{self.env.episode}.png')

            if self.env.episode == 0:
                self.results[str(self.env.episode)] = self.env.world_state['total_score']
            if (self.env.episode+1)%50 == 0:
                self.results[str(self.env.episode+1)] = self.env.world_state['total_score']
            if self.env.episode == TERMINATING_EPISODE:
                print('saving results')
                self.save_results()

        print(f'======================= Done with simulation =======================')
        explicit_chop_rewards = self.env.world_state['explicit_rewards']['chop']
        explicit_cook_rewards = self.env.world_state['explicit_rewards']['cook']
        explicit_serve_rewards = self.env.world_state['explicit_rewards']['serve']
        print(f'Current EXPLICIT chop rewards: {explicit_chop_rewards}')
        print(f'Current EXPLICIT cook rewards: {explicit_cook_rewards}')
        print(f'Current EXPLICIT serve rewards: {explicit_serve_rewards}')

        end_time = datetime.now()
        experiment_runtime = (end_time - start_time).seconds
        experiment_runtime_min = experiment_runtime//60
        experiment_runtime_sec = experiment_runtime%60
        print(f'Simulation Experiment took {experiment_runtime_min} mins, {experiment_runtime_sec} secs to run.')

        agent_types = [agent.is_inference_agent for agent in self.env.world_state['agents']]
        # video_name_ext = helpers.get_video_name_ext(agent_types, episodes, MAP)
        video_name_ext = helpers.get_video_name_ext(self.env.world_state['agents'], TERMINATING_EPISODE, MAP)
        helpers.make_video_from_image_dir(
            map_folder,
            simulations_folder,
            video_name_ext
        )
        sys.exit()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

@click.command()
@click.option('--num_ai_agents', default=1, help='Number of AI agents to initialize')
@click.option('--is_simulation', default=False, help='Run Simulation or Human Experiment?')
@click.option('--simulation_episodes', default=500, help='Number of simulations to run')
@click.option('--is_tom', default=False, help='Is agent ToM-based?')
@click.option('--experiment_id', default='1', help='ID of the experiment')
def main(num_ai_agents, is_simulation, simulation_episodes, is_tom, experiment_id):
    # create the game object
    g = Game(num_ai_agents, is_simulation, simulation_episodes, is_tom, experiment_id)
    g.show_start_screen()
    while True:
        g.new(
            g.PLAYERS, g.TABLE_TOPS, g.INGREDIENTS, g.CHOPPING_BOARDS, g.PLATES, g.POTS,
            g.INGREDIENTS_STATION, g.SERVING_STATION, g.RETURN_STATION
        )
        g.run()
        g.show_go_screen()


if __name__ == "__main__":
    main()
