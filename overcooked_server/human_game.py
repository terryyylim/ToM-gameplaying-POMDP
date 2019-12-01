from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import click
import copy
import sys
import random
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
        simulation_episodes: int=500
    ) -> None:
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)

        AI_AGENTS_TO_INITIALIZE = {}
        for idx in range(1, num_ai_agents+1):
            idx = str(idx)
            AI_AGENTS_TO_INITIALIZE[idx] = AI_AGENTS[idx]
        if is_simulation:
            self.env = OvercookedEnv(
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                queue_episodes=QUEUE_EPISODES
            )
            self.load_data()

            self.run_simulation(simulation_episodes)
        else:
            self.env = OvercookedEnv(
                human_agents=HUMAN_AGENTS,
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                queue_episodes=QUEUE_EPISODES
            )
            self.player_1_input = None
            self.player_2_input = None
            self.load_data()
        self.results_filename = 'results/' + self.env.results_filename + '.csv'
        self.results = defaultdict(int)
        self.results_col = []
        for i in range(TERMINATING_EPISODE):
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

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        
        while self.playing:
            if self.env.episode%50 == 0:
                self.results[str(self.env.episode)] = self.env.world_state['total_score']
            if self.env.episode > TERMINATING_EPISODE:
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
        score = font.render(str(current_score), True, GREEN, SCOREBOARD_BG)
        order = font.render(str(current_order), True, GREEN, SCOREBOARD_BG)
        scoreRect = score.get_rect()
        orderRect = order.get_rect()
        scoreRect.center = (
            (SCOREBOARD_SCORE[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2
        )
        orderRect.center = (
            (SCOREBOARD_ORDERS[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2    
        )
        self.screen.blit(score, scoreRect)
        self.screen.blit(order, orderRect)
        pg.display.flip()

    def events(self):    
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYUP:
                print(f'\nStart of episode {self.env.episode}')
                goal_space = self.env.world_state['goal_space']
                goal_info = self.env.world_state['goal_space_count']
                print(f'Current goal space: \n{goal_space}\n')
                print(f'Current goal info: \n{goal_info}\n')
                print([agent.holding for agent in self.env.world_state['agents']])

                if event.key in [
                    pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
                    pg.K_m, pg.K_COMMA, pg.K_PERIOD, pg.K_SLASH, pg.K_RSHIFT,
                    pg.K_z, pg.K_x, pg.K_c, pg.K_v, pg.K_b, pg.K_n
                ]:
                    self.player_2_input = event.key
                
                if event.key in [
                    pg.K_g, pg.K_j, pg.K_y, pg.K_h,
                    pg.K_7, pg.K_q, pg.K_w, pg.K_e, pg.K_r,
                    pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6
                ]:
                    self.player_1_input = event.key
                
                print('inputs...')
                print(self.player_1_input)
                print(self.player_2_input)
                if self.player_1_input and self.player_2_input:
                    cur_player_1_input = self.player_1_input
                    cur_player_2_input = self.player_2_input
                    self.player_1_input = None
                    self.player_2_input = None
                    best_goals = {}
                    for player_id in range(1,3):
                        if player_id == 1:
                            player_input = cur_player_1_input
                        elif player_id == 2:
                            player_input = cur_player_2_input
                        player_action_validity, action_type, action_task, goal_id = self._check_action_validity(player_id, player_input)
                        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]

                        if not player_action_validity:
                            best_goals[player_object] = [-1, {'steps': [8], 'rewards': -2}]
                        else:
                            action_mapping, reward_mapping = self._get_action_mapping_info(player_input)
                            if action_type == 'movement':
                                best_goals[player_object] = [-1, {'steps': [action_mapping], 'rewards': reward_mapping}]
                            else:
                                # its a task_action being taken
                                best_goals[player_object] = [goal_id, {'steps': action_task, 'rewards': reward_mapping}]


                    print('best goals')
                    print(best_goals)
                    # Only works for 2 agents
                    both_same = False
                    both_goals = [info[0] for agent, info in best_goals.items()]
                    if (both_goals[0] == both_goals[1]) and (both_goals[0] == 3):
                        both_same = True
                    if both_same:
                        # chosen agent to not perform
                        chosen_agent_id = random.randint(1,2)
                        chosen_agent = [agent for agent, info in best_goals.items() if int(agent.id) == chosen_agent_id][0]
                        best_goals[chosen_agent] = [-1, {'steps': [8], 'rewards': -2}]

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
            pg.K_m: [0, 0],

            pg.K_g: [0, -1],
            pg.K_j: [0, 1],
            pg.K_y: [-1, 0],
            pg.K_h: [1, 0],
            pg.K_q: [-1, -1],
            pg.K_w: [-1, 1],
            pg.K_e: [1, -1],
            pg.K_r: [1, 1],
            pg.K_7: [0, 0]
        }
        movement_keys = [
            pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
            pg.K_COMMA, pg.K_PERIOD, pg.K_SLASH, pg.K_RSHIFT, pg.K_m,
            pg.K_y, pg.K_g, pg.K_h, pg.K_j,
            pg.K_q, pg.K_w, pg.K_e, pg.K_r, pg.K_7
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
            if (action == pg.K_z) or (action == pg.K_1):
                # Check if there's anything to pick in (UP, DOWN, LEFT, RIGHT)
                valid_flag, action_task, goal_id = self._check_pick_validity(player_id)
            elif (action == pg.K_x) or (action == pg.K_2):
                valid_flag, action_task, goal_id = self._check_chop_validity(player_id)
            elif (action == pg.K_c) or (action == pg.K_3):
                valid_flag, action_task, goal_id = self._check_cook_validity(player_id)
            elif (action == pg.K_v) or (action == pg.K_4):
                valid_flag, action_task, goal_id = self._check_scoop_validity(player_id)
            elif (action == pg.K_b) or (action == pg.K_5):
                valid_flag, action_task, goal_id = self._check_serve_validity(player_id)
            elif (action == pg.K_n) or (action == pg.K_6):
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
            pg.K_z: [9, 10], # PICK
            pg.K_x: [10, 30], # CHOP
            pg.K_c: [11, 45], # COOK
            pg.K_v: [12, 50], # SCOOP
            pg.K_b: [13, 100], # SERVE
            pg.K_n: [14, 0], # DROP

            pg.K_g: [0, -1],
            pg.K_j: [1, -1],
            pg.K_y: [2, -1],
            pg.K_h: [3, -1],
            pg.K_q: [4, -2], # DIAG-UP-LEFT
            pg.K_w: [5, -2], # DIAG-UP-RIGHT
            pg.K_e: [6, -2], # DIAG-DOWN-LEFT
            pg.K_r: [7, -2], # DIAG-DOWN-RIGHT
            pg.K_7: [8, -2], # STAY
            pg.K_1: [9, 10], # PICK
            pg.K_2: [10, 30], # CHOP
            pg.K_3: [11, 45], # COOK
            pg.K_4: [12, 50], # SCOOP
            pg.K_5: [13, 100], # SERVE
            pg.K_6: [14, 0], # DROP
        }
        return action_reward_mapping[action][0], action_reward_mapping[action][1]
    
    def _get_ingredient(self, coords):
        ingredient_name = None
        for ingredient in INGREDIENTS_INITIALIZATION:
            for ingredient_coord in INGREDIENTS_INITIALIZATION[ingredient]['location']:
                if ingredient_coord == coords:
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

        if action_task:
            pick_validity = True

        return pick_validity, action_task, goal_id

    def _check_chop_validity(self, player_id):
        print('agent@_check_chop_validity')
        chop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []
        goal_id = None

        # Have to hold unchopped ingredient before chopping
        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]
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
        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]
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
        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]
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
        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]
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
        player_object = [agent for agent in self.env.world_state['agents'] if int(agent.id) == player_id][0]
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
        
        action_mapping = {}
        for agent in best_goals:
            action_mapping[agent] = (
                best_goals[agent][0],
                best_goals[agent][1]['steps'][0]
            )
        for agent in action_mapping:
            self.env.world_state['historical_actions'][agent.id] = [action_mapping[agent][1]]

        print(action_mapping)
        print('@rollout - Starting step function')
        self.env.step(action_mapping)

        explicit_chop_rewards = self.env.world_state['explicit_rewards']['chop']
        explicit_cook_rewards = self.env.world_state['explicit_rewards']['cook']
        explicit_serve_rewards = self.env.world_state['explicit_rewards']['serve']
        print(f'Current EXPLICIT chop rewards: {explicit_chop_rewards}')
        print(f'Current EXPLICIT cook rewards: {explicit_cook_rewards}')
        print(f'Current EXPLICIT serve rewards: {explicit_serve_rewards}')

    def run_simulation(self, episodes:int=500):
        game_folder = os.path.dirname(__file__)
        simulations_folder = os.path.join(game_folder, 'simulations')
        video_folder = os.path.join(game_folder, 'videos')

        helpers.check_dir_exist(simulations_folder)
        helpers.check_dir_exist(video_folder)

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
        video_name_ext = helpers.get_video_name_ext(agent_types, episodes)
        helpers.make_video_from_image_dir(
            video_folder,
            simulations_folder,
            video_name_ext
        )
        sys.exit()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

@click.command()
@click.option('--num_ai_agents', default=0, help='Number of AI agents to initialize')
@click.option('--is_simulation', default=False, help='Run Simulation or Human Experiment?')
@click.option('--simulation_episodes', default=500, help='Number of simulations to run')
def main(num_ai_agents, is_simulation, simulation_episodes):
    # create the game object
    g = Game(num_ai_agents, is_simulation, simulation_episodes)
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
