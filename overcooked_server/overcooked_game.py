from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import copy
import sys
import pygame as pg

from map_env import MapEnv
from overcooked_env import OvercookedEnv
from map_configs import RECIPES
from overcooked_game_settings import *
from sprites import *

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)

        self.env = OvercookedEnv(human_agents=HUMAN_AGENTS, ai_agents=AI_AGENTS)
        self.load_data()

    def _deep_copy(self, obj):
        return copy.deepcopy(obj)

    def load_data(self):
        print(f'Loading data')
        print(self.env.world_state)
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
            self.PLAYERS[agent.agent_id] = {
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
                for k,v in pot.ingredient_count.items():
                    pot_ingredient, pot_ingredient_count = k, v 
            else:
                pot_ingredient, pot_ingredient_count = None, 0
            self.POTS[pot.pot_id] = {
                'ingredient': pot_ingredient,
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
        self.EXTINGUISHER = EXTINGUISHER
        self.TRASH_BIN = TRASH_BIN
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
        extinguisher: Tuple[int,int],
        trash_bin: Tuple[int,int],
        walls: List[Tuple[int,int]]=WALLS
    ) -> None:
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.table_tops = pg.sprite.Group()
        self.ingredients = pg.sprite.Group()
        self.chopping_boards = pg.sprite.Group()
        self.plates = pg.sprite.Group()
        self.pot_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.serving_stations = pg.sprite.Group()
        self.return_station = pg.sprite.Group()
        self.extinguisher = pg.sprite.Group()
        self.trash_bin = pg.sprite.Group()
        self.walls = walls
        self.player_count = len(players)

        for idx in range(1, self.player_count+1):
            idx = str(idx)
            if idx == '1':
                self.player_1 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '2':
                self.player_2 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
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
            pot_ingredient = val['ingredient']
            pot_ingredient_count = val['ingredient_count']
            pot_coord = val['coords']
            PotStation(self, pot_ingredient, pot_ingredient_count, pot_coord[1], pot_coord[0])
        for key, val in ingredient_stations.items():
            ingredient = key
            chopping_board_coord = val
            IngredientStation(self, ingredient, chopping_board_coord[1], chopping_board_coord[0])
        for serving_station_coord in serving_stations:
            ServingStation(self, serving_station_coord[1], serving_station_coord[0])
            
        ReturnStation(self, return_station['state'], return_station['coords'][1], return_station['coords'][0])
        ExtinguisherStation(self, extinguisher[1], extinguisher[0])
        TrashBin(self, trash_bin[1], trash_bin[0])


    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        
        while self.playing:
            if self.env.episode > 500:
                self.playing = False
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
        pg.display.flip()

    def events(self):    
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                print(self.walls)

                player_action_validity, action_type, action_task = self._check_action_validity(1, event.key)
                print(self.env.world_state['agents'])
                player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
                print('player obj')
                print(player_object)
                best_goals = self.env.find_agents_best_goal()

                if not player_action_validity:
                    best_goals[player_object] = [-1, {'steps': [8], 'rewards': -2}]
                else:
                    action_mapping, reward_mapping = self._get_action_mapping_info(event.key)
                    if action_type == 'movement':
                        best_goals[player_object] = [-1, {'steps': [action_mapping], 'rewards': reward_mapping}]
                    else:
                        # its a task_action being taken
                        # TODO: find best goal that maps to action
                        goal_id = player_object._find_suitable_goal(action_mapping, action_task)
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
                    self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION, self.EXTINGUISHER, self.TRASH_BIN
                )

                if event.key == pg.K_ESCAPE:
                    self.quit()
                # if event.key == pg.K_LEFT:
                #     print(self.player_1.x)
                #     print(self.player_1.y)
                #     self.player_1.move(dx=-1)
                # if event.key == pg.K_RIGHT:
                #     self.player_1.move(dx=1)
                # if event.key == pg.K_UP:
                #     self.player_1.move(dy=-1)
                # if event.key == pg.K_DOWN:
                #     self.player_1.move(dy=1)
                print(f'Just completed episode {self.env.episode}')
                print([agent.location for agent in self.env.world_state['agents']])
                self.env.update_episode()
                pg.image.save(self.screen, f'episodes/episode_{self.env.episode}.png')

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
            pg.K_q: [-1, -1],
            pg.K_w: [-1, 1],
            pg.K_e: [1, -1],
            pg.K_r: [1, 1],
            pg.K_s: [0, 0]
        }
        movement_keys = [
            pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
            pg.K_q, pg.K_w, pg.K_e, pg.K_r, pg.K_s
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
        else:
            # Do actions
            action_type = 'action'
            if action == pg.K_z:
                # Check if there's anything to pick in (UP, DOWN, LEFT, RIGHT)
                valid_flag, action_task = self._check_pick_validity(player_id)
            elif action == pg.K_x:
                valid_flag, action_task = self._check_chop_validity(player_id)
            elif action == pg.K_c:
                valid_flag, action_task = self._check_cook_validity(player_id)
            elif action == pg.K_v:
                valid_flag, action_task = self._check_scoop_validity(player_id)
            elif action == pg.K_b:
                valid_flag, action_task = self._check_serve_validity(player_id)
            elif action == pg.K_a:
                valid_flag, action_task = self._check_drop_validity(player_id)
            else:
                pass

        return valid_flag, action_type, action_task

    def _get_action_mapping_info(self, action):
        action_reward_mapping = {
            pg.K_LEFT: [0, -1],
            pg.K_RIGHT: [1, -1],
            pg.K_UP: [2, -1],
            pg.K_DOWN: [3, -1],
            pg.K_q: [4, -2],
            pg.K_w: [5, -2],
            pg.K_e: [6, -2],
            pg.K_r: [7, -2],
            pg.K_s: [8, -2],
            pg.K_z: [9, 10],
            pg.K_x: [10, 30],
            pg.K_c: [11, 45],
            pg.K_v: [12, 50],
            pg.K_b: [13, 100],
            pg.K_a: [14, 0],
        }
        return action_reward_mapping[action][0], action_reward_mapping[action][1]
    
    def _check_pick_validity(self, player_id):
        pick_validity = False
        player_pos = self._get_pos(player_id)
        all_valid_pick_items = []
        all_valid_pick_items_pos = []
        action_task = []

        # Plates, ingredients
        for plate in self.env.world_state['plate']:
            all_valid_pick_items.append(plate)
            all_valid_pick_items_pos.append(plate.location)
        for ingredient in self.env.world_state['ingredients']:
            all_valid_pick_items.append(ingredient)
            all_valid_pick_items_pos.append(ingredient.location)
        for ingredient in INGREDIENTS_STATION:
            all_valid_pick_items_pos.append(INGREDIENTS_STATION[ingredient])

        print('now picking action')
        print(player_pos)
        surrounding_cells_xy = [[-1,0], [0,1], [1,0], [0,-1]]
        for surrounding_cell_xy in surrounding_cells_xy:
            surrounding_cell = [sum(x) for x in zip(list(player_pos), surrounding_cell_xy)]
            surrounding_cell = tuple(surrounding_cell)
            if surrounding_cell in all_valid_pick_items_pos:
                item = [item for item in all_valid_pick_items if item.location == surrounding_cell]
                print('picking action now')
                print(surrounding_cell)
                print(item)
                if item:
                    item = item[0]
                    print('item 0')
                    print(item)
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
                    else:
                        print('not picking plate')
                        # if item.state == 'unchopped':
                        #     action_task.append([
                        #         'PICK',
                        #         {
                        #             'is_new': False,
                        #             'is_last': True,
                        #             'pick_type': 'ingredient',
                        #             'task_coord': surrounding_cell
                        #         },
                        #         player_pos
                        #     ])
                        # else:
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
                else:
                    print('pick from new station')
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
        # Have to drop before picking again
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if player_object.holding:
            pick_validity = False

        if action_task:
            pick_validity = True
            
        return pick_validity, action_task

    def _check_chop_validity(self, player_id):
        chop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []

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
        if action_task:
            chop_validity = True

        return chop_validity, action_task

    def _check_cook_validity(self, player_id):
        cook_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []

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
                    ingredient_in_hand = player_object.holding.name
                    # Ugly hack
                    if ingredient_in_hand == 'onion':
                        dish = 'onion_soup'
                    # No ingredient pot yet
                    if len(pot.ingredient_count) == 0:
                        action_task.append(
                            ['COOK', True, surrounding_cell, player_pos]
                        )
                    # Already has an ingredient in pot    
                    elif len(pot.ingredient_count) != 0:
                        if pot.ingredient_count[ingredient_in_hand] != RECIPES_INGREDIENTS_COUNT[dish]:
                            action_task.append(
                                ['COOK', True, surrounding_cell, player_pos]
                            )
                    # NEG CASE: if pot ingredient is different as ingredient in hand
        if action_task:
            cook_validity = True

        return cook_validity, action_task
    
    def _check_scoop_validity(self, player_id):
        scoop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []

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
                    # Ugly hack 1: assume only 1 pot such that self.world_state['cooked_dish_count'][dish] += 1 always = 1 at most
                    # Ugly hack 2: assume only 1 dish
                    dish = RECIPES[0]
                    if self.env.world_state['cooked_dish_count'][dish] == 1:
                        action_task.append([
                            'SCOOP',
                            {
                                'is_last': True,
                                'task_coord': surrounding_cell
                            },
                            player_pos
                        ])
        if action_task:
            scoop_validity = True
        return scoop_validity, action_task
    
    def _check_serve_validity(self, player_id):
        serve_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []

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
        if action_task:
            serve_validity = True

        return serve_validity, action_task

    def _check_drop_validity(self, player_id):
        drop_validity = False
        player_pos = self._get_pos(player_id)
        action_task = []

        # Have to hold something before dropping
        player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        if not player_object.holding:
            drop_validity = False
        else:
            valid_item_cells = self.env.world_state['valid_item_cells']
            print('drop valid item cells')
            print(valid_item_cells)

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
        
        return drop_validity, action_task

    def rollout(self, best_goals, horizon=50, save_path=None):
        print('rollout@rollout')
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
            self.env.world_state['historical_actions'][agent.id].append(
                action_mapping[agent][1]
            )

        print(action_mapping)
        print('@rollout - Starting step function')
        self.env.step(action_mapping)
        # print(f'@rollout - Currently at horizon - {horizon}')
        # self.env.render('./ipomdp/images/timestep'+str(horizon))

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

def main():
    # create the game object
    g = Game()
    g.show_start_screen()
    while True:
        g.new(
            g.PLAYERS, g.TABLE_TOPS, g.INGREDIENTS, g.CHOPPING_BOARDS, g.PLATES, g.POTS,
            g.INGREDIENTS_STATION, g.SERVING_STATION, g.RETURN_STATION, g.EXTINGUISHER, g.TRASH_BIN
        )
        g.run()
        g.show_go_screen()


if __name__ == "__main__":
    main()