from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import numpy as np
from ray.rllib.env import MultiAgentEnv

from astar_search import AStarGraph
from settings import MAP_ACTIONS, WALLS
from overcooked_agent import OvercookedAgent, RLAgent
from human_agent import HumanAgent
from overcooked_item_classes import Plate, Ingredient
from agent_configs import REWARDS

class MapEnv(MultiAgentEnv):
    def __init__(
        self,
    ) -> None:
        """
        Parameters
            ----------
            ascii_map: list of strings
                Specify what the map should look like. Look at constant.py for
                further explanation
            num_agents: int
                Number of agents to have in the system.
            render: bool
                Whether to render the environment
            color_map: dict
                Specifies how to convert between ascii chars and colors
        """
        self.agents = {}
        self.task_id_count = 0
        self.world_state = defaultdict(list)
        self.world_state['task_id_count'] = 0
        self.world_state['historical_actions'] = defaultdict(list)

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn table tops and items"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like pick or chop
        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken
        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    def ascii_to_numpy(self, ascii_list: List[str]):
        """
        Converts a list of strings into a numpy array

        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    # Undone
    def step(self, agent_actions, reward_mapping):
        """Takes in a dict of actions and converts them to a map update
        Parameters
        ----------
        actions: dict {agent: action}
            The agent interprets the action ([int - move action]/[list - explains task action])
            and converts it to a command.
        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym

        QUESTIONS TO CONSIDER:
        1. What to return for observations? since we already have world_state
        """
        print('@map_env - step()')
        print(agent_actions)
        
        orig_pos = {agent: tuple(agent.location) for agent in self.world_state['agents']}
        orig_holding = {agent:agent.holding for agent in self.world_state['agents']}

        agent_executed = self.update_moves(agent_actions)
        curr_pos = {agent: tuple(agent.location) for agent in self.world_state['agents']}

        final_rewards = {}
        # Calculate task rewards
        for agent in reward_mapping:
            # Performed task action
            if agent in agent_executed:
                final_rewards[agent.id] = agent_executed[agent]
            else:
                # Stayed
                if orig_pos[agent] == curr_pos[agent]:
                    final_rewards[agent.id] = REWARDS['STAY']
                # Moved 
                else:
                    movement = list(np.subtract(curr_pos[agent], orig_pos[agent]))
                    if movement in [[0,1], [0,-1], [1,0], [-1,0]]:
                        final_rewards[agent.id] = -1
                    else:
                        final_rewards[agent.id] = -2
        
        # Update map barriers for agent's A* Search
        temp_astar_map = None
        # Get A* Search map
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                temp_astar_map = agent.astar_map
            elif isinstance(agent, HumanAgent):
                temp_astar_map = self.walls
        
        for agent in curr_pos:
            if curr_pos[agent] != orig_pos[agent]:
                self.world_state['valid_cells'].append(orig_pos[agent])
                self.world_state['valid_cells'].remove(curr_pos[agent])

                # Update barriers in map used for A* Search
                temp_astar_map.barriers[0].remove(orig_pos[agent])
                temp_astar_map.barriers[0].append(curr_pos[agent])
        
        for agent in agent_actions:
            action = agent_actions[agent][1]

            if isinstance(action, list):
                # Index 0: action_type; Index 1: action_info
                if action[0] == 'PICK':
                    if action[1]['pick_type'] == 'plate':
                        if agent.holding:
                            # Edge-Case: AgentL about to serve, AgentR trying to pick plate for the same goal that is going to be removed
                            # Continue and not pick instead
                            cur_plate_pos = action[1]['task_coord']

                            if cur_plate_pos not in self.world_state['return_counter']:
                                self.world_state['valid_item_cells'].append(cur_plate_pos)

                    elif action[1]['pick_type'] == 'ingredient':
                        if agent.holding:
                            all_raw_chop_locations = [cb.location for cb in self.world_state['chopping_board']]
                            all_raw_ingredients_locations = [self.world_state['ingredient_'+agent.holding.name][0]]
                            
                            cur_ingredient_pos = action[1]['task_coord']
                            if cur_ingredient_pos not in all_raw_chop_locations and cur_ingredient_pos not in all_raw_ingredients_locations: 
                                self.world_state['valid_item_cells'].append(cur_ingredient_pos)

                if action[0] == 'DROP':
                    if type(orig_holding[agent]) == Plate:
                        cur_plate_pos = orig_holding[agent].location

                        # Edge case: Randomly chosen spot to drop item must be a table-top cell
                        if cur_plate_pos in self.world_state['valid_item_cells']:
                            self.world_state['valid_item_cells'].remove(cur_plate_pos)
                    elif type(orig_holding[agent]) == Ingredient:
                        cur_ingredient_pos = orig_holding[agent].location

                        # Edge case: Randomly chosen spot to drop item must be a table-top cell
                        if cur_ingredient_pos in self.world_state['valid_item_cells']:
                            self.world_state['valid_item_cells'].remove(cur_ingredient_pos)

        # Update A* Search map for all agents
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                agent.astar_map = temp_astar_map
        
        # TO-DO: Add mechanism to store past observations, rewards
        return final_rewards

    # Taking only 1 grid cell movement now (correct?)
    def update_moves(self, actions):
        """
        #Converts agent action tuples into a new map and new agent positions.
        #Also resolves conflicts over multiple agents wanting a cell.
        """
        print('@map_env - update_moves()')
        # fix for np.array weird bug
        print(actions)
        for agent in actions:
            agent.location = tuple(agent.location)

        # Stores non-grid cells movement (if any)
        agent_tasks = {}
        agent_actions = {}

        for agent, task_action in actions.items():
            print(task_action)
            task = task_action[0]
            action = task_action[1]
            if type(action) == int and action <= 8:
                # Case: Just movements
                print('map_env@update_moves - Movement found')
                agent_action = agent.action_map(action)
                agent_actions[agent] = agent_action
            else:
                # Case: Pick/Chop/Cook/Plate/Scoop/Serve actions
                print('map_env@update_moves - Action found')
                agent_tasks[agent] = [task, action]
                agent_action = agent.action_map(8)
                agent_actions[agent] = agent_action
                
        reserved_slots = []
        agent_moves = {}
        for agent, action in agent_actions.items():
            selected_action = MAP_ACTIONS[action]
            
            new_pos = tuple([x + y for x, y in zip(list(agent.location), selected_action)])
            new_pos = agent.return_valid_pos(new_pos)
            agent_moves[agent] = new_pos
            reserved_slots.append((new_pos, agent))

        agent_by_pos = {tuple(agent.location): agent for agent in self.world_state['agents']}

        # list of moves and their corresponding agents
        move_slots = [slot[0] for slot in reserved_slots]
        agent_to_slot = [slot[1] for slot in reserved_slots]

        print('@map_env - Starting moves (if any)')
        print(move_slots)
        print(agent_to_slot)
        print(agent_by_pos)
        print(agent_moves)

        # cut short computation if there are no moves (to be used if we consider rotation)
        if len(agent_to_slot) > 0:
            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            # unique_move is the position the agents want to move to
            # return_count is the number of times the unique_move is wanted
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)
            print('@map_env - Starting fix (if any)')
            print(unique_move)
            print(indices)
            print(return_count)
            print(search_list)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent in all_agents:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            locs = [list(agent.location) for agent in self.world_state['agents']]
                            if move.tolist() in locs: #self.agent_pos
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent = agent_by_pos[tuple(move)]
                                curr_pos = list(agent.location)
                                curr_conflict_pos = list(conflicting_agent.location)
                                conflict_move = agent_moves.get(
                                    conflicting_agent,
                                    curr_conflict_pos)

                                # Condition (1):
                                # a STAY command has been issued
                                if agent == conflicting_agent:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent in moves_copy.keys():
                                    if conflicting_agent.location == curr_pos and \
                                            move.tolist() == conflicting_agent.location.tolist():
                                        print('keep trying to move into each other')
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            agent_idx = [idx for idx, agent_obj in enumerate(self.world_state['agents']) if id(agent_obj) == id(agent_to_slot[index])][0]
                            self.world_state['agents'][agent_idx].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.location):
                                            agent for agent in self.world_state['agents']}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent in all_agents:
                            agent_moves[agent] = agent.location
            
            print('@map_env - Ended fix (if any)')
            print(move_slots)
            print(agent_to_slot)
            print(agent_by_pos)
            print(agent_moves)

            print('@map_env - Starting un-conflicted moves (if any)')
            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.location): agent for agent in self.world_state['agents']}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent, move in moves_copy.items():
                    print('inside agent move')
                    print(agent)
                    print(move)
                    print([agent.location for agent in self.world_state['agents']])
                    if agent in del_keys:
                        continue
                    if list(move) in [list(agent.location) for agent in self.world_state['agents']]:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent = agent_by_pos[tuple(move)]
                        curr_pos = list(agent.location)
                        curr_conflict_pos = list(conflicting_agent.location)
                        conflict_move = agent_moves.get(conflicting_agent, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent == conflicting_agent:
                            del agent_moves[agent]
                            del_keys.append(agent)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent]
                            del_keys.append(agent)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent in moves_copy.keys():
                            if agent_moves[conflicting_agent] == curr_pos and \
                                    move == conflicting_agent.location.tolist():
                                print('keep trying to move into each other 2')
                                del agent_moves[conflicting_agent]
                                del agent_moves[agent]
                                del_keys.append(agent)
                                del_keys.append(conflicting_agent)
                    # this move is unconflicted so go ahead and move
                    else:
                        agent_idx = [idx for idx, agent_obj in enumerate(self.world_state['agents']) if id(agent_obj) == id(agent)][0]
                        self.world_state['agents'][agent_idx].update_agent_pos(move)
                        del agent_moves[agent]
                        del_keys.append(agent)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent, move in agent_moves.items():
                        self.world_state['agents'][agent].update_agent_pos(move)
                    break

        # All possible action handlers
        print('@map_env - Executing Task Handlers')
        agent_executed = {}
        for agent in agent_tasks:
            print('conducting tasks now')
            task_id = agent_tasks[agent][0]
            task_action = agent_tasks[agent][1]
            print(task_action)
            agent_rewards = REWARDS[task_action[0]]
            # do we still need the second check?
            if task_action[0] == 'PICK' and task_action[2] == agent.location:
                print('@map_env - Executing Pick Action')
                # agent.pick(task_id, task_action[1], task_action[2], task_action[3])
                agent.pick(task_id, task_action[1])
                agent_executed[agent] = agent_rewards
            elif task_action[0] == 'CHOP' and task_action[3] == agent.location:
                print('@map_env - Executing Chop Action')
                agent.chop(task_id, task_action[1], task_action[2])
                agent_executed[agent] = agent_rewards
            elif task_action[0] == 'COOK' and task_action[3] == agent.location:
                print('@map_env - Executing Cook Action')
                agent.cook(task_id, task_action[1], task_action[2])
                agent_executed[agent] = agent_rewards
            elif task_action[0] == 'SCOOP' and task_action[2] == agent.location:
                print('@map_env - Executing Scoop Action')
                agent.scoop(task_id, task_action[1])
                agent_executed[agent] = agent_rewards
            elif task_action[0] == 'SERVE' and task_action[2] == agent.location:
                print('@map_env - Executing Serve Action')
                agent.serve(task_id, task_action[1])
                agent_executed[agent] = agent_rewards
            elif task_action[0] == 'DROP':
                print('@map_env - Executing Drop Action')
                agent.drop(task_id)
                agent_executed[agent] = agent_rewards
        
        return agent_executed

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
            
    def _get_pos(self, player_id):
        if player_id == 1:
            return self.player_1.y, self.player_1.x
        elif player_id == 2:
            return self.player_2.y, self.player_2.x    

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