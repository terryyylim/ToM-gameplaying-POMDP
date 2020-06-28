from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import copy
import numpy as np
import random

from agent_configs import ACTIONS, REWARDS
from astar_search import AStarGraph
from overcooked_item_classes import Ingredient, Plate, Dish
from settings import RECIPES_INFO, RECIPE_ACTION_NAME, INGREDIENT_ACTION_NAME, \
    MAP_ACTIONS, RECIPES_ACTION_MAPPING, FLATTENED_RECIPES_ACTION_MAPPING

class OvercookedAgent():
    def __init__(
        self,
        agent_id,
        location,
        barriers,
        is_inference_agent=False,
        is_assigned=False,
        can_update=True,
        goals=None,
        holding=None,
        actions=ACTIONS,
        rewards=REWARDS
    ) -> None:
        self.world_state = {}
        self.id = agent_id
        self.location = location
        self.is_inference_agent = is_inference_agent
        self.is_assigned = is_assigned
        self.can_update = can_update
        self.goals = goals
        self.holding = holding
        self.actions = actions
        self.rewards = rewards
        self.get_astar_map(barriers)

    def get_astar_map(self, barriers: List[List[Tuple[int,int]]]) -> None:
        self.astar_map = AStarGraph(barriers)

    def calc_travel_cost(self, items: List[str], items_coords: List[List[Tuple[int,int]]]):
        # get valid cells for each goal
        item_valid_cell_states = defaultdict(list)
        for item_idx in range(len(items)):
            item_valid_cell_states[items[item_idx]] = self.find_valid_cell(items_coords[item_idx])

        travel_costs = defaultdict(tuple)
        for item_idx in range(len(items)):
            cur_item_instances = items_coords[item_idx]
            for cur_item_instance in cur_item_instances:
                try:
                    valid_cells = item_valid_cell_states[items[item_idx]][cur_item_instance]
                    # Edge case: Multiple plates/pots and already at an empty one
                    # if len(valid_cells) == 0:
                    #     # No need to move
                    #     travel_costs[items[item_idx]] = ([], 0, cur_item_instance)
                    for valid_cell in valid_cells:
                        temp_item_instance = self.AStarSearch(valid_cell)
                        if not travel_costs[items[item_idx]]:
                            travel_costs[items[item_idx]] = (temp_item_instance[0], temp_item_instance[1], cur_item_instance)
                        else:
                            if travel_costs[items[item_idx]][1] == temp_item_instance[1]:
                                # Randomizing item instance to go towards should prevents being stuck
                                random_int = random.randint(0,1)
                                temp_item_instance = list(temp_item_instance)
                                temp_item_instance.append(cur_item_instance)
                                temp_item_instance = tuple(temp_item_instance)

                                random_selection = [travel_costs[items[item_idx]], temp_item_instance][random_int]
                                travel_costs[items[item_idx]] = random_selection
                            # Only replace if existing travel cost is greater (ensure only 1 path is returned given same cost)
                            elif travel_costs[items[item_idx]][1] > temp_item_instance[1]:
                                travel_costs[items[item_idx]] = (temp_item_instance[0], temp_item_instance[1], cur_item_instance)
                            continue
                except KeyError:
                    raise KeyError('No valid path to get to item!')
        return travel_costs

    def AStarSearch(self, dest_coords: Tuple[int,int]):
        """
        A* Path-finding algorithm
        Type: Heuristic-Search - Informed Search Algorithm

        F: Estimated movement cost of start to end going via this position
        G: Actual movement cost to each position from the start position
        H: heuristic - estimated distance from the current node to the end node

        It is important for heuristic to always be an underestimation of the total path, as an overestimation
        will lead to A* searching through nodes that may not be the 'best' in terms of f value.

        TO-DO: Invert calculations
        """
        start = tuple(self.location)
        end = dest_coords
        G = {}
        F = {}
    
        # Initialize starting values
        G[start] = 0 
        F[start] = self.astar_map.heuristic(start, end)
    
        closedVertices = set()
        openVertices = set([start])
        cameFrom = {}
    
        while len(openVertices) > 0:
            # Get the vertex in the open list with the lowest F score
            current = None
            currentFscore = None
            for pos in openVertices:
                if current is None or F[pos] < currentFscore:
                    currentFscore = F[pos]
                    current = pos
    
            # Check if we have reached the goal
            if current == end:
                # Retrace our route backward
                path = [current]
                while current in cameFrom:
                    current = cameFrom[current]
                    path.append(current)
                path.reverse()
                return path, F[end] # Done!
    
            # Mark the current vertex as closed
            openVertices.remove(current)
            closedVertices.add(current)
    
            # Update scores for vertices near the current position
            for neighbour in self.astar_map.get_vertex_neighbours(current):
                if neighbour in closedVertices: 
                    continue # We have already processed this node exhaustively
                candidateG = G[current] + self.astar_map.move_cost(current, neighbour)

                if neighbour not in openVertices:
                    openVertices.add(neighbour) # Discovered a new vertex
                elif candidateG >= G[neighbour]:
                    continue # This G score is worse than previously found
    
                #Adopt this G score
                cameFrom[neighbour] = current
                G[neighbour] = candidateG
                H = self.astar_map.heuristic(neighbour, end)
                F[neighbour] = G[neighbour] + H
    
        raise RuntimeError("A* failed to find a solution")

    def find_valid_cell(self, item_coords: List[Tuple[int,int]]) -> Tuple[int,int]:
        """
        Items can only be accessible from Up-Down-Left-Right of item cell.
        Get all cells agent can step on to access item.

        Returns
        -------
        all_valid_cells: Dict[str,List[Tuple[int,int]]]
        """
        all_valid_cells = defaultdict(list)
        # item_instance is Tuple[int,int]
        # removing agent.location from valid_cells screws this check up

        agent_locs = [tuple(agent.location) for agent in self.world_state['agents']]
        for item_instance in item_coords:
            # Edge Case: Convert elif to if statements to consider item pick-up points with 2 valid end_coords
            if (item_instance[0], item_instance[1]+1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]+1))
            if (item_instance[0], item_instance[1]-1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]-1))
            if (item_instance[0]-1, item_instance[1]) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0]-1, item_instance[1]))
            if (item_instance[0]+1, item_instance[1]) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0]+1, item_instance[1]))

            if (item_instance[0], item_instance[1]+1) not in self.world_state['valid_cells'] \
                and (item_instance[0], item_instance[1]+1) in agent_locs:
                    all_valid_cells[item_instance].append((item_instance[0], item_instance[1]+1))
            if (item_instance[0], item_instance[1]-1) not in self.world_state['valid_cells'] \
                and (item_instance[0], item_instance[1]-1) in agent_locs:
                    all_valid_cells[item_instance].append((item_instance[0], item_instance[1]-1))
            if (item_instance[0]-1, item_instance[1]) not in self.world_state['valid_cells'] \
                and (item_instance[0]-1, item_instance[1]) in agent_locs:
                    all_valid_cells[item_instance].append((item_instance[0]-1, item_instance[1]))
            if (item_instance[0]+1, item_instance[1]) not in self.world_state['valid_cells'] \
                and (item_instance[0]+1, item_instance[1]) in agent_locs:
                    all_valid_cells[item_instance].append((item_instance[0]+1, item_instance[1]))

        return all_valid_cells

    def map_path_actions(self, path: List[Tuple[int,int]]):
        path_actions_mapping = []
        for step in range(len(path)-1):
            cur_pos = path[step]
            next_pos = path[step+1]
            difference = tuple(np.subtract(next_pos, cur_pos))
            if difference == (0, -1):
                path_actions_mapping.append(0)
            elif difference == (0, 1):
                path_actions_mapping.append(1)
            elif difference == (-1, 0):
                path_actions_mapping.append(2)
            elif difference == (1, 0):
                path_actions_mapping.append(3)
            elif difference == (-1, -1):
                path_actions_mapping.append(4)
            elif difference == (-1, 1):
                path_actions_mapping.append(5)
            elif difference == (1, -1):
                path_actions_mapping.append(6)
            elif difference == (1, 1):
                path_actions_mapping.append(7)
        return path_actions_mapping

    def return_valid_pos(self, new_pos):
        """
        Checks that the next pos is legal, if not return current pos
        """
        if new_pos in self.world_state['valid_cells']:
            return new_pos

        # you can't walk through walls nor another agent
        
        return self.location

    def update_agent_pos(self, new_coords: List[int]) -> None:
        self.location = new_coords

    def action_map(self, action_number: int) -> str:
        return ACTIONS[action_number]

    def find_best_goal(self, observer_filtered_goals=[]):
        agent_goal_costs = defaultdict(dict)

        final_goal_list = [goal for goal in self.world_state['goal_space'] if goal not in observer_filtered_goals]
        final_goal_list = [goal for goal in final_goal_list if self.world_state['goal_space_count'][goal] > 0]
        
        # For all goals in final_goal_list, can access info through first element
        # PICK GOALS - onion, tomato
        for goal in final_goal_list:
            total_rewards = 0
            if goal in FLATTENED_RECIPES_ACTION_MAPPING['PICK']:
                print(f'@agent - Entered PICK logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]
                print(task_info)

                # Case: Not holding ingredient and it does not exist in map
                if not self.holding:
                    path_cost = self.calc_travel_cost(['ingredient_'+task_info['ingredient']], [self.world_state['ingredient_'+task_info['ingredient']]])
                    task_coord = self.world_state['ingredient_'+task_info['ingredient']][0]
                    end_coord = self.location # no need to move anymore
                    if path_cost:
                        end_coord = path_cost['ingredient_'+task_info['ingredient']][0][-1]
                        path_actions += self.map_path_actions(path_cost['ingredient_'+task_info['ingredient']][0])

                    path_actions.append([
                        'PICK',
                        {
                            'is_new': True,
                            'is_last': True,
                            'pick_type': 'ingredient',
                            'task_coord': task_coord,
                            'for_task': 'PICK'
                        },
                        end_coord
                    ])
                # Case: Holding object and has to be dropped first
                else:
                    holding_type = None
                    if isinstance(self.holding, Plate):
                        holding_type = 'PLATE'
                    elif isinstance(self.holding, Ingredient):
                        holding_type = 'INGREDIENT'
                    path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                    task_coord = path_cost['valid_item_cells'][2]
                    end_coord = path_cost['valid_item_cells'][0][-1]
                    valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                    path_actions += valid_drop_path_actions
                    path_actions.append([
                        'DROP',
                        {
                            'for_task': 'PICK'
                        }
                    ])
            # CHOP GOALS - onion, tomato
            if goal in FLATTENED_RECIPES_ACTION_MAPPING['CHOP']:
                print(f'@agent - Entered CHOP logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]

                wanted_ingredient = [
                    ingredient.location for ingredient in self.world_state['ingredients'] if \
                        (ingredient.name == task_info['ingredient'] and ingredient.state == task_info['state'])]
                if self.holding:
                    if isinstance(self.holding, Plate):
                        holding_type = 'PLATE'
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'CHOP'
                            }
                        ])
                    elif isinstance(self.holding, Ingredient) and self.holding.name != task_info['ingredient']:
                        holding_type = 'INGREDIENT'
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'CHOP'
                            }
                        ])
                    elif isinstance(self.holding, Ingredient) and self.holding.name == task_info['ingredient']:
                        if self.holding.state != task_info['state']:
                            holding_type = 'INGREDIENT'
                            path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                            task_coord = path_cost['valid_item_cells'][2]
                            end_coord = path_cost['valid_item_cells'][0][-1]
                            valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                            path_actions += valid_drop_path_actions
                            path_actions.append([
                                'DROP',
                                {
                                    'for_task': 'CHOP'
                                }
                            ])
                        elif self.holding.state == task_info['state']:
                            try:
                                chopping_board_cells = [chopping_board.location for chopping_board in self.world_state['chopping_board'] if chopping_board.state == 'empty']
                                chopping_path_cost = self.calc_travel_cost(['chopping_board'], [chopping_board_cells])
                                task_coord = [board.location for board in self.world_state['chopping_board'] if board.state == 'empty'][0]
                                end_coord = self.location # no need to move anymore

                                if chopping_path_cost:
                                    task_coord = chopping_path_cost['chopping_board'][2]
                                    end_coord = chopping_path_cost['chopping_board'][0][-1]
                                    chopping_path_actions = self.map_path_actions(chopping_path_cost['chopping_board'][0])
                                    path_actions += chopping_path_actions
                                path_actions.append(['CHOP', True, task_coord, end_coord])
                            except IndexError:
                                # No empty chopping board
                                continue
                else:
                    # Case: Not holding ingredient but it exist in map
                    if wanted_ingredient:
                        path_cost = self.calc_travel_cost(['ingredient_'+task_info['ingredient']], [wanted_ingredient])
                        task_coord = path_cost['ingredient_'+task_info['ingredient']][2]
                        end_coord = path_cost['ingredient_'+task_info['ingredient']][0][-1]
                        path_actions += self.map_path_actions(path_cost['ingredient_'+task_info['ingredient']][0])

                        path_actions.append([
                            'PICK',
                            {
                                'is_new': False,
                                'is_last': False,
                                'pick_type': 'ingredient',
                                'task_coord': task_coord,
                                'for_task': 'CHOP'
                            },
                            end_coord
                        ])
                    else:
                        # Should we make default behaviour to pick up new ingredient from crate?
                        continue

            # COOK GOALS - onion, tomato
            if goal in FLATTENED_RECIPES_ACTION_MAPPING['COOK']:
                """ CONDITION TO FULFIL: Holding chopped onion """
                print(f'@agent - Entered COOK logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]

                wanted_ingredient = [
                    ingredient.location for ingredient in self.world_state['ingredients'] if \
                        (ingredient.name == task_info['ingredient'] and ingredient.state == task_info['state'])]
                if self.holding:
                    if isinstance(self.holding, Plate):
                        holding_type = 'PLATE'
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': holding_type
                            }
                        ])
                    elif isinstance(self.holding, Ingredient) and self.holding.name != task_info['ingredient']:
                        holding_type = 'INGREDIENT'
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'COOK'
                            }
                        ])
                    elif isinstance(self.holding, Ingredient) and self.holding.name == task_info['ingredient']:
                        if self.holding.state != task_info['state']:
                            holding_type = 'INGREDIENT'
                            path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                            task_coord = path_cost['valid_item_cells'][2]
                            end_coord = path_cost['valid_item_cells'][0][-1]
                            valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                            path_actions += valid_drop_path_actions
                            path_actions.append([
                                'DROP',
                                {
                                    'for_task': 'COOK'
                                }
                            ])
                        elif self.holding.state == task_info['state']:
                            recipe_ingredient_count = self.get_recipe_ingredient_count(task_info['recipe'], task_info['ingredient'])
                            recipe_total_ingredients_count = self.get_recipe_total_ingredient_count(task_info['recipe'])
                            # Fill pot with ingredient before considering empty pots
                            # pot_cells = [pot.location for pot in self.world_state['pot'] if pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count]
                            pot_cells = []
                            for pot in self.world_state['pot']:
                                # complex recipe
                                curr_pot_ingredient_count = sum(pot.ingredient_count.values())
                                if curr_pot_ingredient_count > 1:
                                    if (pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count) and (pot.ingredient_count[task_info['ingredient']] == 0) and (recipe_total_ingredients_count > 0):
                                        pot_cells.append(pot.location)
                                    elif (pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count) and (pot.ingredient_count[task_info['ingredient']] == 0) and (recipe_total_ingredients_count == 0):
                                        pot_cells.append(pot.location)
                                    elif (pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count) and (curr_pot_ingredient_count > 0):
                                        pot_cells.append(pot.location)
                                else:
                                    # > 0 to prioritize filling up pot already with ingredients
                                    if (pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count) and (curr_pot_ingredient_count > 0):
                                        pot_cells.append(pot.location)
                            # pot_cells = [pot.location for pot in self.world_state['pot'] \
                            #     if (pot.ingredient_count[task_info['ingredient']] < recipe_ingredient_count) \
                            #         and (pot.ingredient_count[task_info['ingredient']] > 0)]
                            if not pot_cells:
                                pot_cells = [pot.location for pot in self.world_state['pot'] if pot.is_empty]
                            if pot_cells:
                                cooking_path_cost = self.calc_travel_cost(['pot'], [pot_cells])

                                # TO-FIX (if > 1, randomly choose 1)
                                task_coord = cooking_path_cost['pot'][2]
                                end_coord = self.location # no need to move anymore

                                if cooking_path_cost:
                                    end_coord = cooking_path_cost['pot'][0][-1]
                                    cooking_path_actions = self.map_path_actions(cooking_path_cost['pot'][0])
                                    path_actions += cooking_path_actions
                                path_actions.append(['COOK', True, task_coord, end_coord])
                            else:
                                # If still no available pots
                                # TO-FIX: Causes inference agent to pick and drop continuously because no pot to cook at
                                print(f'agent@cook - Trying to cook but no available pot')
                                continue
                else:
                    # Case: Not holding ingredient but it exist in map
                    if wanted_ingredient:
                        path_cost = self.calc_travel_cost(['ingredient_'+task_info['ingredient']], [wanted_ingredient])
                        task_coord = path_cost['ingredient_'+task_info['ingredient']][2]
                        end_coord = path_cost['ingredient_'+task_info['ingredient']][0][-1]
                        path_actions += self.map_path_actions(path_cost['ingredient_'+task_info['ingredient']][0])

                        path_actions.append([
                            'PICK',
                            {
                                'is_new': False,
                                'is_last': False,
                                'pick_type': 'ingredient',
                                'task_coord': task_coord,
                                'for_task': 'COOK'
                            },
                            end_coord
                        ])
                    else:
                        # Should we make default behaviour to pick up new ingredient from crate?
                        print(f'SHOULD WE DEFAULT BEHAVIOUR?')
                        continue
            
            # SCOOP GOALS - onion, tomato
            if goal in FLATTENED_RECIPES_ACTION_MAPPING['SCOOP']:
                """ CONDITION TO FULFIL: Holding empty plate """
                print(f'@agent - Entered SCOOP logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]

                if self.holding:
                    if not isinstance(self.holding, Plate):
                        holding_type = 'INGREDIENT' # can only be ingredient for now
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'SCOOP'
                            }
                        ])
                    elif isinstance(self.holding, Plate) and self.holding.state != task_info['state']:
                        holding_type = 'PLATE'
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'SCOOP'
                            }
                        ])
                    elif isinstance(self.holding, Plate) and self.holding.state == task_info['state']:
                        dish = self.get_recipe_name(goal)
                        try:
                            pot_cells = [pot.location for pot in self.world_state['pot'] if pot.dish == dish]
                            collection_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                            task_coord = collection_path_cost['pot'][2]
                            end_coord = collection_path_cost['pot'][0][-1]
                            collection_path_actions = self.map_path_actions(collection_path_cost['pot'][0])
                            path_actions += collection_path_actions

                            path_actions.append([
                                'SCOOP',
                                {
                                    'is_last': True,
                                    'task_coord': task_coord
                                },
                                end_coord
                            ])
                        # If pot with dish does not exists
                        except IndexError:
                            continue
                else:
                    try:
                        # If plate exists in the map
                        plate_board_cells = [plate.location for plate in self.world_state['plate']]
                        plate_path_cost = self.calc_travel_cost(['plate'], [plate_board_cells])
                        task_coord = plate_path_cost['plate'][2]
                        end_coord = plate_path_cost['plate'][0][-1]
                        path_actions += self.map_path_actions(plate_path_cost['plate'][0])

                        path_actions.append([
                            'PICK',
                            {
                                'is_new': False,
                                'is_last': False,
                                'pick_type': 'plate',
                                'task_coord': task_coord,
                                'for_task': 'SCOOP'
                            },
                            end_coord
                        ])
                    except IndexError:
                        print('@base_agent - scoop IndexError')
                        continue
            
            # SERVE GOALS - onion, tomato
            if goal in FLATTENED_RECIPES_ACTION_MAPPING['SERVE']:
                """ CONDITION TO FULFIL: Holding plated plate """
                print(f'@agent - Entered SERVE logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]

                if self.holding:
                    if not isinstance(self.holding, Plate):
                        holding_type = 'INGREDIENT' # can only be ingredient for now
                        path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                        task_coord = path_cost['valid_item_cells'][2]
                        end_coord = path_cost['valid_item_cells'][0][-1]
                        valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                        path_actions += valid_drop_path_actions
                        path_actions.append([
                            'DROP',
                            {
                                'for_task': 'SERVE'
                            }
                        ])
                    elif isinstance(self.holding, Plate) and self.holding.state != task_info['state']:
                        dish = self.get_recipe_name(goal)
                        try:
                            pot_cells = [pot.location for pot in self.world_state['pot'] if pot.dish == dish]
                            collection_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                            task_coord = collection_path_cost['pot'][2]
                            end_coord = collection_path_cost['pot'][0][-1]
                            collection_path_actions = self.map_path_actions(collection_path_cost['pot'][0])
                            path_actions += collection_path_actions

                            goal = goal - 1 # assume SCOOP is always before SERVE
                            path_actions.append([
                                'SCOOP',
                                {
                                    'is_last': True,
                                    'task_coord': task_coord
                                },
                                end_coord
                            ])
                        # If pot with dish does not exists
                        except IndexError:
                            continue
                    elif isinstance(self.holding, Plate) and self.holding.state == task_info['state']:
                        print('@base_agent - Entered serve logic')
                        service_path_cost = self.calc_travel_cost(['service_counter'], [self.world_state['service_counter']])
                        task_coord = service_path_cost['service_counter'][2]
                        end_coord = service_path_cost['service_counter'][0][-1]
                        service_path_actions = self.map_path_actions(service_path_cost['service_counter'][0])
                        path_actions += service_path_actions

                        path_actions.append([
                            'SERVE',
                            {
                                'is_last': True,
                                'task_coord': task_coord
                            },
                            end_coord
                        ])
                else:
                    print('@base_agent - Entered serve logic')
                    # If plate exists in the map and there is dish to serve
                    if sum(self.world_state['cooked_dish_count'].values()) > 0:
                        plate_board_cells = [plate.location for plate in self.world_state['plate']]
                        plate_path_cost = self.calc_travel_cost(['plate'], [plate_board_cells])
                        task_coord = plate_path_cost['plate'][2]
                        end_coord = plate_path_cost['plate'][0][-1]
                        path_actions += self.map_path_actions(plate_path_cost['plate'][0])

                        path_actions.append([
                            'PICK',
                            {
                                'is_new': False,
                                'is_last': False,
                                'pick_type': 'plate',
                                'task_coord': task_coord,
                                'for_task': 'SERVE'
                            },
                            end_coord
                        ])
                    else:
                        continue

            for action in path_actions:
                try:
                    total_rewards += self.rewards[self.actions[action]]
                except TypeError:
                    action_abbrev = action[0]
                    # Give more importance to picking plate (to serve)
                    # Missing penalty for taking same task (steps to take after pursueing something another agent is doing)
                    if action_abbrev == 'PICK' and action[1]['for_task'] == 'COOK':
                        total_rewards += 40 - 10
                    elif action_abbrev == 'PICK' and action[1]['for_task'] == 'SCOOP':
                        total_rewards += 50 - 10
                    elif action_abbrev == 'PICK' and action[1]['for_task'] == 'SERVE':
                        total_rewards += 100 - 10
                    else:
                        total_rewards += self.rewards[action_abbrev]
            if self.contains_invalid(path_actions):
                pass
            else:
                agent_goal_costs[goal] = {
                    'steps': path_actions,
                    'rewards': total_rewards
                }
        return agent_goal_costs
    
    def contains_invalid(self, check_list):
        check_list_coords = []
        for ele in check_list:
            if type(ele) == int:
                new_cell_movemet = MAP_ACTIONS[ACTIONS[ele]]
                if not check_list_coords:
                    new_coords = tuple([sum(x) for x in zip(list(self.location), new_cell_movemet)])
                else:
                    new_coords = tuple([sum(x) for x in zip(list(check_list_coords[-1]), new_cell_movemet)])
                check_list_coords.append(new_coords)

        invalid_flag = False
        for coords in check_list_coords:
            if coords in self.world_state['invalid_movement_cells']:
                invalid_flag = True
                break
        return invalid_flag

    def get_recipe_ingredient_count(self, recipe, ingredient):
        recipe_ingredient_count = RECIPES_INFO[recipe][ingredient]
        
        return recipe_ingredient_count

    def get_recipe_dish(self, ingredient):
        recipe_dish = None
        for recipe in RECIPES_INFO:
            if RECIPES_INFO[recipe]['ingredient'] == ingredient:
                recipe_dish = recipe
        
        return recipe_dish
    
    def get_recipe_total_ingredient_count(self, recipe):
        return sum(RECIPES_INFO[recipe].values())

    def get_ingredient_name(self, task_id):
        ingredient_name = None
        for ingredient in INGREDIENT_ACTION_NAME:
            if task_id in INGREDIENT_ACTION_NAME[ingredient]:
                ingredient_name = ingredient
        return ingredient_name
    
    def get_recipe_name(self, task_id):
        recipe_name = None
        for recipe in RECIPE_ACTION_NAME:
            if task_id in RECIPE_ACTION_NAME[recipe]:
                recipe_name = recipe
        return recipe_name

    def complete_cooking_check(self, recipe, ingredient_counts):
        return RECIPES_INFO[recipe] == ingredient_counts
    
    def get_general_goal_id(self, recipe, action):
        return RECIPES_ACTION_MAPPING[recipe]['general'][action]

    # ACTIONS
    def pick(self, task_id: int, pick_info) -> None:
        print('agent@pick')
        is_new = pick_info['is_new']
        is_last = pick_info['is_last']
        pick_type = pick_info['pick_type']
        task_coord = pick_info['task_coord']

        if pick_type == 'ingredient':
            ingredient_name = self.get_ingredient_name(task_id)
            if is_new:
                self.world_state['goal_space_count'][task_id] -= 1
                self.world_state['goal_space_count'][task_id+1] += 1
                state = 'unchopped'
                new_ingredient = Ingredient(
                    ingredient_name,
                    state,
                    'ingredient',
                    task_coord
                )
                new_ingredient.location = tuple(self.location)
                self.holding = new_ingredient
            if is_last:
                try:
                    self.world_state['goal_space'][task_id].pop(0)
                except IndexError:
                    # Both Agents try to pick at the same time
                    pass
                self.world_state['goal_space'][task_id+1].append({
                    'state': 'unchopped',
                    'ingredient': ingredient_name
                })
            else:
                print(f'IndexError when trying to pop goal_space - PICK')
                ingredient_location = [ingredient.location for ingredient in self.world_state['ingredients']]

                if task_coord in ingredient_location:
                    # if another agent took it already and is now missing; do nothing
                    old_ingredient = [
                        ingredient for ingredient in self.world_state['ingredients'] if ingredient.location == task_coord
                    ][0]

                    # check if ingredient to be picked is on chopping_board
                    cb = [chopping_board for chopping_board in self.world_state['chopping_board'] if chopping_board.location == old_ingredient.location]
                    if len(cb) == 1:
                        cb[0].state = 'empty'

                    # update ingredient to be held by agent and in agent's current position
                    old_ingredient.location = tuple(self.location)
                    self.holding = old_ingredient

                    # Need to remove from world state after agent has picked it
                    self.world_state['ingredients'] = [ingredient for ingredient in self.world_state['ingredients'] if id(ingredient) != id(old_ingredient)]
        
        elif pick_type == 'plate':
            plate_location = [plate.location for plate in self.world_state['plate']]
            if task_coord in plate_location:
                # if another agent took it already and is now missing; do nothing
                old_plate = [
                    plate for plate in self.world_state['plate'] if plate.location == task_coord
                ][0]

                # update plate to be held by agent and in agent's current position
                old_plate.location = tuple(self.location)
                self.holding = old_plate

                # Need to remove from world state after agent has picked it
                self.world_state['plate'] = [plate for plate in self.world_state['plate'] if id(plate) != id(old_plate)]

    def find_random_empty_cell(self) -> Tuple[int,int]:
        all_valid_surrounding_cells = []
        surrounding_cells_xy = [
            [-1,0], [0,1], [1,0], [0,-1]
            # [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1]
        ]

        temp_valid_cells = self.world_state['valid_item_cells'].copy()
        for surrounding_cell_xy in surrounding_cells_xy:
            surrounding_cell = [sum(x) for x in zip(self.location, surrounding_cell_xy)]
            if tuple(surrounding_cell) in temp_valid_cells:
                all_valid_surrounding_cells.append(tuple(surrounding_cell))

        return random.choice(all_valid_surrounding_cells)

    def drop(self, task_id: int) -> None:
        """
        This action assumes agent is currently holding an item.
        Prerequisite
        ------------
        - Drop item at where agent is currently

        TO IMPROVE:
        - Prevent stacking of item
        - Dropping item blocks grid cell
        """
        print('base_agent@drop - Drop item in-hand')
        random_empty_cell = self.find_random_empty_cell()

        if type(self.holding) == Ingredient:
            holding_ingredient = self.holding
            holding_ingredient.location = random_empty_cell
            self.world_state['ingredients'].append(holding_ingredient)
        # For now, just plate
        elif type(self.holding) == Plate:
            holding_plate = self.holding
            holding_plate.location = random_empty_cell
            self.world_state['plate'].append(holding_plate)
        self.holding = None

    def chop(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('agent@chop')
        print(task_id)
        ingredient_name = self.get_ingredient_name(task_id)
        recipe_name = self.get_recipe_name(task_id)
        self.world_state['explicit_rewards']['chop'] += 1
        self.world_state['goal_space_count'][task_id] -= 1
        self.world_state['goal_space_count'][task_id+1] += 1
        if is_last:
            print('base_agent@chop - Remove chopping task')
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['goal_space'][task_id+1].append({
                'state': 'chopped',
                'ingredient': ingredient_name,
                'recipe': recipe_name
            })

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand
        holding_ingredient.location = task_coord

        # agent drops ingredient to chopping board
        self.holding = None
        holding_ingredient.state = 'chopped'
        self.world_state['ingredients'].append(holding_ingredient)
        used_chopping_board = [board for board in self.world_state['chopping_board'] if board.location == task_coord][0]

        # update chopping board to be 'taken'
        used_chopping_board.state = 'taken'
        
    def cook(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('agent@cook')
        ingredient_name = self.get_ingredient_name(task_id)
        dish = self.get_recipe_name(task_id)
        ingredient_count = self.get_recipe_ingredient_count(dish, ingredient_name)

        # Find the chosen pot - useful in maps with more than 1 pot
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        if pot.ingredient_count[ingredient_name] == ingredient_count:
            # Maxed out already
            pass
        else:
            self.world_state['explicit_rewards']['cook'] += 1
            self.world_state['goal_space_count'][task_id] -= 1
            self.world_state['goal_space'][task_id].pop(0)

            holding_ingredient = self.holding
            # only update location after reaching, since ingredient is in hand
            holding_ingredient.location = task_coord

            # agent drops ingredient to pot
            pot.ingredient_count[ingredient_name] += 1
            pot.is_empty = False

            # remove ingredient from world state since used for cooking
            for idx, ingredient in enumerate(self.world_state['ingredients']):
                if id(ingredient) == id(holding_ingredient):
                    del self.world_state['ingredients'][idx]
                    break
            # remove ingredient from agent's hand since no longer holding
            self.holding = None

            # if pot.ingredient_count == ingredient_count:
            if self.complete_cooking_check(dish, pot.ingredient_count):
                print('agent@cook - Add completed dish to pot')
                print(task_id)

                # Create new Dish Class object
                new_dish = Dish(dish, pot.location)
                self.world_state['cooked_dish'].append(new_dish)
                pot.dish = dish

                # Add Scoop to Goal Space
                new_task_id = self.get_general_goal_id(dish, 'SCOOP')
                print(new_task_id)
                self.world_state['goal_space_count'][new_task_id] += 1
                self.world_state['goal_space'][new_task_id].append({
                    'state': 'empty',
                    'ingredient': ingredient_name
                })

    def scoop(self, task_id: int, scoop_info):
        print('agent@scoop')
        dish = self.get_recipe_name(task_id)
        is_last = scoop_info['is_last']
        task_coord = scoop_info['task_coord']
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

         # Empty the pot as well
        pot.ingredient_count = defaultdict(int)
        pot.is_empty = True
        pot.dish = None

        try:
            dish_to_plate = [dish for dish in self.world_state['cooked_dish'] if dish.location == task_coord][0]
            # let the plate which agent is holding, hold the completed dish
            self.world_state['cooked_dish_count'][dish_to_plate.name] -= 1
            self.holding.dish = dish_to_plate
            self.holding.state = 'plated'

            # remove dish from world state since used for plating
            for idx, cooked_dish in enumerate(self.world_state['cooked_dish']):
                if id(cooked_dish) == id(dish_to_plate):
                    del self.world_state['cooked_dish'][idx]
                    break

            self.world_state['goal_space_count'][task_id] -= 1
            if is_last:
                print('base_agent@scoop - Remove scooping task')
                new_task_id = self.get_general_goal_id(dish, 'SERVE')
                self.world_state['goal_space'][task_id].pop(0)
                self.world_state['goal_space_count'][new_task_id] += 1
                self.world_state['goal_space'][new_task_id].append({
                    'state': 'plated',
                    'dish': dish
                })
        except IndexError:
            # Both Agents try to scoop at the same time
            pass
        
    def serve(self, task_id: int, serve_info):
        print('agent@serve')
        self.world_state['explicit_rewards']['serve'] += 1
        self.world_state['total_score'] += self.world_state['score'].pop(0)
        is_last = serve_info['is_last']

        # plate returns to return point (in clean form for now)
        self.holding.dish = None
        self.holding.state = 'empty'
        self.holding.location = self.world_state['return_counter']
        self.world_state['plate'].append(self.holding)

        # remove dish from plate
        self.holding = None

        # remove order from TaskList
        if is_last:
            print('base_agent@serve - Remove serve task')
            print(task_id)
            self.world_state['goal_space_count'][task_id] -= 1
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['order_count'] -= 1

    def observer_inference(self):
        """Perform inference derivation"""
        print(f'Replicate prev environment')
        from overcooked_env import OvercookedEnv
        prev_env = OvercookedEnv()
        prev_env.world_state = self.world_state['historical_world_state']
        prev_best_goals = prev_env.find_agents_possible_goals()

        print(f'Previous best goals')
        print(prev_best_goals)

        # Considers all other agents except itself - allows for > 2 agents inference
        prev_best_goals = {agent: info for agent, info in prev_best_goals.items() if agent.id != self.id}
        print(f'Previous best goals - after filtering out own best goal')
        print(prev_best_goals)

        inferred_goals_info = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for agent in prev_best_goals:
            for goal in prev_best_goals[agent]:

                sampling_count = 0
                all_best_paths = prev_env.generate_possible_paths(agent, prev_best_goals[agent][goal])
                while sampling_count != 100:

                    best_path = None
                    if all_best_paths != -1:
                        best_path = random.choice(all_best_paths)
                        best_path.append(prev_best_goals[agent][goal]['steps'][-1])
                    else:
                        best_path = -1
                    try:
                        if isinstance(best_path[0], int):
                            inferred_goals_info[agent][best_path[0]] += 1

                    except TypeError:
                        # Case where best_path = -1
                        print(f'@observer_inference - TypeError')
                        print(f'Encountered best action to take, is not a movement.')

                        # print(prev_best_goals[agent])
                        # print(prev_best_goals[agent][goal])
                        # print(prev_best_goals[agent][goal]['steps'][-1])
                        # print(best_path)

                        non_movement_action = prev_best_goals[agent][goal]['steps'][-1]
                        if non_movement_action[0] == 'PICK':
                            inferred_goals_info[agent][goal][9] += 1
                        elif non_movement_action[0] == 'CHOP':
                            inferred_goals_info[agent][goal][10] += 1
                        elif non_movement_action[0] == 'COOK':
                            inferred_goals_info[agent][goal][11] += 1
                        elif non_movement_action[0] == 'SCOOP':
                            inferred_goals_info[agent][goal][12] += 1
                        elif non_movement_action[0] == 'SERVE':
                            inferred_goals_info[agent][goal][13] += 1
                        elif non_movement_action[0] == 'DROP':
                            inferred_goals_info[agent][goal][14] += 1
                    sampling_count += 1
                # Apply laplace smoothing
                inferred_goals_info[agent][goal] = self._laplace_smoothing(inferred_goals_info[agent][goal], 'action')

        print(f'Done with gathering samples')
        print(inferred_goals_info)
        inferred_goals_conditional_distribution = _get_conditional_distribution(inferred_goals_info)
        print(f'Done with deriving conditional distribution')
        print(inferred_goals_conditional_distribution)
        observer_task_to_not_do = self.observer_coordination_planning(inferred_goals_conditional_distribution)

        return observer_task_to_not_do

    def observer_coordination_planning(self, action_conditional_distribution):
        """
        PARAMETERS
        ----------
        action_conditional_distribution
            Make use of P(goal,t|action,t) to derive P(a,t+1|goal,t)
        """
        print(f'@observer_coordination_planning')
        print(self.world_state['historical_actions'])
        print(action_conditional_distribution)
        goal_probability_distribution = {}
        for agent in action_conditional_distribution:
            agent_prev_action = self.world_state['historical_actions'][agent.id][-1]
            print(f'Agent {agent.id} Prev action')
            print(agent_prev_action)
            try:
                goal_probability_distribution[agent] = {k:v for k,v in action_conditional_distribution[agent].items() if k == agent_prev_action}[agent_prev_action]
            except TypeError:
                print(f'@observer_coordination_planning - TypeError - {agent_prev_action[0]}')
                if agent_prev_action[0] == 'PICK':
                    agent_prev_action = 9
                elif agent_prev_action[0] == 'CHOP':
                    agent_prev_action = 10
                elif agent_prev_action[0] == 'COOK':
                    agent_prev_action = 11
                elif agent_prev_action[0] == 'SCOOP':
                    agent_prev_action = 12
                elif agent_prev_action[0] == 'SERVE':
                    agent_prev_action = 13
                elif agent_prev_action[0] == 'DROP':
                    agent_prev_action = 14
                goal_probability_distribution[agent] = {k:v for k,v in action_conditional_distribution[agent].items() if k == agent_prev_action}[agent_prev_action]
        print(goal_probability_distribution)

        # Run one episode for other agents from current world state
        print(f'Replicate current environment')
        from overcooked_env import OvercookedEnv
        curr_env = OvercookedEnv()
        curr_env.world_state = copy.deepcopy(self.world_state)
        curr_best_goals = curr_env.find_agents_possible_goals()
        print('\nCurrent best goals - Pre-smoothing')
        print(curr_best_goals)

        GOAL_SPACE = list(self.world_state['goal_space'])
        curr_best_goals = self._laplace_smoothing(curr_best_goals, 'goals', GOAL_SPACE)
        print('\nCurrent best goals - Post-smoothing')
        print(curr_best_goals)

        # Perform goal weighting to select best next action to take
        goal_probability_distribution = {agent.id:val for agent, val in goal_probability_distribution.items()}
        curr_best_goals = {agent.id:val for agent, val in curr_best_goals.items()}
        print(f'\nAll goal probability distribution')
        print(goal_probability_distribution)
        print(f'\nRe-mapping of Current best goals')
        print(curr_best_goals)

        # Edge case: Task gets removed from TaskList (eg. after cooking/serving)
        # for agent in curr_best_goals:
        filtered_goal_probability_distribution = defaultdict(lambda: defaultdict(float))
        for agent in goal_probability_distribution:
            for task in goal_probability_distribution[agent]:
                temp_task_prob = goal_probability_distribution[agent][task]
                if task in curr_best_goals[agent]:
                    # only bother with valid tasks
                    filtered_goal_probability_distribution[agent][task] = temp_task_prob
        goal_probability_distribution = filtered_goal_probability_distribution
        print(f'All goal probability distribution after filtering')
        print(goal_probability_distribution)

        # Current as in look-ahead this current episode
        curr_agent_goal_rewards = {}
        for agent in goal_probability_distribution:
            curr_agent_goal_rewards[agent] = {
                k:goal_probability_distribution[agent][k]*curr_best_goals[agent][k]['rewards'] 
                for k,v in goal_probability_distribution[agent].items()
            }
        print(f'\nCurrent agent goal rewards')
        print(curr_agent_goal_rewards)

        all_agents_best_inferred_goals = {}
        for agent in curr_agent_goal_rewards:
            # Randomly choose one if multiple goals have the same weighting
            best_goal_list = list()
            max_weighted_reward = max(curr_agent_goal_rewards[agent].items(), key=lambda x: x[1])[1]
            for task_id, weighted_reward in curr_agent_goal_rewards[agent].items():
                if weighted_reward == max_weighted_reward:
                    best_goal_list.append(task_id)
            
            all_agents_best_inferred_goals[agent] = random.choice(best_goal_list)

        print(f'\nCurrent Best Inferred goal')
        print(all_agents_best_inferred_goals)

        observer_goal_to_not_do = []
        agent_own_id = self.id
        print(f'Own Agent Best goals')
        print(curr_best_goals[agent_own_id])
        for agent in all_agents_best_inferred_goals:
            print(f'Other agent.. Agent {agent} Best goals')
            print(curr_best_goals[agent])
            print(curr_best_goals[agent][all_agents_best_inferred_goals[agent]])
            print(all_agents_best_inferred_goals[agent])
            if curr_best_goals[agent][all_agents_best_inferred_goals[agent]]['rewards'] > \
                curr_best_goals[agent_own_id][all_agents_best_inferred_goals[agent]]['rewards']:
                observer_goal_to_not_do.append(all_agents_best_inferred_goals[agent])
        print(f'\nObserver to not do goals')
        print(observer_goal_to_not_do)

        # Not do goal only if there's only 1 goal to pursue
        multiple_goals = []
        for goal in observer_goal_to_not_do:
            if self.world_state['goal_space_count'][goal] > 1:
                multiple_goals.append(goal)
        observer_goal_to_not_do = [goal for goal in observer_goal_to_not_do if goal not in multiple_goals]
        print(f'\nAfter checking whether to pursue anyway')
        print(observer_goal_to_not_do)

        return observer_goal_to_not_do

    def _laplace_smoothing(self, p_distribution, type, goal_space=None):
        if type == 'action':
            ACTION_SPACE = 15 # up to 14 actions
            for i in range(ACTION_SPACE):
                p_distribution[i] += 1
        elif type == 'goals':
            GOALS_SPACE = goal_space
            temp_p_distribution = defaultdict(dict)
            for agent in p_distribution:
                for goal_id in GOALS_SPACE:
                    if goal_id not in p_distribution[agent]:
                        # Already filtered out by agent.find_best_goal() function
                        # Require this to prevent key error when doing goal weighting
                        # p_distribution[agent][goal_id] = {'rewards': 0}
                        temp_p_distribution[agent][goal_id] = {'rewards': 0}
                    else:
                        temp_p_distribution[agent][goal_id] = p_distribution[agent][goal_id]
            p_distribution = temp_p_distribution

        return p_distribution

class RLAgent(OvercookedAgent):
    def __init__(self,agent_id, location, barriers, is_inference_agent=False, is_assigned=False, can_update=True, goals=None,
                    holding=None,actions=ACTIONS, rewards=REWARDS):
        super().__init__(
        agent_id,
        location,
        barriers,
        is_inference_agent=False,
        is_assigned=False,
        can_update=True,
        goals=None,
        holding=None,
        actions=ACTIONS,
        rewards=REWARDS
    )
def _get_conditional_distribution(sampled_goal_space_actions):
    """
    Calculates conditional probability of taking each goal given action using Bayes rule.

    goal_1 = {1: 37, 2: 12, 3: 35, 4: 3}
    goal_2 = {1: 54, 2: 14, 3: 23, 4: 13}
    P(goal_1|a=1) = P(a=1|goal_1)P(goal_1)/SUMMATION(goal')P(a=1|goal')P(goal')

    Returns
    -------
    Dictionary of Agents
        of Dictionary of Actions
            of Dictionary of Goals
    which represents P(a|goal).
    """
    print(f'@_conditional_distribution')
    ACTION_SPACE = 15 # up to 14 actions
    conditional_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    for agent in sampled_goal_space_actions:
        for action in range(ACTION_SPACE):
            total_sampled_counts = sum([val for goal in sampled_goal_space_actions[agent] for key, val in sampled_goal_space_actions[agent][goal].items() if key == action])
            for goal_id in sampled_goal_space_actions[agent]:
                cur_goal_sampled_count = sampled_goal_space_actions[agent][goal_id][action]
                conditional_distribution[agent][action][goal_id] = cur_goal_sampled_count/total_sampled_counts

    return conditional_distribution
