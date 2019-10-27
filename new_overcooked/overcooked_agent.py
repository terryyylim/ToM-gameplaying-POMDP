from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import numpy as np
import random

from agent_configs import ACTIONS, REWARDS
from astar_search import AStarGraph
from overcooked_item_classes import Ingredient, Plate, Dish
from settings import RECIPES_INFO, INGREDIENTS_INITIALIZATION

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
            if goal in [0, 5]:
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
                            'task_coord': task_coord
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
                            'for_task': holding_type
                        }
                    ])
            # CHOP GOALS - onion, tomato
            if goal in [1,6]:
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
                                'for_task': holding_type
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
                                    'for_task': holding_type
                                }
                            ])
                        elif self.holding.state == task_info['state']:
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
                                'task_coord': task_coord
                            },
                            end_coord
                        ])
                    else:
                        # Should we make default behaviour to pick up new ingredient from crate?
                        pass

            # COOK GOALS - onion, tomato
            if goal in [2,7]:
                """ CONDITION TO FULFIL: Holding chopped onion """
                print(f'@agent - Entered COOK logic')
                path_actions = []
                task_info = self.world_state['goal_space'][goal][0]
                print(task_info)
                print(self.holding)

                wanted_ingredient = [
                    ingredient.location for ingredient in self.world_state['ingredients'] if \
                        (ingredient.name == task_info['ingredient'] and ingredient.state == task_info['state'])]
                if self.holding:
                    print('am holding')
                    print(self.holding)
                    print(self.holding.name)
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
                                'for_task': holding_type
                            }
                        ])
                    elif isinstance(self.holding, Ingredient) and self.holding.name == task_info['ingredient']:
                        print(f'SOMEHOW HERE')
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
                                    'for_task': holding_type
                                }
                            ])
                        elif self.holding.state == task_info['state']:
                            print(f'YEAAAA IN STATE')
                            recipe_ingredient_count = self.get_recipe_ingredient_count(task_info['ingredient'])
                            # Fill pot with ingredient before considering empty pots
                            pot_cells = [pot.location for pot in self.world_state['pot'] if pot.ingredient == task_info['ingredient'] if pot.ingredient_count < recipe_ingredient_count]
                            print('pot_cells1')
                            print(pot_cells)
                            if not pot_cells:
                                pot_cells = [pot.location for pot in self.world_state['pot'] if pot.state == 'empty']
                            print('pot_cells2')
                            print(pot_cells)
                            if pot_cells:
                                cooking_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                                print('pot_cells3')
                                print(cooking_path_cost)

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
                                pass
                else:
                    # Case: Not holding ingredient but it exist in map
                    print('YEAH PICKING')
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
                                'task_coord': task_coord
                            },
                            end_coord
                        ])
                    else:
                        # Should we make default behaviour to pick up new ingredient from crate?
                        print(f'SHOULD WE DEFAULT BEHAVIOUR?')
                        pass
            
            # SCOOP GOALS - onion, tomato
            if goal in [3,8]:
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
                                'for_task': holding_type
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
                                'for_task': holding_type
                            }
                        ])
                    elif isinstance(self.holding, Plate) and self.holding.state == task_info['state']:
                        dish = self.get_recipe_dish(task_info['ingredient'])
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
                                'task_coord': task_coord
                            },
                            end_coord
                        ])
                    except IndexError:
                        print('@base_agent - scoop IndexError')
                        continue
            
            # SERVE GOALS - onion, tomato
            if goal in [4,9]:
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
                                'for_task': holding_type
                            }
                        ])
                    elif isinstance(self.holding, Plate) and self.holding.state != task_info['state']:
                        dish = self.get_recipe_dish(task_info['ingredient'])
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
                            'task_coord': task_coord
                        },
                        end_coord
                    ])

            for action in path_actions:
                try:
                    total_rewards += self.rewards[self.actions[action]]
                except TypeError:
                    action_abbrev = action[0]
                    # Give more importance to picking plate (to serve)
                    if action_abbrev == 'PICK' and action[1]['pick_type'] == 'plate':
                        total_rewards += self.rewards[action_abbrev] + 30
                    elif action_abbrev == 'DROP' and action[1]['for_task'] == 'PLATE':
                        total_rewards += self.rewards[action_abbrev] + 30
                    elif action_abbrev == 'DROP' and action[1]['for_task'] == 'INGREDIENT':
                        total_rewards += self.rewards[action_abbrev] + 10
                    else:
                        total_rewards += self.rewards[action_abbrev]
            agent_goal_costs[goal] = {
                'steps': path_actions,
                'rewards': total_rewards
            }
        return agent_goal_costs

    def get_recipe_ingredient_count(self, ingredient):
        recipe_ingredient_count = None
        for recipe in RECIPES_INFO:
            if RECIPES_INFO[recipe]['ingredient'] == ingredient:
                recipe_ingredient_count = RECIPES_INFO[recipe]['count']
        
        return recipe_ingredient_count

    def get_recipe_dish(self, ingredient):
        recipe_dish = None
        for recipe in RECIPES_INFO:
            if RECIPES_INFO[recipe]['ingredient'] == ingredient:
                recipe_dish = recipe
        
        return recipe_dish

    def get_ingredient_name(self, task_id):
        ingredient = None
        if task_id in [0, 1, 2, 3, 4]:
            ingredient = 'onion'
        elif task_id in [5, 6, 7, 8, 9]:
            ingredient = 'tomato'
        return ingredient

    # ACTIONS
    def pick(self, task_id: int, pick_info) -> None:
        print('agent@pick')
        print(task_id)
        print(pick_info)
        print(self.world_state['goal_space_count'])
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
                    INGREDIENTS_INITIALIZATION[ingredient_name]
                )
                new_ingredient.location = tuple(self.location)
                self.holding = new_ingredient
            if is_last:
                self.world_state['goal_space'][task_id].pop(0)                
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
            [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1]
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
        self.world_state['explicit_rewards']['chop'] += 1
        self.world_state['goal_space_count'][task_id] -= 1
        self.world_state['goal_space_count'][task_id+1] += 1
        print('goal space check')
        print(self.world_state['goal_space_count'])
        if is_last:
            print('base_agent@chop - Remove chopping task')
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['goal_space'][task_id+1].append({
                'state': 'chopped',
                'ingredient': ingredient_name
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
        ingredient_count = self.get_recipe_ingredient_count(ingredient_name)
        dish = self.get_recipe_dish(ingredient_name)
        self.world_state['explicit_rewards']['cook'] += 1
        self.world_state['goal_space_count'][task_id] -= 1
        self.world_state['goal_space'][task_id].pop(0)

        # Find the chosen pot - useful in maps with more than 1 pot
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand
        holding_ingredient.location = task_coord

        # agent drops ingredient to pot
        pot.ingredient = ingredient_name
        pot.ingredient_count += 1

        # remove ingredient from world state since used for cooking
        for idx, ingredient in enumerate(self.world_state['ingredients']):
            if id(ingredient) == id(holding_ingredient):
                del self.world_state['ingredients'][idx]
                break
        # remove ingredient from agent's hand since no longer holding
        self.holding = None

        if pot.ingredient_count == ingredient_count:
            print('base_agent@cook - Add completed dish to pot')

            # Create new Dish Class object
            new_dish = Dish(dish, pot.location)
            self.world_state['cooked_dish'].append(new_dish)
            pot.dish = dish

            # Add Scoop to Goal Space
            self.world_state['goal_space_count'][task_id+1] += 1
            self.world_state['goal_space'][task_id+1].append({
                'state': 'empty',
                'ingredient': ingredient_name
            })

    def scoop(self, task_id: int, scoop_info):
        print('agent@scoop')
        ingredient_name = self.get_ingredient_name(task_id)
        is_last = scoop_info['is_last']
        task_coord = scoop_info['task_coord']
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

         # Empty the pot as well
        pot.ingredient = None
        pot.ingredient_count = 0
        pot.dish = None

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
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['goal_space_count'][task_id+1] += 1
            self.world_state['goal_space'][task_id+1].append({
                'state': 'plated',
                'ingredient': ingredient_name
            })
        
    def serve(self, task_id: int, serve_info):
        print('agent@serve')
        self.world_state['explicit_rewards']['serve'] += 1
        is_last = serve_info['is_last']

        # plate returns to return point (in clean form for now)
        self.holding.dish = None
        self.holding.state = 'empty'
        self.holding.location = (5,0)
        self.world_state['plate'].append(self.holding)

        # remove dish from plate
        self.holding = None

        # remove order from TaskList
        if is_last:
            print('base_agent@serve - Remove serve task')
            self.world_state['goal_space_count'][task_id] -= 1
            self.world_state['goal_space'][task_id].pop(0)
