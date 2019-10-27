from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# import ray
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool, Process, Queue, cpu_count
import queue
import threading
import matplotlib.pyplot as plt
import numpy as np
import operator
import random
import copy
import os

from agent_configs import *
from astar_search import AStarGraph
from overcooked_classes import *


class BaseAgent:
    def __init__(
        self,
        agent_id,
        location
    ) -> None:
        """
        Parameters
        ----------
        agent_id: str
            a unique id allowing the map to identify the agents
        location: Tuple[int,int]
            x-y coordinate of agent
        """

        self.agent_id = agent_id
        self.location = location

    @property
    def action_space(self):
        """
        Identify the dimension and bounds of the action space.

        MUST BE implemented in new environments.
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

class OvercookedAgent(BaseAgent):
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
        super().__init__(agent_id, location)
        """
        can_update: True if hasn't update for this task else False
        """
        self.world_state = {}
        self.id = agent_id
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

    def find_best_goal(self, observer_task_to_not_do=None):
        """
        Finds all path and costs of give n possible action space.

        For picking, keep track of `old` vs `new` because when performing action later on,
        only `new` requires creation of new Ingredient object.

        All actions return
        ------------------
        task_coord: coordinate to perform task on eg. pick ingredient at task_coord
        end_coord: coordinate where agent must be at before performing task at task_coord
        """
        agent_goal_costs = defaultdict(dict)

        for task_list in self.world_state['goal_space']:
            if id(task_list) in observer_task_to_not_do:
                continue
            else:
                path_actions = []
                # If task is to plate/scoop/serve, to check for dish, not ingredients
                if task_list.head.task == 'plate':
                    print('@base_agent - Entered plate logic')
                    if not isinstance(self.holding, Plate):
                        # Go to plate, Do picking

                        if isinstance(self.holding, Ingredient):
                            # If is holding ingredient, need to drop first
                            # Edge case: Randomly chosen spot to drop item must be a table-top cell
                            print('Contains Item in hand - Cannot pick')
                            print(self.holding)
                            print(task_list.ingredient)
                            path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                            task_coord = path_cost['valid_item_cells'][2]
                            end_coord = path_cost['valid_item_cells'][0][-1]
                            valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                            path_actions += valid_drop_path_actions
                            path_actions.append([
                                'DROP',
                                {
                                    'for_task': 'PLATE'
                                }
                            ])
                        else:
                            # In the case with multiple plates, a random one will be chosen
                            try:
                                plate_board_cells = [plate.location for plate in self.world_state['plate']]
                                plate_path_cost = self.calc_travel_cost(['plate'], [plate_board_cells])
                                task_coord = plate_path_cost['plate'][2]
                                end_coord = plate_path_cost['plate'][0][-1]
                                path_actions += self.map_path_actions(plate_path_cost['plate'][0])

                                path_actions.append([
                                    'PICK',
                                    {
                                        'is_new': False,
                                        'is_last': True,
                                        'pick_type': 'plate',
                                        'task_coord': task_coord
                                    },
                                    end_coord
                                ])
                            except IndexError:
                                print('@base_agent - plate IndexError')
                                # condition where agent is infront of plate
                                # no need to move anymore
                                end_coord = self.location
                                task_coord = None

                                plate_board_cells = [plate.location for plate in self.world_state['plate']]
                                for plate_cell in plate_board_cells:
                                    if (self.location[0]+1, self.location[1]) == plate_cell or \
                                            (self.location[0]-1, self.location[1]) == plate_cell or \
                                                (self.location[0], self.location[1]+1) == plate_cell or \
                                                    (self.location[0], self.location[1]-1) == plate_cell:
                                                        task_coord = plate_cell

                                path_actions.append([
                                    'PICK',
                                    {
                                        'is_new': False,
                                        'is_last': True,
                                        'pick_type': 'plate',
                                        'task_coord': task_coord
                                    },
                                    end_coord
                                ])
                elif task_list.head.task == 'scoop' or task_list.head.task == 'serve':
                    print('@base_agent - Entered scoop/serve logic')
                    if not isinstance(self.holding, Plate):
                        if self.world_state['cooked_dish_count'][task_list.dish] > 0:
                            print(f'Want scoop/serve but not holding plate')
                            
                            if isinstance(self.holding, Ingredient):
                                # If is holding ingredient, need to drop first
                                # Edge case: Randomly chosen spot to drop item must be a table-top cell
                                print('Contains Item in hand - Cannot pick')
                                path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                                task_coord = path_cost['valid_item_cells'][2]
                                end_coord = path_cost['valid_item_cells'][0][-1]
                                valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                                path_actions += valid_drop_path_actions
                                path_actions.append([
                                    'DROP',
                                    {
                                        'for_task': 'PLATE'
                                    }
                                ])
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
                                    # continue
                                except IndexError:
                                    print('@base_agent - scoop/serve IndexError')
                                    continue
                        else:
                            try:
                                print('inside pick filled plate')
                                # If filled plate exists in the map
                                plate_board_cells = [plate.location for plate in self.world_state['plate'] if plate.state == 'plated']
                                if plate_board_cells:
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
                                else:
                                    # Edge Case: Prevent PICK & DROP plate continuously while 'SERVE' is in order queue
                                    continue
                            except IndexError:
                                print('@base_agent - scoop/serve IndexError')
                                continue
                    else:
                        if task_list.head.task == 'scoop':
                            if self.world_state['cooked_dish_count'][task_list.dish] > 0:
                                print('@base_agent - Entered scoop logic')
                                # If pot with dish exists, Do scooping
                                try:
                                    pot_cells = [pot.location for pot in self.world_state['pot'] if pot.dish == task_list.dish]
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
                                # Edge Case: Prevent PICK & DROP plate continuously while 'SERVE' is in order queue
                                continue
                        elif task_list.head.task == 'serve':
                            try:
                                if self.holding.dish.name == task_list.dish:
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
                            except AttributeError:
                                continue
                # If task is not to plate/scoop/serve, to check for ingredients, not dish
                else:
                    wanted_ingredient = [
                        ingredient.location for ingredient in self.world_state['ingredients'] if \
                            (ingredient.name == task_list.ingredient and ingredient.state == task_list.head.state)]

                    if task_list.head.task == 'pick':
                        
                        print('@base_agent - Entered pick logic')
                        """
                        To fix/implement: Drop
                        - If already holding ingredient, can still pick?
                        - If pick, what happens to the one in hand?
                        """
                        # If no such ingredient exist
                        if not wanted_ingredient and not self.holding:
                            # Just take fresh ones
                            path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [self.world_state['ingredient_'+task_list.ingredient]])
                            # no need to move anymore
                            task_coord = self.world_state['ingredient_'+task_list.ingredient][0]
                            end_coord = self.location
                            if path_cost:
                                end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                                path_actions += self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])

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
                        elif not wanted_ingredient and self.holding:
                            # TO-BE FIXED (?)
                            if isinstance(self.holding, Ingredient):
                                if self.holding.name == task_list.ingredient:
                                    print('Contains Item in hand - But valid item for different task')
                                    continue
                            else:
                                # Edge case: Randomly chosen spot to drop item must be a table-top cell
                                print('Contains Item in hand - Cannot pick')
                                print(self.holding)
                                print(task_list.ingredient)
                                path_cost = self.calc_travel_cost(['valid_item_cells'], [self.world_state['valid_item_cells']])
                                task_coord = path_cost['valid_item_cells'][2]
                                end_coord = path_cost['valid_item_cells'][0][-1]
                                valid_drop_path_actions = self.map_path_actions(path_cost['valid_item_cells'][0])
                                path_actions += valid_drop_path_actions
                                path_actions.append([
                                    'DROP',
                                    {
                                        'for_task': 'INGREDIENT'
                                    }
                                ])
                        else:
                            # valid_ingredient_cells = [ingredient.location for ingredient in self.world_state['ingredients'] if ingredient.name == task_list.ingredient]
                            # path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [valid_ingredient_cells])
                            print('Ingredient required exist')
                            print(wanted_ingredient)
                            path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                            print(path_cost)
                            task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                            end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                            path_actions += self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])

                            path_actions.append([
                                'PICK',
                                {
                                    'is_new': False,
                                    'is_last': True,
                                    'pick_type': 'ingredient',
                                    'task_coord': task_coord
                                },
                                end_coord
                            ])
                    else:
                        # Does not guaranteed to have ingredient to chop/cook
                        print(f'@base_agent {id(self)} - Check for guaranteed presence of ingredient\n')

                        # Edge case: No available ingredient, this task is invalid
                        if not wanted_ingredient and not isinstance(self.holding, Ingredient):
                            continue
                        else:
                            # Get all paths + task, paths + task into [path_actions] array and returning
                            if task_list.head.task == 'chop':
                                print('@base_agent - Entered chop logic')

                                # If not holding ingredient: Go to ingredient, Do picking
                                if not isinstance(self.holding, Ingredient):
                                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                                    # no need to move anymore
                                    task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                                    end_coord = self.location
                                    if path_cost:
                                        end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                                        path_actions += self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])

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

                                # If holding ingredient: Go to chopping board, Do slicing
                                else:
                                    chopping_board_cells = [chopping_board.location for chopping_board in self.world_state['chopping_board'] if chopping_board.state == 'empty']
                                    chopping_path_cost = self.calc_travel_cost(['chopping_board'], [chopping_board_cells])
                                    # no need to move anymore ('might' have bug)
                                    task_coord = [board.location for board in self.world_state['chopping_board'] if board.state == 'empty'][0]
                                    end_coord = self.location
                                    if chopping_path_cost:
                                        task_coord = chopping_path_cost['chopping_board'][2]
                                        end_coord = chopping_path_cost['chopping_board'][0][-1]
                                        chopping_path_actions = self.map_path_actions(chopping_path_cost['chopping_board'][0])
                                        path_actions += chopping_path_actions
                                    path_actions.append(['CHOP', True, task_coord, end_coord])
                            elif task_list.head.task == 'cook':
                                print('@base_agent - Entered cook logic')

                                # If not holding (chopped) ingredient: Go to ingredient, Do picking
                                if not isinstance(self.holding, Ingredient) or self.holding.state != 'chopped':
                                    try:
                                        # If chopped ingredient exists in the map
                                        path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                                        task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                                        end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                                        path_actions += self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])

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
                                    # Edge case: No available chopped ingredient, this task is invalid
                                    except IndexError:
                                        continue

                                # If holding (chopped) ingredient: Go to pot, Do cooking
                                else:
                                    pot_cells = [pot.location for pot in self.world_state['pot']]
                                    cooking_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                                    # no need to move anymore
                                    # only works for 1 pot now
                                    task_coord = self.world_state['pot'][0].location
                                    end_coord = self.location

                                    task_dish, task_ingredient = task_list.dish, task_list.ingredient
                                    task_ingredient_count = RECIPES_INGREDIENTS_COUNT[task_dish][task_ingredient]
                                    pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]
                                    if pot.ingredient_count[task_ingredient] == task_ingredient_count:
                                        # Or should we drop?
                                        continue
                                    else:
                                        if cooking_path_cost:
                                            end_coord = cooking_path_cost['pot'][0][-1]
                                            cooking_path_actions = self.map_path_actions(cooking_path_cost['pot'][0])
                                            path_actions += cooking_path_actions
                                        path_actions.append(['COOK', True, task_coord, end_coord])

                total_rewards = 0
                print('@base_agent - Calculate total rewards')
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
                agent_goal_costs[id(task_list)] = {
                    'steps': path_actions,
                    'rewards': total_rewards
                }

        return agent_goal_costs

    def return_valid_pos(self, new_pos):
        """
        Checks that the next pos is legal, if not return current pos
        """
        if new_pos in self.world_state['valid_cells']:
            return new_pos

        # you can't walk through walls nor another agent
        
        return self.location

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
            print('got item instance')
            print(item_instance)
            print(agent_locs)
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

    def update_agent_pos(self, new_coords: List[int]) -> None:
        self.location = new_coords

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

    def action_map(self, action_number: int) -> str:
        return ACTIONS[action_number]

    def stay(self) -> Optional[List]:
        """
        Stay at current position and do nothing.
        """
        return [[], 0]

    def pick(self, task_id: int, pick_info) -> None:
        """
        This action assumes agent has already done A* search and decided which goal state to achieve.
        Prerequisite
        ------------
        - Item object passed to argument should be item instance
        - At grid with accessibility to item [GO-TO]

        pick_info
        ---------
        'is_new': True/False - whether to take from box and create new Ingredient object or not
        'is_last': True/False - whether this action is last of everything - to update TaskList
        'task_coord': coordinate to perform task on eg. pick ingredient at task_coord
        'end_coord': coordinate where agent must be at before performing task at task_coord

        TO-CONSIDER: 
        Do we need to check if agent is currently holding something?
        Do we need to set item coord to agent coord when the item is picked up?
        """
        is_new = pick_info['is_new']
        is_last = pick_info['is_last']
        pick_type = pick_info['pick_type']
        task_coord = pick_info['task_coord']

        try:
            task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]

            if pick_type == 'ingredient':
                if is_new:
                    print('base_agent@pick - is_new - create new ingredient')
                    ingredient_name = task.ingredient
                    state = task.head.state
                    new_ingredient = Ingredient(
                        ingredient_name,
                        state,
                        'ingredient',
                        INGREDIENTS_INITIALIZATION[ingredient_name]
                    )
                    new_ingredient.location = tuple(self.location)
                    new_ingredient.state = 'unchopped'
                    self.holding = new_ingredient

                if is_last and task.head.task == 'pick':
                    print('@base_agent@pick - last picked ingredient')
                    task.head = task.head.next
                else:
                    print('base_agent@pick - picking not is_new ingredient')
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

                        if is_last:
                            task.head = task.head.next
            
            elif pick_type == 'plate':
                print('base_agent@pick - plate')
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

                    # Edge case: both agents pick a plate at the same time (prevent task from being updated twice)
                    if is_last and task.head.task == 'plate':
                        task.head = task.head.next
        except IndexError:
            # Edge-Case: AgentL about to serve, AgentR trying to pick plate for the same goal that is going to be removed
            pass

    def find_random_empty_cell(self) -> Tuple[int,int]:
        all_valid_surrounding_cells = []
        surrounding_cells_xy = [
            [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1]
        ]

        temp_valid_cells = self.world_state['valid_item_cells'].copy()
        # for agent in self.world_state['agents']:
        #     temp_valid_cells.append(agent.location)
        for surrounding_cell_xy in surrounding_cells_xy:
            surrounding_cell = [sum(x) for x in zip(self.location, surrounding_cell_xy)]
            if tuple(surrounding_cell) in temp_valid_cells:
                all_valid_surrounding_cells.append(tuple(surrounding_cell))
        print(f'Found all valid surrounding cells')
        print(all_valid_surrounding_cells)

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
        print('base_agent@chop')
        self.world_state['explicit_rewards']['chop'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand (?)
        holding_ingredient.location = task_coord

        # agent drops ingredient to chopping board
        self.holding = None
        holding_ingredient.state = 'chopped'
        self.world_state['ingredients'].append(holding_ingredient)
        used_chopping_board = [board for board in self.world_state['chopping_board'] if board.location == task_coord][0]

        # update chopping board to be 'taken'
        used_chopping_board.state = 'taken'

        # Edge case: task.head.next can point to None if agentR reaches here after agentL already in midst of performing next task
        if is_last and task.head.next:
            print('base_agent@chop - Remove chopping task')
            task.head = task.head.next

    def cook(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('base_agent@cook')
        self.world_state['explicit_rewards']['cook'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        dish = task.dish
        ingredient = task.head.ingredient
        ingredient_count = RECIPES_INGREDIENTS_COUNT[dish]

        # Find the chosen pot - useful in maps with more than 1 pot
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand (?)
        holding_ingredient.location = task_coord

        # agent drops ingredient to pot
        pot.ingredient_count[ingredient] += 1

        # remove ingredient from world state since used for cooking
        for idx, ingredient in enumerate(self.world_state['ingredients']):
            if id(ingredient) == id(holding_ingredient):
                del self.world_state['ingredients'][idx]
                break
        # remove ingredient from agent's hand since no longer holding
        self.holding = None

        print('base_agent@cook - check for dish ingredients\' prep')
        if pot.ingredient_count == ingredient_count:
            print('base_agent@cook - Add completed dish to pot')

            # Create new Dish Class object
            new_dish = Dish(dish, pot.location)
            self.world_state['cooked_dish'].append(new_dish)
            self.world_state['task_id_count'] += 1
            self.world_state['goal_space'].append(TaskList(dish, RECIPES_SERVE_TASK[dish], dish, self.world_state['task_id_count']))
            self.world_state['cooked_dish_count'][dish] += 1
            for task_list in self.world_state['goal_space']:
                if task_list.id not in self.world_state['task_id_mappings']:
                    self.world_state['task_id_mappings'][task_list.id] = id(task_list)
            pot.dish = dish

        if is_last:
            print('base_agent@cook - Remove cooking task')
            task.head = task.head.next
            if task.head == None:
                for idx, task_no in enumerate(self.world_state['goal_space']):
                    if id(task_no) == task_id:
                        del self.world_state['goal_space'][idx]
                        break

    def scoop(self, task_id: int, scoop_info):
        """
        Current logic flaw
        ------------------
        Only works for dishes with 1 ingredient
        """
        print('base_agent@scoop')
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        is_last = scoop_info['is_last']
        task_coord = scoop_info['task_coord']
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        # Empty the pot as well
        pot.ingredient_count = defaultdict(int)
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
        if is_last and task.head.next:
            print('base_agent@scoop - Update plate -> serve task')
            task.head = task.head.next

    def serve(self, task_id: int, serve_info):
        print('base_agent@serve')
        self.world_state['explicit_rewards']['serve'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
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
            task.head = task.head.next
            if task.head == None:
                for idx, task_no in enumerate(self.world_state['goal_space']):
                    if id(task_no) == task_id:
                        del self.world_state['goal_space'][idx]
                        break

    def get_action(self) -> None:
        """
        Return best action for agent to perform.
        1. Get world state [self.world_state]
        2. Get best guess of observer's action
        3. Store temporary predicted future world state (beliefs)
        4. Perform action based on beliefs
        """
        # best_guess = self.best_guess_of_obs_action()
        # pred_future_state = self.get_predicted_future_world_state(best_guess)

    def best_guess_of_obs_action(self) -> None:
        """
        Current: Assume only 2 agents in environment.
        """
        observer = [value for key,value in self.world_state['agents'] if key != self.agent_id][0]
        best_guess = observer.get_action()
        return best_guess

    def get_predicted_future_world_state(self, best_guess):
        """
        Return temp world state where observer performed best_guess action
        """
        
    def get_best_action(self, predicted_future_world_state):
        """
        Find optimal action given temporary predicted future world state.

        // Do inference here? Using "observations" which is temporary predicted future world state?
        """

    def observer_inference(self):
        """
        Perform inference derivation.
        """
        print(f'Starting inference')
        print(f'Printing current agent references')
        print(self.location)
        print(self.id)
        agents_reference_move = {
            agent.id: {
                'prev_ref': agent,
                'prev_move': self.world_state['historical_actions'][agent.id][-1]
            } 
            for agent in self.world_state['historical_world_state']['agents'] if agent.id != self.id
        }

        for agent in self.world_state['agents']:
            if agent.id in agents_reference_move:
                agents_reference_move[agent.id]['cur_ref'] = agent
        print('Finished storing prev_move, prev_ref and cur_ref')
        print(agents_reference_move)

        print(f'Check if goals are correctly replicated')
        prev_goal_info = [(id(goal), goal.head, id(goal.head), goal.head.state, goal.head.task) for goal in self.world_state['goal_space']]
        cur_goal_info = [(id(goal), goal.head, id(goal.head), goal.head.state, goal.head.task) for goal in self.world_state['historical_world_state']['goal_space']]
        print(f'Previous goal info (goal_id; goal_head; goal_head_id; goal_head_state, goal_head_task): \n{prev_goal_info}\n')
        print(f'Current goal info (goal_id; goal_head; goal_head_id; goal_head_state, goal_head_task): \n{cur_goal_info}\n')

        print(f'Replicate prev environment')
        from ipomdp.envs.overcooked_map_env import OvercookedEnv
        prev_env = OvercookedEnv()
        prev_env.world_state = self.world_state['historical_world_state']
        prev_best_goals = prev_env.find_agents_possible_goals()

        # Considers all other agents except itself - allows for > 2 agents inference
        prev_best_goals = {agent: info for agent, info in prev_best_goals.items() if agent.id != self.id}
        print('prev best goals')
        print(prev_best_goals)

        # Eg. {1: 5143012944, 2: 5147600592, 3: 5147600528}
        print('check prev task_id mappings')
        print(prev_env.world_state['task_id_mappings'])
        
        # Eg. {1: 5249261392, 2: 5249260496, 3: 5249310800}
        deep_copy_task_id_mappings = {task.id:id(task) for task in prev_env.world_state['goal_space']}
        print('check cur task_id mappings')
        print(deep_copy_task_id_mappings)

        inferred_goals_info = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for agent in prev_best_goals:
            for goal in prev_best_goals[agent]:
                _id = list(deep_copy_task_id_mappings.keys())[list(deep_copy_task_id_mappings.values()).index(goal)]
                
                # Multi-processing solution (not useful in straightforward task here)
                # final_p = []
                # pool = Pool(processes=cpu_count())
                # for i in range(100):
                #     final_p += pool.starmap(self._do_sampling, [[prev_env, prev_best_goals, agent, goal, _id]])
                # pool.close()
                
                # print('pooling done')
                # print('final p')
                # print(final_p)
                # final_p = np.array(final_p)
                # final_p = np.sum(final_p, axis=0)
                # final_p = list(final_p)
                # print(final_p)

                # for idx, count in enumerate(final_p):
                #     inferred_goals_info[agent][_id][idx] = count

                # print('done with enumeration')
                # print(inferred_goals_info)

                sampling_count = 0
                # All path finding will be the same for each goal (so just do it once)
                all_best_paths = prev_env.generate_possible_paths(agent, prev_best_goals[agent][goal])
                while sampling_count != 50:

                    best_path = None
                    if all_best_paths != -1:
                        best_path = random.choice(all_best_paths)
                        best_path.append(prev_best_goals[agent][goal]['steps'][-1])
                    else:
                        best_path = -1
                    try:
                        if isinstance(best_path[0], int):
                            inferred_goals_info[agent][_id][best_path[0]] += 1

                    except TypeError:
                        # Case where best_path = -1
                        print(f'@observer_inference - TypeError')
                        print(f'Encountered best action to take, is not a movement.')

                        print(prev_best_goals[agent])
                        print(prev_best_goals[agent][goal])
                        print(prev_best_goals[agent][goal]['steps'][-1])
                        print(best_path)

                        non_movement_action = prev_best_goals[agent][goal]['steps'][-1]
                        if non_movement_action[0] == 'PICK':
                            inferred_goals_info[agent][_id][9] += 1
                        elif non_movement_action[0] == 'CHOP':
                            inferred_goals_info[agent][_id][10] += 1
                        elif non_movement_action[0] == 'COOK':
                            inferred_goals_info[agent][_id][11] += 1
                        elif non_movement_action[0] == 'SCOOP':
                            inferred_goals_info[agent][_id][12] += 1
                        elif non_movement_action[0] == 'SERVE':
                            inferred_goals_info[agent][_id][13] += 1
                        elif non_movement_action[0] == 'DROP':
                            inferred_goals_info[agent][_id][14] += 1
                        
                        # 'int' object is not subscriptable; -1 is returned
                        # inferred_goals_info[agent][_id][8] += 1
                    sampling_count += 1
                
                # Apply laplace smoothing
                inferred_goals_info[agent][_id] = self._laplace_smoothing(inferred_goals_info[agent][_id], 'action')
            
        print(f'Done with gathering samples')
        print(inferred_goals_info)
        inferred_goals_conditional_distribution = _get_conditional_distribution(inferred_goals_info)
        print(f'Done with deriving conditional distribution')
        print(inferred_goals_conditional_distribution)
        observer_task_to_not_do = self.observer_coordination_planning(inferred_goals_conditional_distribution, agents_reference_move)
        
        return observer_task_to_not_do

    def observer_coordination_planning(self, action_conditional_distribution, agents_reference_move):
        """
        PARAMETERS
        ----------
        action_conditional_distribution
            Make use of P(goal,t|action,t) to derive P(a,t+1|goal,t)
        agents_reference_move
            Information about action taken in previous timestep. To be used for identifying 
            which conditional distribution to use.

        TO-DO: Consider case where last action taken is an action eg 'PICK'
        """
        print(f'@observer_coordination_planning')
        print(self.world_state['historical_actions'])
        print(action_conditional_distribution)
        goal_probability_distribution = {}
        for agent in action_conditional_distribution:
            agent_id = agent.agent_id
            # agent_prev_action = agents_reference_move[agent]['prev_move']?
            agent_prev_action = self.world_state['historical_actions'][agent_id][-1]
            print('prev action')
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

        print('ok can')
        print(goal_probability_distribution)
        # Run one timestep for other agents from current world state
        print(f'Replicate current environment')
        from ipomdp.envs.overcooked_map_env import OvercookedEnv
        curr_env = OvercookedEnv()
        curr_env.world_state = copy.deepcopy(self.world_state)
        curr_best_goals = curr_env.find_agents_possible_goals()
        print('done with curr_best_goals - before smoothing')
        print(curr_best_goals)

        # Deep-Copy causes reference task id to change
        GOAL_SPACE = {
            goal.id: id(goal) for goal in curr_env.world_state['goal_space']
        }
        curr_best_goals = self._laplace_smoothing(curr_best_goals, 'tasks', GOAL_SPACE)
        print('done with curr_best_goals')
        print(curr_best_goals)

        # Perform goal weighting to select best next action to take
        # Deep-Copy causes reference agent id to change
        goal_probability_distribution = {agent.agent_id:val for agent, val in goal_probability_distribution.items()}
        curr_best_goals = {agent.agent_id:val for agent, val in curr_best_goals.items()}

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

        t_plus_1_agent_goal_rewards = {}
        for agent in goal_probability_distribution:
            t_plus_1_agent_goal_rewards[agent] = {
                k:goal_probability_distribution[agent][k]*curr_best_goals[agent][k]['rewards'] 
                for k,v in goal_probability_distribution[agent].items()
            }
        
        print('done with curr_best_goals')
        print(curr_best_goals)
        print('done with t_plus_1')
        print(t_plus_1_agent_goal_rewards)

        all_agents_best_inferred_goals = {}
        for agent in t_plus_1_agent_goal_rewards:
            # Randomly choose one if multiple goals have the same weighting
            best_goal_list = list()
            max_weighted_reward = max(t_plus_1_agent_goal_rewards[agent].items(), key=lambda x: x[1])[1]
            for task_id, weighted_reward in t_plus_1_agent_goal_rewards[agent].items():
                if weighted_reward == max_weighted_reward:
                    best_goal_list.append(task_id)
            
            all_agents_best_inferred_goals[agent] = random.choice(best_goal_list)
        print('random weight done')
        print(self.agent_id)
        print(all_agents_best_inferred_goals)
        print(goal_probability_distribution)
        print(curr_best_goals)

        observer_task_to_not_do = []
        agent_own_id = self.agent_id
        for agent in all_agents_best_inferred_goals:
            print('debug check')
            print(agent_own_id)
            print(curr_best_goals[agent_own_id])
            print(curr_best_goals[agent])
            print(curr_best_goals[agent][all_agents_best_inferred_goals[agent]])
            print(all_agents_best_inferred_goals[agent])
            if curr_best_goals[agent][all_agents_best_inferred_goals[agent]]['rewards'] > \
                curr_best_goals[agent_own_id][all_agents_best_inferred_goals[agent]]['rewards']:
                observer_task_to_not_do.append(all_agents_best_inferred_goals[agent])
        
        return observer_task_to_not_do

    def _laplace_smoothing(self, p_distribution, type, goal_space=None):
        """
        This function does +1 to all possible actions in the action space to the 
        action probability distribution.

        Parameters
        ----------
        p_distribution: Dict[int,int]
            Show current distribution after sampling
        """
        if type == 'action':
            ACTION_SPACE = 15 # up to 14 actions
            for i in range(ACTION_SPACE):
                p_distribution[i] += 1
        elif type == 'tasks':
            GOALS_SPACE = goal_space
            temp_p_distribution = defaultdict(dict)
            for agent in p_distribution:
                for task_id, _id in GOALS_SPACE.items():
                    if _id not in p_distribution[agent]:
                        # Already filtered out by agent.find_best_goal() function
                        # Require this to prevent key error when doing goal weighting
                        # p_distribution[agent][task_id] = {'rewards': 0}
                        temp_p_distribution[agent][task_id] = {'rewards': 0}
                    else:
                        temp_p_distribution[agent][task_id] = p_distribution[agent][_id]
            p_distribution = temp_p_distribution

        return p_distribution

    def _do_sampling(self, prev_env, prev_best_goals, agent, goal, _id):
        inferred_goals_info = [0]*15
        best_path = prev_env.generate_possible_paths(agent, prev_best_goals[agent][goal])

        if best_path != -1:
            best_path.append(prev_best_goals[agent][goal]['steps'][-1])

        try:
            if isinstance(best_path[0], int):
                inferred_goals_info[best_path[0]] += 1
            elif isinstance(best_path[0], list):
                inferred_goals_info[best_path[0][0]] += 1
        except TypeError:
            print(f'@observer_inference - TypeError')
            print(f'Encountered best action to take, is not a movement.')

            non_movement_action = prev_best_goals[agent][goal]['steps'][-1]
            if non_movement_action[0] == 'PICK':
                inferred_goals_info[9] += 1
            elif non_movement_action[0] == 'CHOP':
                inferred_goals_info[10] += 1
            elif non_movement_action[0] == 'COOK':
                inferred_goals_info[11] += 1
            elif non_movement_action[0] == 'SCOOP':
                inferred_goals_info[12] += 1
            elif non_movement_action[0] == 'SERVE':
                inferred_goals_info[13] += 1
            elif non_movement_action[0] == 'DROP':
                inferred_goals_info[14] += 1
        
        return inferred_goals_info


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
