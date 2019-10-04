from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# import ray
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import random

from ipomdp.agents.agent_configs import *
from ipomdp.agents.astart_search import AStarGraph
from ipomdp.overcooked import *


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
        cooking_intermediate_states,
        plating_intermediate_states,
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
        self.is_assigned = is_assigned
        self.can_update = can_update
        self.goals = goals
        self.holding = holding
        self.actions = actions
        self.rewards = rewards
        self.cooking_intermediate_states = cooking_intermediate_states
        self.plating_intermediate_states = plating_intermediate_states
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
                    for valid_cell in valid_cells:
                        temp_item_instance = self.AStarSearch(valid_cell)
                        if not travel_costs[items[item_idx]]:
                            travel_costs[items[item_idx]] = (temp_item_instance[0], temp_item_instance[1], cur_item_instance)
                        else:
                            # Only replace if existing travel cost is greater (ensure only 1 path is returned given same cost)
                            if travel_costs[items[item_idx]][1] > temp_item_instance[1]:
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
        start = self.location
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

    def find_best_goal(self):
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

            path_actions = []
            # If task is to plate/scoop/serve, to check for dish, not ingredients
            if task_list.head.task == 'plate':
                print('@base_agent - Entered plate logic')
                if not isinstance(self.holding, Plate):
                    # Go to plate, Do picking
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
                        continue
                    except IndexError:
                        print('@base_agent - scoop/serve IndexError')
                        continue
                else:
                    if task_list.head.task == 'scoop':
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
                    if not wanted_ingredient:
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
                    else:
                        path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
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
                    # Does not guaranteed to have ingredient to slice/cook
                    print(f'@base_agent {id(self)} - Check for guaranteed presence of ingredient\n')

                    # Edge case: No available ingredient, this task is invalid
                    if not wanted_ingredient and not isinstance(self.holding, Ingredient):
                        continue
                    else:
                        # Get all paths + task, paths + task into [path_actions] array and returning
                        if task_list.head.task == 'slice':
                            print('@base_agent - Entered slice logic')

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
                                task_coord = self.world_state['pot'][0].location
                                end_coord = self.location
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
        for item_instance in item_coords:
            if (item_instance[0], item_instance[1]+1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]+1))
            elif (item_instance[0], item_instance[1]-1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]-1))
            elif (item_instance[0]-1, item_instance[1]) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0]-1, item_instance[1]))
            elif (item_instance[0]+1, item_instance[1]) in self.world_state['valid_cells']:
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


    def drop(self, path: List[Tuple[int,int]], item: Item, drop_coord: Tuple[int,int]) -> None:
        """
        This action assumes agent is currently holding an item.
        Prerequisite
        ------------
        - At grid with accessibility to dropping coord.
        Drop item <X>.
        """
        self.move(path)
        if type(item) == Ingredient:
            self.world_state['ingredient_'+item.name][item.id]['location'] = drop_coord
            self.holding = None

    def move(self, path: List[Tuple[int,int]]) -> None:
        """
        - Finds item in world state.
        - Go to grid with accessibility to item.

        Parameters
        ----------
        path: for animation on grid to happen
        """
        for step in path:
            # Do animation (sleep 0.5s?)
            self.world_state[self.agent_id] = step
        self.location = path[-1]

    def chop(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('base_agent@chop')
        print(task_coord)
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
            self.world_state['goal_space'].append(TaskList(dish, RECIPES_SERVE_TASK[dish], dish))
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

        dish_to_plate = [dish for dish in self.world_state['cooked_dish'] if dish.location == task_coord][0]
        # let the plate which agent is holding, hold the completed dish
        self.holding.dish = dish_to_plate

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
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        is_last = serve_info['is_last']
        
        # plate returns to return point (in clean form for now)
        self.holding.dish = None
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

def main():
    # ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
    
    oc_agent = OvercookedAgent(
        'Agent',
        (8,6),
        BARRIERS,
        INGREDIENTS,
        RECIPES_COOKING_INTERMEDIATE_STATES,
        RECIPES_PLATING_INTERMEDIATE_STATES)

    results = oc_agent.calc_travel_cost(['c_plates', 'e_boards'], [WORLD_STATE['c_plates'], WORLD_STATE['e_boards']])
    print(results)
    for subgoal in results:
        print("Goal -> ", subgoal)
        print("Route -> ", results[subgoal][0])
        print("Cost -> ", results[subgoal][1])

        plt.plot([v[0] for v in results[subgoal][0]], [v[1] for v in results[subgoal][0]])
        for barrier in [BARRIERS_1]:
            plt.plot([v[0] for v in barrier], [v[1] for v in barrier])
        plt.xlim(-1,13)
        plt.ylim(-1,9)
        plt.show()


if __name__ == '__main__':
    main()
