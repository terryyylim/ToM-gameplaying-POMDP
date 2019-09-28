from typing import Dict
from typing import List
from typing import Tuple

# import ray
from collections import defaultdict
import matplotlib.pyplot as plt

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
        ingredients,
        cooking_intermediate_states,
        plating_intermediate_states,
        goals=None,
    ) -> None:
        super().__init__(agent_id, location)
        self.world_state = {}
        self.goals = goals
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
 
        start = self.world_state['agent'][0]
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
        """
        agent_goal_costs = defaultdict(dict)

        for task_list in self.world_state['goal_space']:
            wanted_ingredient = [
                ingredient.location for ingredient in self.world_state['ingredients'] if \
                    (ingredient.name == task_list.ingredient and ingredient.state == task_list.head.state)]
            if task_list.head.task == 'pick':
                """
                For `pick` action, append to path actions consist of 
                'p': picking action
                'is_new': True/False - whether to take from box and create new Ingredient object or not
                'is_last': True/False - whether this action is last of everything - to update TaskList
                task_coord: coordinate to perform task on eg. pick ingredient at task_coord
                end_coord: coordinate where agent must be at before performing task at task_coord
                """
                print('@base_agent - Entered pick logic')
                # If no such ingredient exist
                if not wanted_ingredient:
                    # Just take fresh ones
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [self.world_state['ingredient_'+task_list.ingredient]])
                    task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                    end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                    path_actions = self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])
                    path_actions.append(['PICK', True, True, task_coord, end_coord])
                else:
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                    task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                    end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                    path_actions = self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])
                    path_actions.append(['PICK', False, True, task_coord, end_coord])
            else:
                # Guaranteed to have ingredient to slice/cook
                print('@base_agent - Check for guaranteed presence of ingredient')
                print(wanted_ingredient)

                # Get all paths + task, paths + task into [path_actions] array and returning
                if task_list.head.task == 'slice':
                    print('@base_agent - Entered slice logic')
                    # Go to ingredient, Do picking
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                    task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                    end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                    path_actions = self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])
                    path_actions.append(['PICK', False, False, task_coord, end_coord])

                    # Go to chopping board, Do slicing
                    chopping_board_cells = [chopping_board.location for chopping_board in self.world_state['chopping_board']]
                    chopping_path_cost = self.calc_travel_cost(['chopping_board'], [chopping_board_cells])
                    task_coord = chopping_path_cost['chopping_board'][2]
                    end_coord = chopping_path_cost['chopping_board'][0][-1]
                    chopping_path_actions = self.map_path_actions(chopping_path_cost['chopping_board'][0])
                    path_actions += chopping_path_actions
                    path_actions.append(['CHOP', True, task_coord, end_coord])
                elif task_list.head.task == 'cook':
                    print('@base_agent - Entered cook logic')
                    # Go to ingredient, Do picking
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
                    task_coord = path_cost['ingredient_'+task_list.ingredient][2]
                    end_coord = path_cost['ingredient_'+task_list.ingredient][0][-1]
                    path_actions = self.map_path_actions(path_cost['ingredient_'+task_list.ingredient][0])
                    path_actions.append(['PICK', False, False, task_coord, end_coord])

                    # Go to pot, Do cooking
                    pot_cells = [pot.location for pot in self.world_state['pot']]
                    cooking_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                    task_coord = cooking_path_cost['pot'][2]
                    end_coord = cooking_path_cost['pot'][0][-1]
                    cooking_path_actions = self.map_path_actions(cooking_path_cost['pot'][0])
                    path_actions += cooking_path_actions
                    path_actions.append(['COOK', True, task_coord, end_coord])
                elif task_list.head.task == 'serve':
                    print('@base_agent - Entered serve logic')
                    # Go to plate, Do picking
                    plate_board_cells = [plate.location for plate in self.world_state['plate']]
                    plate_path_cost = self.calc_travel_cost(['plate'], [plate_board_cells])
                    task_coord = cooking_path_cost['plate'][2]
                    end_coord = cooking_path_cost['plate'][0][-1]
                    path_actions = self.map_path_actions(plate_path_cost['plate'][0])
                    path_actions.append(['PICK', False, False, task_coord, end_coord])

                    # Go to pot, Do pouring
                    # Only works for case with 1 pot
                    pot_cells = [pot.location for pot in self.world_state['pot']]
                    collection_path_cost = self.calc_travel_cost(['pot'], [pot_cells])
                    task_coord = collection_path_cost['pot'][2]
                    end_coord = collection_path_cost['pot'][0][-1]
                    collection_path_actions = self.map_path_actions(collection_path_cost['pot'][0])
                    path_actions += collection_path_actions
                    path_actions.append(['COLLECT', True, task_coord, end_coord])

                    # Go to counter, Do serving
                    service_counter_cells = [service_counter.location for service_counter in self.world_state['service_counter']]
                    service_path_cost = self.calc_travel_cost(['service_counter'], [service_counter_cells])
                    task_coord = service_path_cost['service_counter'][2]
                    end_coord = service_path_cost['service_counter'][0][-1]
                    service_path_actions = self.map_path_actions(service_path_cost['service_counter'][0])
                    path_actions += service_path_actions
                    path_actions.append(['SERVE', True, task_coord, end_coord])

            total_rewards = 0
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

    def perform_best_goal(self, goal_to_complete: Dict[str, Any]):
        """
        Execute best_goal and wait for visualization run to finish.
        Params
        ------
        {'agent_1': { 'action': <TaskList object>, 'path': [], 'cost': 0 }}
        """
        print('executing goal')
        print(goal_to_complete)
        print(goal_to_complete['task'].head)
        # if agents_goal_to_complete[agent]['action'] == 'pick':
                
        # After performing goal (should this even be here?)
        # if not task_list.head.next and tasks_completion == 1:
        #     dish = task_list.dish
        #     ingredient = task_list.ingredient
        #     self.world_state['goal_space'].append(TaskList(dish, ['serve'], ingredient))
        #     self.world_state['goal_space'].remove(task_list)
        #     tasks_completion -= 1
        # elif not task_list.head.next:
        #     self.world_state['goal_space'].remove(task_list)
        #     tasks_completion -= 1
        # else:
        #     task_list.head = task_list.head.next

    def cook_ingredients(self):
        """
        Cooks dish and keeps track of timer.
        """
        # Timer(10, cook, [args]).start()
        return

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

    def pick(self, task_id: int, is_new: bool, is_last: bool, task_coord: Tuple[int, int]) -> None:
        """
        This action assumes agent has already done A* search and decided which goal state to achieve.
        Prerequisite
        ------------
        - Item object passed to argument should be item instance
        - At grid with accessibility to item [GO-TO]
        
        Returns
        -------
        Pick up item
        - Check what item is picked up [to determine if we need to update world state of item]

        TO-CONSIDER: 
        Do we need to check if agent is currently holding something?
        Do we need to set item coord to agent coord when the item is picked up?
        """
        self.move(path)
        if type(item) == Ingredient:
            if item.is_new:
                item.is_new = False
                self.world_state['ingredient_'+item.name].append(item) if item.is_raw else self.world_state['ingredient_'+item.name].append(item)
            self.holding = item


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

    def chop(self, ingredient: str):
        """
        Agent needs to check if ingredient is lying around somewhere in the map and get the nearest one.
        """
        # If there is ingredient lying around
        cur_ingredients = self.world_state['ingredient_'+ingredient]
        if cur_ingredients:
            valid_ingredients = [ingredient.location for ingredient in cur_ingredients if ingredient.state == 'unchopped']
            nearest_valid_ingredient_cost = self.calc_travel_cost(['ingredient_'+ingredient], [valid_ingredients])
            self.pick(nearest_valid_ingredient_cost['ingredient_'+ingredient][0], nearest_valid_ingredient_cost['ingredient_'+ingredient][2])
            # find nearest empty chopping board (do something to always ensure its empty?)
            # go nearest empty chopping board
            # start chopping (add lockdown to agent - to prevent movement ability?)

        # Else - go to an optimal spot and wait ?

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
