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
        """
        agent_goal_costs = defaultdict(dict)

        for task_list in self.world_state['goal_space']:
            wanted_ingredient = [
                ingredient.location for ingredient in self.world_state['ingredients'] if \
                    (ingredient.name == task_list.ingredient and ingredient.state == task_list.head.state)]
            print(wanted_ingredient)
            # Append lowest cost goal into agent_goal_costs
            if task_list.head.task == 'pick':
                print('entered pick')
                # If no such ingredient exist
                if not wanted_ingredient:
                    # Just take fresh ones
                    print(self.world_state['ingredient_'+task_list.ingredient])
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [self.world_state['ingredient_'+task_list.ingredient]])
                else:
                    path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
            else:
                print('entered other act')
                print(task_list.head)
                # Guaranteed to have ingredient to slice/cook
                path_cost = self.calc_travel_cost(['ingredient_'+task_list.ingredient], [wanted_ingredient])
            print('done with this task calc')
            print(path_cost)
            agent_goal_costs[id(task_list)] = {
                'path': path_cost['ingredient_'+task_list.ingredient][0],
                'cost': path_cost['ingredient_'+task_list.ingredient][1]
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

    def pick(self, path: List[Tuple[int,int]], item: Item) -> None:
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
