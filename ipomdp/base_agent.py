from typing import Dict
from typing import List
from typing import Tuple

# import ray
from collections import defaultdict
import matplotlib.pyplot as plt

from configs import *
from astart_search import AStarGraph
from overcooked import *


class BaseAgent:
    def __init__(
        self,
        agent_id,
        world_state
    ) -> None:
        """
        Parameters
        ----------
        agent_id: str
            a unique id allowing the map to identify the agents
        world_state:
            a dictionary indicating world state (coordinates of items in map)
        """

        self.agent_id = agent_id
        self.world_state = world_state

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
        world_state,
        barriers,
        ingredients,
        cooking_intermediate_states,
        plating_intermediate_states,
        goals=None,
    ) -> None:
        super().__init__(agent_id, world_state)
        self.goals = goals
        self.cooking_intermediate_states = cooking_intermediate_states
        self.plating_intermediate_states = plating_intermediate_states
        self.get_astar_map(barriers)
        self.update_world_state_with_ingredients(ingredients)

    def get_astar_map(self, barriers: List[List[Tuple[int,int]]]) -> None:
        self.astar_map = AStarGraph(barriers)

    def update_world_state_with_ingredients(self, ingredients: Dict[str, List[str]]):
        for ingredient in ingredients:
            self.world_state['ingredient_'+ingredient] = []
    
    def update_world_state_with_pots(self, pot_coords: List[Tuple[int,int]]):
        for pot_num in range(len(pot_coords)):
            self.world_state['pot_'+str(pot_num)] = Pot('utensil', pot_coords[pot_num])

    def calc_travel_cost(self, items: List[str]):
        # get valid cells for each goal
        item_valid_cell_states = defaultdict(list)
        for item in items:
            item_valid_cell_states[item] = self.find_valid_cell(item)

        travel_costs = defaultdict(tuple)
        for item in items:
            cur_item_instances = self.world_state[item]
            for cur_item_instance in cur_item_instances:
                try:
                    valid_cells = item_valid_cell_states[item][cur_item_instance]
                    for valid_cell in valid_cells:
                        temp_item_instance = self.AStarSearch(valid_cell)
                        if not travel_costs[item]:
                            travel_costs[item] = temp_item_instance
                        else:
                            if travel_costs[item][1] > temp_item_instance[1]:
                                travel_costs[item] = temp_item_instance
                            continue
                except KeyError:
                    raise KeyError('No valid path to get to item!')

        return travel_costs

    def AStarSearch(self, dest_coords: Tuple[int,int]):
        """
        A* Path-finding algorithm

        F: Estimated movement cost of start to end going via this position
        G: Actual movement cost to each position from the start position
        H: heuristic - estimated distance from the current node to the end node

        It is important for heuristic to always be an underestimation of the total path, as an overestimation
        will lead to A* searching through nodes that may not be the 'best' in terms of f value.
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

    # def generate_goals_as_dags(self, goal):
    #     """
    #     Given new goals, convert goals to subgoals in DAG structure.
    #     """
    #     goal_required_ws = 
    #     dag = Graph(directed=True)
    #     print(dag)


    def find_best_goal(self):
        """
        Finds action which maximizes utility given world state from possible action space
        Returns
        -------
        action: Action that maximizes utility given world state -> str
        """
        return

    def cook_ingredients(self):
        """
        Cooks dish and keeps track of timer.
        """
        # Timer(10, cook, [args]).start()
        return

    def find_valid_cell(self, item: str) -> Tuple[int,int]:
        """
        Items can only be accessible from Up-Down-Left-Right of item cell.
        Get all cells agent can step on to access item.

        Returns
        -------
        all_valid_cells: Dict[str,List[Tuple[int,int]]]
        """
        all_valid_cells = defaultdict(list)
        # item_instance is Tuple[int,int]
        for item_instance in self.world_state[item]:
            if (item_instance[0], item_instance[1]+1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]+1))
            elif (item_instance[0], item_instance[1]-1) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0], item_instance[1]-1))
            elif (item_instance[0]-1, item_instance[1]) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0]-1, item_instance[1]))
            elif (item_instance[0]+1, item_instance[1]) in self.world_state['valid_cells']:
                all_valid_cells[item_instance].append((item_instance[0]+1, item_instance[1]))

        return all_valid_cells

    def pick(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int], path: List[Tuple[int,int]], item: Item, item_coord: Tuple[int,int]) -> None:
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
        self.move(start_coord, end_coord, path)
        if type(item) == Ingredient:
            if item.is_new:
                item.is_new = False
                self.world_state['ingredient_'+item.name].append(item) if item.is_raw else self.world_state['ingredient_'+item.name].append(item)
            self.holding = item

    def drop(self, item: str, coords: Tuple[int, int]) -> None:
        """
        Drop item <X>.
        """
        self.world_state[item] = coords

    def move(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int], path: List[Tuple[int,int]]) -> None:
        """
        - Finds item in world state.
        - Go to grid with accessibility to item.

        Parameters
        ----------
        path: for animation on grid to happen
        """
        self.world_state[self.agent_id] = end_coord

def main():
    # ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
    
    oc_agent = OvercookedAgent(
        'Agent',
        WORLD_STATE,
        BARRIERS,
        INGREDIENTS,
        RECIPES_COOKING_INTERMEDIATE_STATES,
        RECIPES_PLATING_INTERMEDIATE_STATES)

    results = oc_agent.calc_travel_cost(['c_plates'])
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
