from typing import List
from typing import Tuple

# import ray
import matplotlib.pyplot as plt

from configs import *
from astart_search import AStarGraph


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
        goals=None,
    ) -> None:
        super().__init__(agent_id, world_state)
        self.goals = goals
        self.get_astar_map(barriers)

    def get_astar_map(self, barriers: List[List[Tuple[int,int]]]) -> None:
        self.astar_map = AStarGraph(barriers)

    def calc_subgoals_cost(self, subgoals: List[str]):
        subgoals_costs = {subgoal: self.AStarSearch(subgoal) for subgoal in subgoals}

        return subgoals_costs

    def AStarSearch(self, task):
        """
        A* Path-finding algorithm

        F: Estimated movement cost of start to end going via this position
        G: Actual movement cost to each position from the start position
        H: heuristic - estimated distance from the current node to the end node

        It is important for heuristic to always be an underestimation of the total path, as an overestimation
        will lead to A* searching through nodes that may not be the 'best' in terms of f value.
        """
 
        start = self.world_state['agent']
        end = self.world_state[task]
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
        Given goal and world state, to return best action in order to achieve it.

        (?) Perform action which maximizes difference between costs of agent & observer
        """
        return

    def cook_ingredients(self):
        """
        Cooks dish and keeps track of timer.
        """
        return

    def drop(self, item: str, coords: Tuple[int, int]) -> None:
        """
        Drop item <X>.
        """

    def move_to(self, start_coord: Tuple[int, int], end_coord: Tuple[int, int]) -> None:
        self.world_state[self.agent_id] = end_coord

def main():
    # ray.init(num_cpus=4, include_webui=False, ignore_reinit_error=True)
    
    oc_agent = OvercookedAgent('Agent', WORLD_STATE_1, BARRIERS)

    results = oc_agent.calc_subgoals_cost(['board_1'])
    for subgoal in results:
        print("Goal -> ", subgoal)
        print("Route -> ", results[subgoal][0])
        print("Cost -> ", results[subgoal][1])

        plt.plot([v[0] for v in results[subgoal][0]], [v[1] for v in results[subgoal][0]])
        for barrier in [BARRIERS]:
            plt.plot([v[0] for v in barrier], [v[1] for v in barrier])
        plt.xlim(-1,13)
        plt.ylim(-1,9)
        plt.show()


if __name__ == '__main__':
    main()
