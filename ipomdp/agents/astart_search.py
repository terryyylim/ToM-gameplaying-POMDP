from typing import List
from typing import Tuple

class AStarGraph():
	# Define a class board like grid with two barriers
    def __init__(
        self,
        barriers: List[Tuple[int, int]]
    ) -> None:
        self.barriers = [barriers]
        
    def heuristic(self, start, goal):
		# Use Chebyshev distance heuristic if we can move one square either
		# adjacent or diagonal
        D = 1
        D2 = 1
        dx = abs(start[0] - goal[0])
        dy = abs(start[1] - goal[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
 
    def get_vertex_neighbours(self, pos):
        n = []
		# Moves allow link a chess king
        # prevent diagonals ,(1,1),(-1,1),(1,-1),(-1,-1)
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1)]:
            x2 = pos[0] + dx
            y2 = pos[1] + dy
            if x2 < 0 or x2 > 8 or y2 < 0 or y2 > 12:
                continue
            n.append((x2, y2))
        return n
 
    def move_cost(self, a, b):
        for barrier in self.barriers:
            if b in barrier:
                return 10000 # Extremely high cost to enter barrier squares
            return 1 # Normal movement cost
