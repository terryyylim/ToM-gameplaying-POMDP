import unittest

from unittests_configs import *
from overcooked import *
from base_agent import OvercookedAgent

class TestCases(unittest.TestCase):
    def setUp(self):
        
        self.agent = OvercookedAgent(
            'TestAgent',
            WORLD_STATE,
            BARRIERS,
            INGREDIENTS,
            RECIPES_COOKING_INTERMEDIATE_STATES,
            RECIPES_PLATING_INTERMEDIATE_STATES
        )

    def test_astar_search(self):
        self.assertEqual(self.agent.AStarSearch((9,1))[0], [(8, 6), (9, 5), (10, 4), (9, 3), (8, 2), (9, 1)])
        self.assertEqual(self.agent.AStarSearch((9,1))[1], 5)

    def test_pick_onion(self):
        start_coord = (8,6)
        end_coord = (3,7)
        path = [(8, 6), (7, 6), (6, 7), (5, 6), (4, 7), (3, 7)]
        onion = Ingredient('onion', 'unchopped', 'ingredient', is_raw=True, is_new=True)
        item_coord = [(3,8)]

        self.agent.pick(start_coord, end_coord, path, onion, item_coord)
        onion_item = [item for item in self.agent.world_state['ingredient_onion'] if id(item) == id(onion)][0]

        self.assertEqual(onion_item.is_new, False)
        self.assertEqual(self.agent.holding.name, 'onion')


if __name__ == "__main__":
    unittest.main()
