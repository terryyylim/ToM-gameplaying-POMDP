from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import numpy as np

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.map_configs import *
from ipomdp.agents.agent_configs import *
from ipomdp.agents.base_agent import OvercookedAgent
from ipomdp.overcooked import *

class OvercookedEnv(MapEnv):
    def __init__(
        self,
        ascii_map=OVERCOOKED_MAP,
        num_agents: int=1,
        render=True
    ) -> None:
        super().__init__(ascii_map, num_agents, render)
        self.initialize_world_state(ITEMS_INITIALIZATION)

    def custom_reset(self):
        """Initialize the map to original"""

    def custom_map_update(self):
        for agent in self.agents:
            self.agents[agent].world_state = self.world_state

    def initialize_world_state(self, items: Dict[str, List[Tuple]]):
        """ 
        world_state:
            a dictionary indicating world state (coordinates of items in map)
        """
        self.world_state['valid_cells'] = WORLD_STATE['valid_cells']

        for item in items:
            if item == 'chopping_boards':
                for i_state in items[item]:
                    new_item = ChoppingBoard('utensils', i_state)
                    self.world_state[item].append(new_item)
            elif item == 'extinguisher':
                for i_state in items[item]:
                    new_item = Extinguisher('safety', i_state)
                    self.world_state[item].append(new_item)
            elif item == 'plate':
                for i_state in items[item]:
                    new_item = Plate('utensils', i_state)
                    self.world_state[item].append(new_item)
            elif item == 'pot':
                for i_state in items[item]:
                    new_item = Pot('utensils', i_state)
                    self.world_state[item].append(new_item)
            elif item == 'stove':
                for i_state in items[item]:
                    new_item = Stove('utensils', i_state)
                    self.world_state[item].append(new_item)

    def setup_agents(self):
        for agent in range(len(self.agent_initialization)):
            agent_id = agent + 1
            self.agents[agent_id] = OvercookedAgent(
                                    'agent_'+str(agent_id),
                                    self.agent_initialization[agent],
                                    BARRIERS,
                                    INGREDIENTS,
                                    RECIPES_COOKING_INTERMEDIATE_STATES,
                                    RECIPES_PLATING_INTERMEDIATE_STATES,
                                )
            self.world_map[self.agent_initialization[agent]] = agent_id
            self.world_state['agents'].append(self.agents[agent_id])
        self.custom_map_update()

def main() -> None:
    overcooked_env = OvercookedEnv(num_agents=2)
    print(overcooked_env.base_map)
    print(overcooked_env.world_map)
    print(overcooked_env.table_tops)
    print(overcooked_env.agents)
    print('end')
    print(overcooked_env.world_state)
    print('updated state')
    print(overcooked_env.agents[1].world_state)
    overcooked_env.render('./frame.png')


if __name__ == "__main__":
    main()