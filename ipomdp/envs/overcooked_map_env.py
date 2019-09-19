from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.map_configs import *
from ipomdp.agents.agent_configs import *
from ipomdp.agents.base_agent import OvercookedAgent

class OvercookedEnv(MapEnv):
    def __init__(
        self,
        ascii_map=OVERCOOKED_MAP,
        num_agents: int=1,
        render=True
    ) -> None:
        super().__init__(ascii_map, num_agents, render)

    def custom_reset(self):
        """Initialize the """

    def setup_agents(self):
        print('inside setup')
        
        for agent in range(len(self.agent_initialization)):
            agent_id = agent + 1
            self.agents[agent_id] = OvercookedAgent(
                                    'agent_'+str(agent_id),
                                    self.agent_initialization[agent],
                                    BARRIERS,
                                    INGREDIENTS,
                                    RECIPES_COOKING_INTERMEDIATE_STATES,
                                    RECIPES_PLATING_INTERMEDIATE_STATES
                                )
            self.world_map[self.agent_initialization[agent]] = agent_id

def main() -> None:
    overcooked_env = OvercookedEnv(num_agents=2)
    print(overcooked_env.base_map)
    print(overcooked_env.world_map)
    print(overcooked_env.table_tops)
    print(overcooked_env.agents)
    overcooked_env.render('./frame.png')


if __name__ == "__main__":
    main()