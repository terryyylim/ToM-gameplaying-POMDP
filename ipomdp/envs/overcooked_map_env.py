from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import itertools
import logging
import numpy as np
import random

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.map_configs import *
from ipomdp.agents.agent_configs import *
from ipomdp.agents.base_agent import OvercookedAgent
from ipomdp.overcooked import *
from ipomdp.helpers import *

class OvercookedEnv(MapEnv):
    def __init__(
        self,
        ascii_map=OVERCOOKED_MAP,
        num_agents: int=1,
        render=True
    ) -> None:
        super().__init__(ascii_map, num_agents, render)
        self.initialize_world_state(ITEMS_INITIALIZATION, INGREDIENTS_INITIALIZATION)
        self.recipes = RECIPES
        self.recipes_ingredients_task = RECIPES_INGREDIENTS_TASK
        self.recipes_ingredients_count = RECIPES_INGREDIENTS_COUNT
        self.order_queue = []

        # Initialization: Update agent's current cell to be not available
        for agent in self.agents:
            self.world_state['valid_cells'].remove(self.agents[agent].location)

    def custom_reset(self):
        """Initialize the map to original"""

    def custom_map_update(self):
        for agent in self.agents:
            self.agents[agent].world_state = self.world_state

    def random_queue_order(self):
        new_order = random.choice(self.recipes)
        self.order_queue.append(new_order)
        self.initialize_task_list(new_order)

    def initialize_task_list(self, new_order: str):
        tasks = self.recipes_ingredients_task[new_order]
        tasks_count = self.recipes_ingredients_count[new_order]
        for ingredient in tasks:
            for _ in range(tasks_count[ingredient]):
                self.world_state['goal_space'].append(TaskList(new_order, tasks[ingredient], ingredient))

    def find_agents_possible_goals(self):
        agent_goals = {}
        for agent in self.world_state['agents']:
            agent_goals[agent] = agent.find_best_goal()
        return agent_goals

    def find_agents_best_goal(self):
        """
        Finds best action for each agent which maximizes utility, independent of other agents.
        
        Returns
        -------
        Example - for 2 agents
        assigned_best_goal = {
            <ipomdp.agents.base_agent.OvercookedAgent object at 0x133b46490>: [
                task_id,
                {'path': [4, ['p', 'new', (1, 3)]], 'cost': 1}
            ],
            <ipomdp.agents.base_agent.OvercookedAgent object at 0x133b46210>: [
                task_id,
                {'path': [0, 0, 4, 0, 0, ['p', 'new', (1, 3)]], 'cost': 5}
            ]
        }
        """
        print('@overcooked_map_env - find_agents_best_goal()')
        agents_possible_goals = self.find_agents_possible_goals()\

        assigned_best_goal = {}
        for agent in agents_possible_goals:
            tasks_rewards = [agents_possible_goals[agent][task]['rewards'] for task in agents_possible_goals[agent]]
            if tasks_rewards:
                max_task_rewards = max(tasks_rewards)
                max_rewards_task_idx = [idx for idx in range(len(tasks_rewards)) if tasks_rewards[idx] == max_task_rewards]

                # If more than one task with the same cost
                if len(max_rewards_task_idx) > 1:
                    assigned_task_idx = random.choice(max_rewards_task_idx)
                else:
                    # not random
                    assigned_task_idx = max_rewards_task_idx[0]
                assigned_task = list(agents_possible_goals[agent])[assigned_task_idx]
                assigned_best_goal[agent] = [assigned_task, agents_possible_goals[agent][assigned_task]]
            else:
                if agent.location in [(1,3), (1,8), (3,7), (3,5)]:
                    assigned_best_goal[agent] = [-1, {'steps': [0], 'rewards': -2}]
                else:
                    assigned_best_goal[agent] = [-1, {'steps': [8], 'rewards': -2}]
        return assigned_best_goal

    # def find_agents_best_goal(self):
    #     """
    #     Finds best action for each agent which maximizes utility.

    #     Return
    #     ------
    #     {<ipomdp.agents.base_agent.OvercookedAgent object at 0x1380daed0>: {'path': [(2, 4), (1, 3)], 'cost': 1},
    #     <ipomdp.agents.base_agent.OvercookedAgent object at 0x138a56910>: {'path': [(2, 8), (2, 7), (2, 6), (1, 5), (1, 4), (1, 3)], 'cost': 5}}
    #     """
    #     agents_possible_goals = self.find_agents_possible_goals()

    #     all_agents = []
    #     for agent in agents_possible_goals:
    #         agent_temp = []
    #         for goal in agents_possible_goals[agent]:
    #             agent_temp.append((agent, goal, agents_possible_goals[agent][goal]['cost']))
    #         all_agents.append(agent_temp)
    #     if len(agents_possible_goals) == 2:
    #         all_combi = list(itertools.product(all_agents[0], all_agents[1]))
    #     elif len(agents_possible_goals) == 3:
    #         all_combi = list(itertools.product(all_agents[0], all_agents[1], all_agents[2]))

    #     min_cost = float('inf')
    #     min_cost_idx = []
    #     for combi_idx in range(len(all_combi)):
    #         temp_cost = 0
    #         for combi_goal in all_combi[combi_idx]:
    #             temp_cost += combi_goal[2]
    #         if temp_cost == min_cost:
    #             min_cost_idx.append(combi_idx)
    #         if temp_cost < min_cost:
    #             min_cost = temp_cost
    #             min_cost_idx = [combi_idx]
    #     random_min_cost_idx = random.choice(min_cost_idx)
    #     temp_best_goals = all_combi[random_min_cost_idx]

    #     best_goals = {}
    #     for best_goal_info in temp_best_goals:
    #         print('print best goal info')
    #         print(best_goal_info)
    #         best_goals[best_goal_info[0]] = agents_possible_goals[best_goal_info[0]][best_goal_info[1]]
    #         best_goals[best_goal_info[0]]['task'] = best_goal_info[1]
    #     self.assign_agents(best_goals)

        # return best_goals

    def assign_agents(self, best_goals):
        """
        Set agent's assignment to True so that it doesnt search for a new goal.
        """
        for agent in best_goals:
            agent.is_assigned = True

    def initialize_world_state(self, items: Dict[str, List[Tuple]], ingredients: Dict[str, List[Tuple]]):
        """ 
        world_state:
            a dictionary indicating world state (coordinates of items in map)
        """
        self.world_state['valid_cells'] = WORLD_STATE['new_valid_cells']
        self.world_state['service_counter'] = WORLD_STATE['service_counter']

        for item in items:
            if item == 'chopping_board':
                for i_state in items[item]:
                    new_item = ChoppingBoard('utensils', i_state, 'empty')
                    self.world_state[item].append(new_item)
            elif item == 'extinguisher':
                for i_state in items[item]:
                    new_item = Extinguisher('safety', i_state)
                    self.world_state[item].append(new_item)
            elif item == 'plate':
                for i_state in items[item]:
                    new_item = Plate('utensils', i_state, 'empty')
                    self.world_state[item].append(new_item)
            elif item == 'pot':
                for i_state in items[item]:
                    new_item = Pot('utensils', i_state, 'empty')
                    self.world_state[item].append(new_item)
            elif item == 'stove':
                for i_state in items[item]:
                    new_item = Stove('utensils', i_state)
                    self.world_state[item].append(new_item)
        
        for ingredient in ingredients:
            self.world_state['ingredient_'+ingredient] = ingredients[ingredient]['location']

    def setup_agents(self):
        for agent in range(len(self.agent_initialization)):
            agent_id = agent + 1
            self.agents[agent_id] = OvercookedAgent(
                                    str(agent_id),
                                    self.agent_initialization[agent],
                                    BARRIERS,
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
    logger = helpers.get_logger()

    main()
