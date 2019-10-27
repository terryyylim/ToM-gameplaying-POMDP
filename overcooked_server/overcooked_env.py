from typing import Dict
from typing import List
from typing import Tuple

from collections import defaultdict
import math
import itertools
import logging
import numpy as np
import random

from map_env import MapEnv
from agent import OvercookedAgent
from human_agent import HumanAgent
from overcooked_classes import *
from overcooked_game_settings import MAP_ACTIONS, RECIPES, RECIPES_INGREDIENTS_TASK
from agent_configs import ITEMS_INITIALIZATION, INGREDIENTS_INITIALIZATION, \
    RECIPES_INGREDIENTS_COUNT, BARRIERS

class OvercookedEnv(MapEnv):
    def __init__(
        self,
        human_agents=None,
        ai_agents=None
    ) -> None:
        super().__init__()
        self.initialize_world_state(ITEMS_INITIALIZATION, INGREDIENTS_INITIALIZATION)
        self.recipes = RECIPES
        self.recipes_ingredients_task = RECIPES_INGREDIENTS_TASK
        self.recipes_ingredients_count = RECIPES_INGREDIENTS_COUNT
        self.order_queue = []
        self.episode = 0
        self.human_agents = human_agents
        self.ai_agents = ai_agents
        self.setup_agents()
        self.random_queue_order()

        # Initialization: Update agent's current cell to be not available
        print(f'Removing agent current location from valid_cells, valid_item_cells list')
        try:
            for agent in self.agents:
                self.world_state['valid_cells'].remove(self.agents[agent].location)
                self.world_state['valid_item_cells'].remove(self.agents[agent].location)
        except ValueError:
            print('Valid cell is already updated')
    
    def update_episode(self):
        self.episode += 1
        queue_episodes = [70, 140, 210, 280, 350, 420]
        if self.episode in queue_episodes:
            self.random_queue_order()

    def custom_reset(self):
        """Initialize the map to original"""

    def custom_map_update(self):
        for agent in self.agents:
            print('in custom update')
            print(agent)
            print(type(self.agents[agent]))
            self.agents[agent].world_state = self.world_state
        
        temp_astar_map = None
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                temp_astar_map = agent.astar_map
        
        # Update agent locations into map barriers for A* Search
        for agent in self.world_state['agents']:
            temp_astar_map.barriers.append(agent.location)
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                agent.astar_map = temp_astar_map

    def random_queue_order(self):
        new_order = random.choice(self.recipes)
        self.order_queue.append(new_order)
        self.initialize_task_list(new_order)

    def initialize_task_list(self, new_order: str):
        tasks = self.recipes_ingredients_task[new_order]
        tasks_count = self.recipes_ingredients_count[new_order]
        for ingredient in tasks:
            for _ in range(tasks_count[ingredient]):
                self.world_state['task_id_count'] += 1
                self.world_state['goal_space'].append(TaskList(new_order, tasks[ingredient], ingredient, self.world_state['task_id_count']))
        
        self.world_state['task_id_mappings'] = {
            goal.id: id(goal) for goal in self.world_state['goal_space']
        }

    def find_agents_possible_goals(self, observers_task_to_not_do=[]):
        agent_goals = {}
        for agent in self.world_state['agents']:
            observer_task_to_not_do = []
            if isinstance(agent, OvercookedAgent):
                if observers_task_to_not_do:
                    print(f'Observer to not do tasks')
                    print(observers_task_to_not_do)
                    observer_task_to_not_do = observers_task_to_not_do[agent]
                agent_goals[agent] = agent.find_best_goal(observer_task_to_not_do)
            else:
                if isinstance(agent, HumanAgent):
                    print(f'Dummy Agent goals')
                    temp_OvercookedAgent = OvercookedAgent(
                        agent.id,
                        agent.location,
                        agent.barriers,
                        holding=agent.holding
                    )
                    temp_OvercookedAgent.world_state = self.world_state
                    print([(goal.head, goal.task, goal.ingredient) for goal in temp_OvercookedAgent.world_state['goal_space']])
                    agent_goals[agent] = temp_OvercookedAgent.find_best_goal([])
                    print(agent_goals[agent])
                    del temp_OvercookedAgent
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
        # Do inference here; skips inference for first timestep
        observers_task_to_not_do = {}
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                if agent.is_inference_agent and 'historical_world_state' in self.world_state:
                    print(f'Do inference for ToM agent')
                    observers_inference_tasks = agent.observer_inference()
                    observers_task_to_not_do[agent] = [self.world_state['task_id_mappings'][_id] for _id in observers_inference_tasks]
                else:
                    observers_task_to_not_do[agent] = []

        agents_possible_goals = self.find_agents_possible_goals(observers_task_to_not_do)
        print('agents possible goals')
        print(agents_possible_goals)

        assigned_best_goal = {}
        for agent in agents_possible_goals:
            tasks_rewards = [agents_possible_goals[agent][task]['rewards'] for task in agents_possible_goals[agent]]
            print('printing task rewards')
            print(tasks_rewards)

            if tasks_rewards:
                softmax_best_goal = self._softmax(agents_possible_goals[agent], beta=0.5)

                # # Greedy solution
                # max_task_rewards = max(tasks_rewards)
                # max_rewards_task_idx = [idx for idx in range(len(tasks_rewards)) if tasks_rewards[idx] == max_task_rewards]

                # # If more than one task with the same cost
                # if len(max_rewards_task_idx) > 1:
                #     assigned_task_idx = random.choice(max_rewards_task_idx)
                # else:
                #     # not random
                #     assigned_task_idx = max_rewards_task_idx[0]
                # assigned_task = list(agents_possible_goals[agent])[assigned_task_idx]
                # assigned_best_goal[agent] = [assigned_task, agents_possible_goals[agent][assigned_task]]

                print(f'Softmax Best Goal:')
                print(softmax_best_goal)
                all_best_paths = self.generate_possible_paths(agent, agents_possible_goals[agent][softmax_best_goal])

                # best_path == -1; means there's no valid permutations, use the original path
                if all_best_paths != -1:
                    best_path = random.choice(all_best_paths)
                    best_path.append(agents_possible_goals[agent][softmax_best_goal]['steps'][-1])
                    agents_possible_goals[agent][softmax_best_goal]['steps'] = best_path

                assigned_best_goal[agent] = [softmax_best_goal, agents_possible_goals[agent][softmax_best_goal]]
            else:
                # If no task at hand, but blocking stations, move to valid cell randomly
                # TO-DO: Fix case in stage 1 where agent is in cell (4,10) and blocks movements
                if tuple(agent.location) in [(1,3), (1,8), (3,7), (3,5)]:
                    print(f'Entered find random valid action')
                    random_valid_cell_move = self._find_random_valid_action(agent)
                    assigned_best_goal[agent] = [-1, {'steps': [random_valid_cell_move], 'rewards': -1}]
                else:
                    assigned_best_goal[agent] = [-1, {'steps': [8], 'rewards': -2}]
        return assigned_best_goal

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
        self.world_state['valid_cells'] = WORLD_STATE['valid_movement_cells']
        self.world_state['valid_item_cells'] = WORLD_STATE['temporary_valid_item_cells']
        self.world_state['service_counter'] = WORLD_STATE['service_counter']
        self.world_state['return_counter'] = WORLD_STATE['return_counter'][0]
        self.world_state['explicit_rewards'] = {'chop': 0, 'cook': 0, 'serve': 0}
        self.world_state['cooked_dish_count'] = {}

        for recipe in RECIPES:
            self.world_state['cooked_dish_count'][recipe] = 0

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
                plate_idx = 1
                for i_state in items[item]:
                    new_item = Plate(plate_idx, 'utensils', i_state, 'empty')
                    self.world_state[item].append(new_item)
                    plate_idx += 1
            elif item == 'pot':
                pot_idx = 1
                for i_state in items[item]:
                    new_item = Pot(pot_idx, 'utensils', i_state, 'empty')
                    self.world_state[item].append(new_item)
                    pot_idx += 1
            elif item == 'stove':
                for i_state in items[item]:
                    new_item = Stove('utensils', i_state)
                    self.world_state[item].append(new_item)
        
        for ingredient in ingredients:
            self.world_state['ingredient_'+ingredient] = ingredients[ingredient]['location']

    def setup_agents(self):
        if self.human_agents:
            for human_agent in self.human_agents:
                agent_id = human_agent
                coords = self.human_agents[agent_id]['coords']
                self.agents[agent_id] = HumanAgent(
                    agent_id,
                    coords
                )
                self.world_state['agents'].append(self.agents[agent_id])
        human_agent_count = len(self.human_agents) if self.human_agents else 0
        ai_agent_count = human_agent_count
        if self.ai_agents:
            for agent in self.ai_agents:
                ai_agent_count += 1
                is_ToM = self.ai_agents[agent]['ToM']
                coords = self.ai_agents[agent]['coords']
                agent_id = str(ai_agent_count)
                self.agents[agent_id] = OvercookedAgent(
                                        agent_id,
                                        coords,
                                        BARRIERS,
                                        is_inference_agent=is_ToM
                                    )
                self.world_state['agents'].append(self.agents[agent_id])
                
        self.custom_map_update()

    def _softmax(self, rewards_dict, beta:int=1):
        softmax_total = 0
        softmax_dict = defaultdict(int)
        for key in rewards_dict:
            reward = rewards_dict[key]['rewards']
            softmax_value = math.exp(beta*reward)
            softmax_dict[key] = softmax_value
            softmax_total += softmax_value
        softmax_dict = {k:v/softmax_total for k, v in softmax_dict.items()}

        max_softmax_val_arr = []
        max_softmax_val = max(softmax_dict.items(), key=lambda x: x[1])[1]
        for key, value in softmax_dict.items():
            if value == max_softmax_val:
                max_softmax_val_arr.append(key)
        print('After softmax calculation')
        print(softmax_dict)
        
        # Okay to do random.choice even for 1 best task
        return random.choice(max_softmax_val_arr)

    def generate_possible_paths(self, agent, best_goal):
        print(f'Generating best possible path with softmax')
        movement_count = 0
        cur_best_movements = []
        agent_end_idx = None

        for step in best_goal['steps']:
            if isinstance(step, int):
                movement_count += 1
                cur_best_movements.append(step)
            else:
                agent_end_idx = step[-1]
        
        # Currently all movements give reward of -1 (so don't need to check)
        print(f'Agent location:')
        print(agent.location)

        all_permutations = self._generate_permutations(cur_best_movements, agent, agent_end_idx)

        all_valid_paths = []
        for permutation in all_permutations:
            hit_obstacle = False

            temp_agent_location = list(agent.location).copy()
            for movement in range(len(permutation)):
                temp_agent_location = [sum(x) for x in zip(temp_agent_location, MAP_ACTIONS[permutation[movement]])]

                # Check for obstacle in path; and movement == 0
                if tuple(temp_agent_location) not in self.world_state['valid_cells'] and movement == 0:
                    hit_obstacle = True
                    continue
            
            # Append obstacle-free path
            if not hit_obstacle:
                all_valid_paths.append(
                    list(map(
                        lambda x: list(agent.actions.keys())[list(agent.actions.values()).index(x)],
                        permutation)
                    ))
        
        print(f'Done with all permutation mappings')
        if all_valid_paths:
            return all_valid_paths

        return -1
    
    def _find_random_valid_action(self, agent):
        action_space = [
            key for key, value in MAP_ACTIONS.items() \
                if key not in [
                    'STAY', 'MOVE_DIAGONAL_LEFT_UP', 'MOVE_DIAGONAL_RIGHT_UP',
                    'MOVE_DIAGONAL_LEFT_DOWN', 'MOVE_DIAGONAL_RIGHT_DOWN'
                    ]
                ]
        valid_random_cell_move = []

        agent_location = list(agent.location).copy()
        for movement in action_space:
            temp_agent_location = [sum(x) for x in zip(agent_location, MAP_ACTIONS[movement])]

            if tuple(temp_agent_location) in self.world_state['valid_cells']:
                valid_random_cell_move.append(
                    list(agent.actions.keys())[list(agent.actions.values()).index(movement)],
                )
        print(f'Found all possible random movements')
        print(valid_random_cell_move)

        return random.choice(valid_random_cell_move)

    # Not in use currently
    def _old_generate_permutations(self, path, agent):
        """
        Permutations based on the heuristics that a diagonal movement can be split into 2 separate movements
        Eg. MOVE_DIAGONAL_LEFT_UP -> MOVE_LEFT, MOVE_UP / MOVE_UP, MOVE_LEFT
        """
        all_permutations = []
        adjacent_movements = [
            'MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT'
        ]
        diagonal_movements = [
            'MOVE_DIAGONAL_LEFT_UP', 'MOVE_DIAGONAL_RIGHT_UP',
            'MOVE_DIAGONAL_LEFT_DOWN', 'MOVE_DIAGONAL_RIGHT_DOWN'
        ]
        path = list(map(
            lambda x: agent.actions[x],
            path)
        )
        all_permutations.append(path)
        idx_movement_mapping = [(idx, val) for idx, val in reversed(list(enumerate(path))) if val in diagonal_movements]

        flags = [False, True]
        flag_generator = list(itertools.product(flags, repeat=len(idx_movement_mapping)))
        
        # Need to loop twice left, up; up, left
        # Only fixes diagonal edge cases
        for flag in flag_generator:
            # Do replacement from the back to avoid index error
            flag = list(reversed(flag))
            for flag_idx in range(len(flag)):
                temp_path = path.copy()
                if flag[flag_idx] == True:
                    idx = idx_movement_mapping[flag_idx][0]
                    val = idx_movement_mapping[flag_idx][1]

                    if val == 'MOVE_DIAGONAL_LEFT_UP':
                        temp_path[idx: idx+1] = 'MOVE_UP', 'MOVE_LEFT'
                    elif val == 'MOVE_DIAGONAL_LEFT_DOWN':
                        temp_path[idx: idx+1] = 'MOVE_DOWN', 'MOVE_LEFT'
                    elif val == 'MOVE_DIAGONAL_RIGHT_UP':
                        temp_path[idx: idx+1] = 'MOVE_UP', 'MOVE_RIGHT'
                    elif val == 'MOVE_DIAGONAL_RIGHT_DOWN':
                        temp_path[idx: idx+1] = 'MOVE_DOWN', 'MOVE_RIGHT'

                all_permutations.append(temp_path)

            for flag_idx in range(len(flag)):
                temp_path = path.copy()
                if flag[flag_idx] == True:
                    idx = idx_movement_mapping[flag_idx][0]
                    val = idx_movement_mapping[flag_idx][1]

                    if val == 'MOVE_DIAGONAL_LEFT_UP':
                        temp_path[idx: idx+1] = 'MOVE_LEFT', 'MOVE_UP'
                    elif val == 'MOVE_DIAGONAL_LEFT_DOWN':
                        temp_path[idx: idx+1] = 'MOVE_LEFT', 'MOVE_DOWN'
                    elif val == 'MOVE_DIAGONAL_RIGHT_UP':
                        temp_path[idx: idx+1] = 'MOVE_RIGHT', 'MOVE_UP'
                    elif val == 'MOVE_DIAGONAL_RIGHT_DOWN':
                        temp_path[idx: idx+1] = 'MOVE_RIGHT', 'MOVE_DOWN'

                all_permutations.append(temp_path)

        # Need to fix for adjacent edge cases; eg. MOVE_DOWN, MOVE_DIAGONAL_RIGHT_DOWN, but down is blocked
        # Mini-hack to allow swapping & combining first 2 terms
        if len(path) > 1:
            if path[0] in adjacent_movements and path[1] in diagonal_movements:
                print(f'Adjacent and Diagonal movements swap!')
                temp_path = path.copy()
                temp_adj_move = path[0]
                temp_path[0] = path[1]
                temp_path[1] = temp_adj_move

                all_permutations.append(temp_path)

            first_second_combi = [path[0], path[1]]
            temp_path = path.copy()
            if set(first_second_combi) == set(['MOVE_UP', 'MOVE_LEFT']):
                temp_path.pop(0)
                temp_path.pop(0)
                temp_path.insert(0, 'MOVE_DIAGONAL_LEFT_UP')
            elif set(first_second_combi) == set(['MOVE_UP', 'MOVE_RIGHT']):
                temp_path.pop(0)
                temp_path.pop(0)
                temp_path.insert(0, 'MOVE_DIAGONAL_RIGHT_DOWN')
            elif set(first_second_combi) == set(['MOVE_DOWN', 'MOVE_LEFT']):
                temp_path.pop(0)
                temp_path.pop(0)
                temp_path.insert(0, 'MOVE_DIAGONAL_LEFT_DOWN')
            elif set(first_second_combi) == set(['MOVE_DOWN', 'MOVE_LEFT']):
                temp_path.pop(0)
                temp_path.pop(0)
                temp_path.insert(0, 'MOVE_DIAGONAL_LEFT_UP')
            all_permutations.append(temp_path)

        permutations_set = list(set(tuple(perm) for perm in all_permutations))

        return list(permutations_set)

    def _generate_permutations(self, path, agent, agent_end_idx):
        """
        Permutations based on the heuristics that a diagonal movement can be split into 2 separate movements
        Eg. MOVE_DIAGONAL_LEFT_UP -> MOVE_LEFT, MOVE_UP / MOVE_UP, MOVE_LEFT

        First step determines action space
        ----------------------------------
        MOVE_LEFT -> MOVE_DIAGONAL_LEFT_UP, MOVE_DIAGONAL_LEFT_DOWN, MOVE_LEFT
        MOVE_RIGHT -> MOVE_DIAGONAL_RIGHT_UP, MOVE_DIAGONAL_RIGHT_DOWN, MOVE_RIGHT
        MOVE_UP -> MOVE_DIAGONAL_LEFT_UP, MOVE_DIAGONAL_RIGHT_UP, MOVE_UP
        MOVE_DOWN -> MOVE_DIAGONAL_LEFT_DOWN, MOVE_DIAGONAL_RIGHT_DOWN, MOVE_DOWN
        MOVE_DIAGONAL_LEFT_UP -> MOVE_LEFT, MOVE_UP, MOVE_DIAGONAL_LEFT_UP
        MOVE_DIAGONAL_RIGHT_UP -> MOVE_RIGHT, MOVE_UP, MOVE_DIAGONAL_RIGHT_UP
        MOVE_DIAGONAL_LEFT_DOWN -> MOVE_LEFT, MOVE_DOWN, MOVE_DIAGONAL_LEFT_DOWN
        MOVE_DIAGONAL_RIGHT_DOWN -> MOVE_RIGHT, MOVE_DOWN, MOVE_DIAGONAL_RIGHT_DOWN
        """
        heuristic_mapping = {
            'MOVE_LEFT': ['MOVE_DIAGONAL_LEFT_UP', 'MOVE_DIAGONAL_LEFT_DOWN', 'MOVE_LEFT'],
            'MOVE_RIGHT': ['MOVE_DIAGONAL_RIGHT_UP', 'MOVE_DIAGONAL_RIGHT_DOWN', 'MOVE_RIGHT'],
            'MOVE_UP': ['MOVE_DIAGONAL_LEFT_UP', 'MOVE_DIAGONAL_RIGHT_UP', 'MOVE_UP'],
            'MOVE_DOWN': ['MOVE_DIAGONAL_LEFT_DOWN', 'MOVE_DIAGONAL_RIGHT_DOWN', 'MOVE_DOWN'],
            'MOVE_DIAGONAL_LEFT_UP': ['MOVE_LEFT', 'MOVE_UP', 'MOVE_DIAGONAL_LEFT_UP'],
            'MOVE_DIAGONAL_RIGHT_UP': ['MOVE_RIGHT', 'MOVE_UP', 'MOVE_DIAGONAL_RIGHT_UP'],
            'MOVE_DIAGONAL_LEFT_DOWN': ['MOVE_LEFT', 'MOVE_DOWN', 'MOVE_DIAGONAL_LEFT_DOWN'],
            'MOVE_DIAGONAL_RIGHT_DOWN': ['MOVE_RIGHT', 'MOVE_DOWN', 'MOVE_DIAGONAL_RIGHT_DOWN']
        }
        path = list(map(
            lambda x: agent.actions[x],
            path)
        )

        # Determine best reward from path
        best_reward = sum([agent.rewards[action] for action in path])
        print(f'Best reward: {best_reward}')
        valid_permutations = []

        def permutations_dp(
            agent,
            best_reward:int,
            end_location,
            cur_action_chain,
            orig_action_chain,
            heuristic_mapping,
            dp_table,
            valid_cells
        ):
            # weird np.array error
            valid_cells = [tuple(cell) for cell in valid_cells]
            # still have actions left to take
            cur_step = len(orig_action_chain) - len(cur_action_chain)

            if not cur_action_chain:
                return
            
            # Next action to take
            action = cur_action_chain[0]
            # print(f'Current action is: {action}')
            # Returns list of next possible actions to take
            heuristic_map = heuristic_mapping[action]

            # TO-DO: Add in checks if end_location for current iteration is a valid cell
            if cur_step == 0:
                cur_reward = 0
                # print(f'Entered cur_step == 0')

                for heuristic_action in heuristic_map:
                    heuristic_reward = agent.rewards[heuristic_action]
                    new_reward = cur_reward + heuristic_reward
                    new_location = [sum(x) for x in zip(list(agent.location), MAP_ACTIONS[heuristic_action])]
                    new_path = [heuristic_action]

                    # If its a DIAGONAL movement
                    if 'DIAGONAL' in action and 'DIAGONAL' not in heuristic_action:
                        second_action = heuristic_second_action(action, heuristic_action)
                        second_reward = agent.rewards[second_action]
                        new_reward += second_reward
                        new_location = [sum(x) for x in zip(new_location, MAP_ACTIONS[second_action])]
                        new_path.append(second_action)

                    # If still possible to find best reward and not ending in invalid cell
                    if (new_reward > best_reward) and (tuple(new_location) in valid_cells):
                        dp_table[cur_step].append(
                            {
                                'cur_reward': new_reward,
                                'cur_path': new_path,
                                'cur_location': new_location
                            }
                        )
                    
                    if (new_reward == best_reward) and (new_location == list(end_location)):
                        valid_permutations.append(new_path)
            else:
                # print(f'Entered cur_step != 0')
                # Returns valid existing path by far; List of Tuple (cost:int, steps:List[int])
                cur_valid_paths = dp_table[cur_step-1]

                for path_info in cur_valid_paths:
                    cur_reward = path_info['cur_reward']
                    cur_location = path_info['cur_location']
                    cur_path = path_info['cur_path']

                    for heuristic_action in heuristic_map:
                        heuristic_reward = agent.rewards[heuristic_action]
                        new_reward = cur_reward + heuristic_reward
                        new_location = [sum(x) for x in zip(cur_location, MAP_ACTIONS[heuristic_action])]
                        new_path = cur_path.copy()
                        new_path.append(heuristic_action)

                        if (new_reward > best_reward) and (tuple(new_location) in valid_cells):
                            dp_table[cur_step].append(
                                {
                                    'cur_reward': new_reward,
                                    'cur_path': new_path,
                                    'cur_location': new_location
                                }
                            )
                        
                        if (new_reward == best_reward) and (new_location == list(end_location)):
                            valid_permutations.append(new_path)
            
            # Remove completed action
            cur_action_chain.pop(0)
            permutations_dp(
                agent, best_reward, end_location, cur_action_chain,
                orig_action_chain, heuristic_mapping, dp_table, valid_cells
            )

        # Use heuristics to reduce action space
        valid_cells = self.world_state['valid_cells'].copy()
        locs = [agent.location for agent in self.world_state['agents']]
        for loc in locs:
            valid_cells.append(loc)
        permutations_dp(
            agent, best_reward, agent_end_idx, path.copy(), path.copy(),
            heuristic_mapping, defaultdict(list), valid_cells
        )
        print(f'Done with generating valid permutations')

        return valid_permutations
    
def heuristic_second_action(diagonal_action, taken_adj_action):
    heuristic_action_mapping_alt = {
        'MOVE_DIAGONAL_LEFT_UP': {
            'MOVE_LEFT': 'MOVE_UP',
            'MOVE_UP': 'MOVE_LEFT'
        },
        'MOVE_DIAGONAL_RIGHT_UP': {
            'MOVE_RIGHT': 'MOVE_UP',
            'MOVE_UP': 'MOVE_RIGHT'
        },
        'MOVE_DIAGONAL_LEFT_DOWN': {
            'MOVE_LEFT': 'MOVE_DOWN',
            'MOVE_DOWN': 'MOVE_LEFT'
        },
        'MOVE_DIAGONAL_RIGHT_DOWN': {
            'MOVE_RIGHT': 'MOVE_DOWN',
            'MOVE_DOWN': 'MOVE_RIGHT'
        }
    }
    return heuristic_action_mapping_alt[diagonal_action][taken_adj_action]
