from typing import Dict
from typing import List
from typing import Tuple

import collections
import numpy as np
from ray.rllib.env import MultiAgentEnv

from map_configs import *
from agent_configs import *
from agent import *
from overcooked_classes import *

class MapEnv(MultiAgentEnv):
    def __init__(
        self,
        ascii_map: List[str],
        num_agents: int=1,
        render=True,
        color_map: Dict[str, List[int]]=DEFAULT_COLOURS,
        agent_initialization: List[Tuple[int,int]]=AGENTS_INITIALIZATION
    ) -> None:
        """
        Parameters
            ----------
            ascii_map: list of strings
                Specify what the map should look like. Look at constant.py for
                further explanation
            num_agents: int
                Number of agents to have in the system.
            render: bool
                Whether to render the environment
            color_map: dict
                Specifies how to convert between ascii chars and colors
        """
        self.task_id_count = 0
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.world_map = self.base_map
        self.world_state = defaultdict(list)
        self.world_state['task_id_count'] = 0
        self.world_state['historical_actions'] = collections.defaultdict(list)
        self.agent_initialization = agent_initialization

        self.num_agents = len(agent_initialization)
        
        self.agents = {}

        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS

        self.table_tops = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == '@':
                    self.table_tops.append([row, col])
        print('got here')
        # self.setup_agents()

    def custom_reset(self):
        """Reset custom elements of the map. For example, spawn table tops and items"""
        pass

    def custom_action(self, agent, action):
        """Execute any custom actions that may be defined, like pick or chop
        Parameters
        ----------
        agent: agent that is taking the action
        action: key of the action to be taken
        Returns
        -------
        updates: list(list(row, col, char))
            List of cells to place onto the map
        """
        pass

    def custom_map_update(self):
        """Custom map updates that don't have to do with agent actions"""
        pass

    def setup_agents(self):
        """Construct all the agents for the environment"""
        raise NotImplementedError

    def ascii_to_numpy(self, ascii_list: List[str]):
        """
        Converts a list of strings into a numpy array

        Parameters
        ----------
        ascii_list: list of strings
            List describing what the map should look like
        Returns
        -------
        arr: np.ndarray
            numpy array describing the map with ' ' indicating an empty space
        """

        arr = np.full((len(ascii_list), len(ascii_list[0])), ' ')
        for row in range(arr.shape[0]):
            for col in range(arr.shape[1]):
                arr[row, col] = ascii_list[row][col]
        return arr

    # Undone
    def step(self, agent_actions):
        """Takes in a dict of actions and converts them to a map update
        Parameters
        ----------
        actions: dict {agent: action}
            The agent interprets the action ([int - move action]/[list - explains task action])
            and converts it to a command.
        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym

        QUESTIONS TO CONSIDER:
        1. What to return for observations? since we already have world_state
        """
        print('@map_env - step()')
        print(agent_actions)
        
        orig_pos = {agent:tuple(agent.location) for agent in self.world_state['agents']}
        orig_holding = {agent:agent.holding for agent in self.world_state['agents']}

        self.update_moves(agent_actions)

        curr_pos = {agent: tuple(agent.location) for agent in self.world_state['agents']}
        
        # Update map barriers for agent's A* Search
        temp_astar_map = None
        # Get A* Search map
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                temp_astar_map = agent.astar_map
        
        for agent in curr_pos:
            if curr_pos[agent] != orig_pos[agent]:
                self.world_state['valid_cells'].append(orig_pos[agent])
                self.world_state['valid_cells'].remove(curr_pos[agent])

                # Update barriers in map used for A* Search
                temp_astar_map.barriers.remove(orig_pos[agent])
                temp_astar_map.barriers.append(curr_pos[agent])

                # Edge case: Prevent converting to ' ' if another agent is currently on it
                # all_agent_location = [tuple(agent.location) for agent in self.world_state['agents']]

                # if orig_pos[agent] not in all_agent_location:
                #     self.world_map[orig_pos[agent][0], orig_pos[agent][1]] = ' '
                # self.world_map[curr_pos[agent][0], curr_pos[agent][1]] = agent.agent_id
        
        for agent in agent_actions:
            action = agent_actions[agent][1]

            if isinstance(action, list):
                # Index 0: action_type; Index 1: action_info
                if action[0] == 'PICK':
                    if action[1]['pick_type'] == 'plate':
                        if agent.holding:
                            # Edge-Case: AgentL about to serve, AgentR trying to pick plate for the same goal that is going to be removed
                            # Continue and not pick instead
                            cur_plate_pos = action[1]['task_coord']
                            self.world_state['valid_item_cells'].append(cur_plate_pos)

                            # if tuple(cur_plate_pos) == self.world_state['return_counter']:
                            #     # TO-DO: If more than 1 plates returned here, then...colour? Because stackable
                            #     all_plate_locations = [plate.location for plate in self.world_state['plate']]
                            #     if len(set(all_plate_locations)) == 1:
                            #         # All plates are on return counter
                            #         self.world_map[cur_plate_pos[0], cur_plate_pos[1]] = 'P'
                            #     else:
                            #         self.world_map[cur_plate_pos[0], cur_plate_pos[1]] = 'D'
                            # elif tuple(cur_plate_pos) in BARRIERS:
                            #     self.world_map[cur_plate_pos[0], cur_plate_pos[1]] = '@'
                            # else:
                            #     self.world_map[cur_plate_pos[0], cur_plate_pos[1]] = ' '
                    elif action[1]['pick_type'] == 'ingredient':
                        if agent.holding:
                            all_raw_chop_locations = [cb.location for cb in self.world_state['chopping_board']]
                            all_raw_ingredients_locations = [self.world_state['ingredient_'+agent.holding.name][0]]
                            
                            cur_ingredient_pos = action[1]['task_coord']
                            if cur_ingredient_pos not in all_raw_chop_locations and cur_ingredient_pos not in all_raw_ingredients_locations: 
                                self.world_state['valid_item_cells'].append(cur_ingredient_pos)

                            # if tuple(cur_ingredient_pos) in BARRIERS:
                            #     all_raw_chop_locations = [cb.location for cb in self.world_state['chopping_board']]
                            #     all_raw_ingredients_locations = [self.world_state['ingredient_'+agent.holding.name][0]]
                            #     if tuple(cur_ingredient_pos) in all_raw_chop_locations:
                            #         self.world_map[cur_ingredient_pos[0], cur_ingredient_pos[1]] = 'C'
                            #     elif tuple(cur_ingredient_pos) in all_raw_ingredients_locations:
                            #         if agent.holding.name == 'onion':
                            #             self.world_map[cur_ingredient_pos[0], cur_ingredient_pos[1]] = 'O'
                            #     else:
                            #         self.world_map[cur_ingredient_pos[0], cur_ingredient_pos[1]] = '@'

                # If is drop, make 2 colour shade
                if action[0] == 'DROP':
                    if type(orig_holding[agent]) == Plate:
                        cur_plate_pos = orig_holding[agent].location

                        # Edge case: Randomly chosen spot to drop item must be a table-top cell
                        if cur_plate_pos in self.world_state['valid_item_cells']:
                            self.world_state['valid_item_cells'].remove(cur_plate_pos)
                        # self.world_map[cur_plate_pos[0], cur_plate_pos[1]] = 'P'
                    elif type(orig_holding[agent]) == Ingredient:
                        cur_ingredient_pos = orig_holding[agent].location

                        # Edge case: Randomly chosen spot to drop item must be a table-top cell
                        if cur_ingredient_pos in self.world_state['valid_item_cells']:
                            self.world_state['valid_item_cells'].remove(cur_ingredient_pos)

                        # Different colours based on ingredient state
                #         if orig_holding[agent].state == 'unchopped':
                #             self.world_map[cur_ingredient_pos[0], cur_ingredient_pos[1]] = 'Z'
                #         elif orig_holding[agent].state == 'chopped':
                #             self.world_map[cur_ingredient_pos[0], cur_ingredient_pos[1]] = 'X'

                # if action[0] == 'SERVE':
                #     if type(orig_holding[agent]) == Plate:
                #         self.world_map[self.world_state['return_counter'][0], self.world_state['return_counter'][1]] = 'P'

        # Update A* Search map for all agents
        for agent in self.world_state['agents']:
            if isinstance(agent, OvercookedAgent):
                agent.astar_map = temp_astar_map

        print('after 1 cycle')
        print(self.world_state['valid_cells'])
        # TO-DO: Add mechanism to store past observations, rewards

    # Taking only 1 grid cell movement now (correct?)
    def update_moves(self, actions):
        """
        #Converts agent action tuples into a new map and new agent positions.
        #Also resolves conflicts over multiple agents wanting a cell.
        """
        print('@map_env - update_moves()')
        # fix for np.array weird bug
        print(actions)
        for agent in actions:
            agent.location = tuple(agent.location)

        # Stores non-grid cells movement (if any)
        agent_tasks = {}
        agent_actions = {}

        for agent, task_action in actions.items():
            print(task_action)
            task = task_action[0]
            action = task_action[1]
            if type(action) == int:
                # Case: Just movements
                print('map_env@update_moves - Movement found')
                agent_action = agent.action_map(action)
                agent_actions[agent] = agent_action
            else:
                # Case: Pick/Chop/Cook/Plate/Scoop/Serve actions
                print('map_env@update_moves - Action found')
                agent_tasks[agent] = [task, action]
                agent_action = agent.action_map(8)
                agent_actions[agent] = agent_action

        reserved_slots = []
        agent_moves = {}
        for agent, action in agent_actions.items():
            selected_action = MAP_ACTIONS[action]
            
            new_pos = tuple([x + y for x, y in zip(list(agent.location), selected_action)])
            new_pos = agent.return_valid_pos(new_pos)
            agent_moves[agent] = new_pos
            reserved_slots.append((new_pos, agent))

        agent_by_pos = {tuple(agent.location): agent for agent in self.world_state['agents']}

        # list of moves and their corresponding agents
        move_slots = [slot[0] for slot in reserved_slots]
        agent_to_slot = [slot[1] for slot in reserved_slots]

        print('@map_env - Starting moves (if any)')
        print(move_slots)
        print(agent_to_slot)
        print(agent_by_pos)
        print(agent_moves)

        # cut short computation if there are no moves (to be used if we consider rotation)
        if len(agent_to_slot) > 0:
            # shuffle so that a random agent has slot priority
            shuffle_list = list(zip(agent_to_slot, move_slots))
            np.random.shuffle(shuffle_list)
            agent_to_slot, move_slots = zip(*shuffle_list)
            # unique_move is the position the agents want to move to
            # return_count is the number of times the unique_move is wanted
            unique_move, indices, return_count = np.unique(move_slots, return_index=True,
                                                           return_counts=True, axis=0)
            search_list = np.array(move_slots)
            print('@map_env - Starting fix (if any)')
            print(unique_move)
            print(indices)
            print(return_count)
            print(search_list)

            # first go through and remove moves that can't possible happen. Three types
            # 1. Trying to move into an agent that has been issued a stay command
            # 2. Trying to move into the spot of an agent that doesn't have a move
            # 3. Two agents trying to walk through one another

            # Resolve all conflicts over a space
            if np.any(return_count > 1):
                for move, index, count in zip(unique_move, indices, return_count):
                    if count > 1:
                        # check that the cell you are fighting over doesn't currently
                        # contain an agent that isn't going to move for one of the agents
                        # If it does, all the agents commands should become STAY
                        # since no moving will be possible
                        conflict_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents = [agent_to_slot[i] for i in conflict_indices]
                        # all other agents now stay in place so update their moves
                        # to reflect this
                        conflict_cell_free = True
                        for agent in all_agents:
                            moves_copy = agent_moves.copy()
                            # TODO(ev) code duplication, simplify
                            locs = [list(agent.location) for agent in self.world_state['agents']]
                            if move.tolist() in locs: #self.agent_pos
                                # find the agent that is currently at that spot and make sure
                                # that the move is possible. If it won't be, remove it.
                                conflicting_agent = agent_by_pos[tuple(move)]
                                curr_pos = list(agent.location)
                                curr_conflict_pos = list(conflicting_agent.location)
                                conflict_move = agent_moves.get(
                                    conflicting_agent,
                                    curr_conflict_pos)

                                # Condition (1):
                                # a STAY command has been issued
                                if agent == conflicting_agent:
                                    conflict_cell_free = False
                                # Condition (2)
                                # its command is to stay
                                # or you are trying to move into an agent that hasn't
                                # received a command
                                elif conflicting_agent not in moves_copy.keys() or \
                                        curr_conflict_pos == conflict_move:
                                    conflict_cell_free = False

                                # Condition (3)
                                # It is trying to move into you and you are moving into it
                                elif conflicting_agent in moves_copy.keys():
                                    if conflicting_agent.location == curr_pos and \
                                            move.tolist() == conflicting_agent.location.tolist():
                                        print('keep trying to move into each other')
                                        conflict_cell_free = False

                        # if the conflict cell is open, let one of the conflicting agents
                        # move into it
                        if conflict_cell_free:
                            agent_idx = [idx for idx, agent_obj in enumerate(self.world_state['agents']) if id(agent_obj) == id(agent_to_slot[index])][0]
                            self.world_state['agents'][agent_idx].update_agent_pos(move)
                            agent_by_pos = {tuple(agent.location):
                                            agent for agent in self.world_state['agents']}
                        # ------------------------------------
                        # remove all the other moves that would have conflicted
                        remove_indices = np.where((search_list == move).all(axis=1))[0]
                        all_agents = [agent_to_slot[i] for i in remove_indices]
                        # all other agents now stay in place so update their moves
                        # to stay in place
                        for agent in all_agents:
                            agent_moves[agent] = agent.location
            
            print('@map_env - Ended fix (if any)')
            print(move_slots)
            print(agent_to_slot)
            print(agent_by_pos)
            print(agent_moves)

            print('@map_env - Starting un-conflicted moves (if any)')
            # make the remaining un-conflicted moves
            while len(agent_moves.items()) > 0:
                agent_by_pos = {tuple(agent.location): agent for agent in self.world_state['agents']}
                num_moves = len(agent_moves.items())
                moves_copy = agent_moves.copy()
                del_keys = []
                for agent, move in moves_copy.items():
                    print('inside agent move')
                    print(agent)
                    print(move)
                    print([agent.location for agent in self.world_state['agents']])
                    if agent in del_keys:
                        continue
                    if list(move) in [list(agent.location) for agent in self.world_state['agents']]:
                        # find the agent that is currently at that spot and make sure
                        # that the move is possible. If it won't be, remove it.
                        conflicting_agent = agent_by_pos[tuple(move)]
                        curr_pos = list(agent.location)
                        curr_conflict_pos = list(conflicting_agent.location)
                        conflict_move = agent_moves.get(conflicting_agent, curr_conflict_pos)
                        # Condition (1):
                        # a STAY command has been issued
                        if agent == conflicting_agent:
                            del agent_moves[agent]
                            del_keys.append(agent)
                        # Condition (2)
                        # its command is to stay
                        # or you are trying to move into an agent that hasn't received a command
                        elif conflicting_agent not in moves_copy.keys() or \
                                curr_conflict_pos == conflict_move:
                            del agent_moves[agent]
                            del_keys.append(agent)
                        # Condition (3)
                        # It is trying to move into you and you are moving into it
                        elif conflicting_agent in moves_copy.keys():
                            if agent_moves[conflicting_agent] == curr_pos and \
                                    move == conflicting_agent.location.tolist():
                                print('keep trying to move into each other 2')
                                del agent_moves[conflicting_agent]
                                del agent_moves[agent]
                                del_keys.append(agent)
                                del_keys.append(conflicting_agent)
                    # this move is unconflicted so go ahead and move
                    else:
                        agent_idx = [idx for idx, agent_obj in enumerate(self.world_state['agents']) if id(agent_obj) == id(agent)][0]
                        self.world_state['agents'][agent_idx].update_agent_pos(move)
                        del agent_moves[agent]
                        del_keys.append(agent)

                # no agent is able to move freely, so just move them all
                # no updates to hidden cells are needed since all the
                # same cells will be covered
                if len(agent_moves) == num_moves:
                    for agent, move in agent_moves.items():
                        self.world_state['agents'][agent].update_agent_pos(move)
                    break

        # All possible action handlers
        print('@map_env - Executing Task Handlers')
        for agent in agent_tasks:
            print('conducting tasks now')
            task_id = agent_tasks[agent][0]
            task_action = agent_tasks[agent][1]
            print(task_action)
            # do we still need the second check?
            if task_action[0] == 'PICK' and task_action[2] == agent.location:
                print('@map_env - Executing Pick Action')
                # agent.pick(task_id, task_action[1], task_action[2], task_action[3])
                agent.pick(task_id, task_action[1])
            elif task_action[0] == 'CHOP' and task_action[3] == agent.location:
                print('@map_env - Executing Chop Action')
                agent.chop(task_id, task_action[1], task_action[2])
            elif task_action[0] == 'COOK' and task_action[3] == agent.location:
                print('@map_env - Executing Cook Action')
                agent.cook(task_id, task_action[1], task_action[2])
            elif task_action[0] == 'SCOOP' and task_action[2] == agent.location:
                print('@map_env - Executing Scoop Action')
                agent.scoop(task_id, task_action[1])
            elif task_action[0] == 'SERVE' and task_action[2] == agent.location:
                print('@map_env - Executing Serve Action')
                agent.serve(task_id, task_action[1])
            elif task_action[0] == 'DROP':
                print('@map_env - Executing Drop Action')
                agent.drop(task_id)

    def map_to_colors(self, map=None, color_map=None):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        map: np.ndarray
            map to convert to colors
        color_map: dict
            mapping between array elements and desired colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        if map is None:
            map = self.world_map
        if color_map is None:
            color_map = self.color_map

        rgb_arr = np.zeros((map.shape[0], map.shape[1], 3), dtype=int)
        for row_elem in range(map.shape[0]):
            for col_elem in range(map.shape[1]):
                rgb_arr[row_elem, col_elem, :] = color_map[map[row_elem, col_elem]]

        return rgb_arr

    def render(self, filename=None):
        """ Creates an image of the map to plot or save.
        Args:
            path: If a string is passed, will save the image
                to disk at this location.
        """
        print('@map_env - render()')
        rgb_arr = self.map_to_colors(self.world_map)
        plt.imshow(rgb_arr, interpolation='nearest')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)