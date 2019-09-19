from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
from ray.rllib.env import MultiAgentEnv

from ipomdp.envs.map_configs import *
from ipomdp.agents.agent_configs import *
from ipomdp.agents.base_agent import *

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
        self.base_map = self.ascii_to_numpy(ascii_map)
        self.world_map = self.base_map
        self.world_state = defaultdict(list)
        self.agent_initialization = agent_initialization

        self.num_agents = len(agent_initialization)
        
        self.agents = {}

        self.color_map = color_map if color_map is not None else DEFAULT_COLOURS

        self.table_tops = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == '@':
                    self.table_tops.append([row, col])
        self.setup_agents()

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
    def step(self, actions):
        """Takes in a dict of actions and converts them to a map update
        Parameters
        ----------
        actions: dict {agent-id: int}
            dict of actions, keyed by agent-id that are passed to the agent. The agent
            interprets the int and converts it to a command
        Returns
        -------
        observations: dict of arrays representing agent observations
        rewards: dict of rewards for each agent
        dones: dict indicating whether each agent is done
        info: dict to pass extra info to gym
        """

        self.beam_pos = []
        agent_actions = {}
        for agent_id, action in actions.items():
            agent_action = self.agents[agent_id].action_map(action)
            agent_actions[agent_id] = agent_action

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
        rgb_arr = self.map_to_colors(self.world_map)
        plt.imshow(rgb_arr, interpolation='nearest')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def reset_map(self):
        """Resets the map to be empty as well as a custom reset set by subclasses"""
        self.world_map = np.full((len(self.base_map), len(self.base_map[0])), ' ')
        self.build_table_tops()
        # self.custom_reset() -- do we need to reset?

    def build_table_tops(self):
        for i in range(len(self.table_tops)):
            row, col = self.table_tops[i]
            self.world_map[row, col] = '@'