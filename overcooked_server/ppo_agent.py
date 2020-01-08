"""
PPO Agent configured to interact with a Multi-class environment outside of its own class.
Adapted from https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch
"""

import copy
import sys
import torch
import numpy as np
from torch import optim
from rl_base_agent import Base_Agent
from exploration_strategies.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from utilities.Parallel_Experience_Generator import Parallel_Experience_Generator
from utilities.Utility_Functions import normalise_rewards, create_actor_distribution

class PPO(Base_Agent):
    agent_name = "PPO"

    def __init__(self, config):
        Base_Agent.__init__(self, config)
        self.policy_output_size = self.calculate_policy_output_size()
        self.policy_new = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old = self.create_NN(input_dim=self.state_size, output_dim=self.policy_output_size)
        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.policy_new_optimizer = optim.Adam(self.policy_new.parameters(), lr=self.hyperparameters["learning_rate"], eps=1e-4)
        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.config.seed,
                                                                  self.hyperparameters, self.action_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

    def calculate_policy_output_size(self):
        """Initialise policies"""
        if self.action_types == "DISCRETE":
            return self.action_size
        elif self.action_types == "CONTINUOUS":
            return self.action_size * 2 #Because for mean and std

    

