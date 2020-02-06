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
        #initialise self.environment as a config
        self.experience_generator = Parallel_Experience_Generator(self.environment, self.policy_new, self.config.seed,
                                                                  self.hyperparameters, self.action_size)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)

    def calculate_policy_output_size(self):
        """Initialise policies"""
        if self.action_types == "DISCRETE":
            return self.action_size
        elif self.action_types == "CONTINUOUS":
            return self.action_size * 2 #Because for mean and std

    def step(self):
        """Run a step for the PPO agent"""
        #needs to return desired action
        exploration_epsilon = self.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.episode_number})
        self.many_episode_states, self.many_episode_actions, self.many_episode_rewards = self.experience_generator.play_n_episodes(self.hyperparameters["episodes_per_learning_round"], exploration_epsilon)
        self.episode_number += self.hyperparameters["episodes_per_learning_round"]
        self.policy_learn()
        self.update_learning_rate(self.hyperparameters["learning_rate"], self.policy_new_optimizer)
        self.equalise_policies()

    def policy_learn(self):
        """A learning iteration for the policy"""
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        for _ in range(self.hyperparameters["learning_iterations_per_round"]):
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)

    def calculate_all_discounted_returns(self):
        """Calculates the cumulative discounted return for each episode which we will then use in a learning iteration"""
        all_discounted_returns = []
        for episode in range(len(self.many_episode_states)):
            discounted_returns = [0]
            for ix in range(len(self.many_episode_states[episode])):
                return_value = self.many_episode_rewards[episode][-(ix + 1)] + self.hyperparameters["discount_rate"]*discounted_returns[-1]
                discounted_returns.append(return_value)
            discounted_returns = discounted_returns[1:]
            all_discounted_returns.extend(discounted_returns[::-1])
        return all_discounted_returns


    


