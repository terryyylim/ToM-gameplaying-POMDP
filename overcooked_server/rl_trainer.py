import copy
import os
import logging
import sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from .rl_networks import ConvMLPNetwork
from .utilities.rl_utils import flip_array, vectorize_world_state, setup_logger
from .utilities.Epsilon_Greedy_Exploration import Epsilon_Greedy_Exploration
from .utilities.Utility_Functions import normalise_rewards, create_actor_distribution

class PPOTrainer():
    """Master class to orchestrate training of PPO Algorithm in Overcooked AI"""
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger()
        self.policy_new = self.create_NN(self.config.hyperparameters["obs_space"], 
                                        self.config.hyperparameters["action_space"], 
                                        self.config.hyperparameters["nn_params"])
                                        
        self.policy_old = self.create_NN(self.config.hyperparameters["obs_space"], 
                                        self.config.hyperparameters["action_space"], 
                                        self.config.hyperparameters["nn_params"])

        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.policy_new_optim = optim.Adam(self.policy_new.parameters(), lr = self.config['learning_rate'], eps=1e-4)
        self.exploration_strategy = Epsilon_Greedy_Exploration(self.config)
        self.episode_number = 0
        self.timesteps = 0

        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []
        self.rolling_results = []

    def init_batch_lists(self):
        if self.states_batched:
            self.all_states.extend(self.states_batched)
            self.all_actions.extend(self.actions_batched)
            self.all_rewards.extend(self.rewards_batched)
        self.states_batched = []
        self.actions_batched = []
        self.rewards_batched = []

    def setup_agents(self, agent_list):
        self.agents = agent_list
        self.num_agents = len(agent_list)
        self.reset_game()

    def update_learning_rate(self, starting_lr,  optimizer):
        """Lowers the learning rate according to how close we are to the solution"""
        if len(self.rolling_results) > 0:
            last_rolling_score = self.rolling_results[-1]
            if last_rolling_score > 0.75 * self.average_score_required_to_win:
                new_lr = starting_lr / 100.0
            elif last_rolling_score > 0.6 * self.average_score_required_to_win:
                new_lr = starting_lr / 20.0
            elif last_rolling_score > 0.5 * self.average_score_required_to_win:
                new_lr = starting_lr / 10.0
            elif last_rolling_score > 0.25 * self.average_score_required_to_win:
                new_lr = starting_lr / 2.0
            else:
                new_lr = starting_lr
            for g in optimizer.param_groups:
                g['lr'] = new_lr
        if random.random() < 0.001: self.logger.info("Learning rate {}".format(new_lr))

    def reset_game(self):
        self.current_episode_state = {}
        self.current_episode_action = {}
        self.current_episode_rewards= {}
        for agent in self.agents:
            self.current_episode_state[agent] = []
            self.current_episode_action[agent] = []
            self.current_episode_reward[agent] = []
            
    def pick_action(self, state, exploration_episilon):
        if random.random() <= exploration_episilon:
            action = random.randint(0, self.output_dim - 1)
            return action
        
        state = torch.from_numpy(state).float()
        actor_output = self.policy_new.forward(state)
        if self.action_choice_output_columns is not None: #whats this?
            actor_output = actor_output[:, self.action_choice_output_columns]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action = action_distribution.sample().cpu()
        return action

    def step(self, agent_id, world_state):
        world_state_np = vectorize_world_state(world_state)
        exploration_epsilon =  self.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.episode_number})
        flipped_arr = flip_array(agent_id, world_state_np)
        action = self.pick_action(flipped_arr, exploration_epsilon)
        best_goal = [-1, {'steps': [action], 'rewards': -1}]
        return best_goal

    def policy_learn(self):
        all_discounted_returns = self.calculate_all_discounted_returns()
        if self.hyperparameters["normalise_rewards"]:
            all_discounted_returns = normalise_rewards(all_discounted_returns)
        for _ in range(self.hyperparameters["learning_iterations_per_round"]):   #number of epochs 
            all_ratio_of_policy_probabilities = self.calculate_all_ratio_of_policy_probabilities()
            loss = self.calculate_loss([all_ratio_of_policy_probabilities], all_discounted_returns)
            self.take_policy_new_optimisation_step(loss)
        self.init_batch_lists()

    def end_episode(self):
        self.episode_number += 1
        self.states_batched.extend(list(self.current_episode_state.values()))
        self.actions_batched.extend(list(self.current_episode_action.values()))
        self.rewards_batched.extend(list(self.current_episode_rewards.values()))
        if episode_number % self.hyperparameters['episodes_per_learning_round'] == 0:
            loss = self.policy_learn()
            self.update_learning_rate(self.hyperparameters['learning_rate'], self.policy_new_optimizer)
            self.equalise_policies()
        self.reset_game()
    
    def calculate_all_ratio_of_policy_probabilities(self):
        """For each action calculates the ratio of the probability that the new policy would have picked the action vs.
         the probability the old policy would have picked it. This will then be used to inform the loss"""
        all_states = [state for states in self.states_batched for state in states]
        all_actions = [[action] for actions in self.actions_batched for action in actions]
        all_states = torch.stack([torch.Tensor(states).float().to(self.device) for states in all_states])

        all_actions = torch.stack([torch.Tensor(actions).float().to(self.device) for actions in all_actions])
        all_actions = all_actions.view(-1, len(all_states))

        new_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_new, all_states, all_actions)
        old_policy_distribution_log_prob = self.calculate_log_probability_of_actions(self.policy_old, all_states, all_actions)
        ratio_of_policy_probabilities = torch.exp(new_policy_distribution_log_prob) / (torch.exp(old_policy_distribution_log_prob) + 1e-8)
        return ratio_of_policy_probabilities

    def calculate_log_probability_of_actions(self, policy, states, actions):
        """Calculates the log probability of an action occuring given a policy and starting state"""
        policy_output = policy.forward(states).to(self.device)
        policy_distribution = create_actor_distribution("DISCRETE", policy_output, self.action_size)
        policy_distribution_log_prob = policy_distribution.log_prob(actions)
        return policy_distribution_log_prob

    def calculate_loss(self, all_ratio_of_policy_probabilities, all_discounted_returns):
        """Calculates the PPO loss"""
        all_ratio_of_policy_probabilities = torch.squeeze(torch.stack(all_ratio_of_policy_probabilities))
        all_ratio_of_policy_probabilities = torch.clamp(input=all_ratio_of_policy_probabilities,
                                                        min = -sys.maxsize,
                                                        max = sys.maxsize)
        all_discounted_returns = torch.tensor(all_discounted_returns).to(all_ratio_of_policy_probabilities)
        potential_loss_value_1 = all_discounted_returns * all_ratio_of_policy_probabilities
        potential_loss_value_2 = all_discounted_returns * self.clamp_probability_ratio(all_ratio_of_policy_probabilities)
        loss = torch.min(potential_loss_value_1, potential_loss_value_2)
        loss = -torch.mean(loss)
        return loss

    def take_policy_new_optimisation_step(self, loss):
        """Takes an optimisation step for the new policy"""
        self.policy_new_optimizer.zero_grad()  # reset gradients to 0
        loss.backward()  # this calculates the gradients
        torch.nn.utils.clip_grad_norm_(self.policy_new.parameters(), self.hyperparameters[
            "gradient_clipping_norm"])  # clip gradients to help stabilise training
        self.policy_new_optimizer.step()  # this applies the gradients

    def clamp_probability_ratio(self, value):
        """Clamps a value between a certain range determined by hyperparameter clip epsilon"""
        return torch.clamp(input=value, min=1.0 - self.hyperparameters["clip_epsilon"],
                                  max=1.0 + self.hyperparameters["clip_epsilon"])

    def equalise_policies(self):
        """Sets the old policy's parameters equal to the new policy's parameters"""
        for old_param, new_param in zip(self.policy_old.parameters(), self.policy_new.parameters()):
            old_param.data.copy_(new_param.data)
    
    def create_NN(self, input_dim, output_dim, hyperparameters):
        return ConvMLPNetwork(input_dim, output_dim, hyperparameters)

    def receive_rewards(self, rewards):
        for agent_id in self.agents:
            self.current_episode_rewards[agent_id].append(rewards[agent_id])
        self.timesteps += 1

    def set_random_seeds(self, random_seed):
        """Sets all possible random seeds so results can be reproduced"""
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
            torch.cuda.manual_seed(random_seed)