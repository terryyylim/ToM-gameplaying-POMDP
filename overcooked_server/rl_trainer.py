import copy

import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class PPOTrainer():
    def __init__(self, config):
        self.config = config

        self.policy_new = self.create_NN(input_dim, output_dim, hyperparameters) # what are the configs for inputdim, outputdim, and hyperparams?
        self.policy_old = self.create_NN(input_dim, output_dim, hyperparameters)
        self.policy_old.load_state_dict(copy.deepcopy(self.policy_new.state_dict()))
        self.policy_new_optim = optim.Adam(self.policy_new.parameters(), lr = self.config['learning_rate'], eps=1e-4)
        self.episode_number = 0
        self.many_episode_states = []
        self.many_episode_actions = []
        self.many_episode_rewards = []

        self.setup_agents()

    def setup_agents(self):
        pass
    
    def step(self, world_state):
        action_dict = {}
        #world_state to np array
        #exploration_epsilon =  self.exploration_strategy.get_updated_epsilon_exploration({"episode_number": self.episode_number})
        for agent in agents:
            #flip_array
            action = agent.pick_action(flipped_arr, exploration_episilon) #maybe just need to do policy.forward dont need to update via agent
            action_dict[agent.agent_id] = [-1, {'goal': [action], 'rewards': -1}]
        return action_dict




        pass

    def policy_learn():
        pass


    def create_NN(self, input_dim, output_dim, hyperparameters):
        return ConvMLPNetwork(input_dim, output_dim, hyperparameters)



class ConvMLPNetwork(nn.Module):
    """"CNN and MLP network to process OvercookedAI world_state. 
    Adapted from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/baselines_utils.py"""
    
    def __init__(self, input_size, output_size, params):
        super(ConvMLPNetwork, self).__init__()
        num_hidden_layers = params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = params["SIZE_HIDDEN_LAYERS"]
        num_filters = params["NUM_FILTERS"]
        num_convs = params["NUM_CONV_LAYERS"]
        shape = params['OBS_STATE_SHAPE']

        #Need to double check the padding and calculations
        self.conv_initial = nn.Conv2d(input_size, num_filters, 5, padding = self.calculate_padding(5, 'same'))
        self.conv_layers = nn.ModuleList([nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = self.calculate_padding(5, 'same')) for i in range(0,num_convs - 2)])
        self.conv_final = nn.Conv2d(num_filters, num_filters, kernel_size = 3, padding = self.calculate_padding(3, 'valid'))
        self.linear_initial = nn.Linear(shape[0] * shape[1] * num_filters, size_hidden_layers)
        self.linear_layers = nn.ModuleList([nn.Linear(size_hidden_layers, size_hidden_layers) for i in range(num_hidden_layers)])
        self.linear_final = nn.Linear(size_hidden_layers, output_size)

    def forward(self, x):
        x = self.conv_initial(x)
        for convs in self.conv_layers:
            x = F.leaky_relu(convs(x))
            print (x.shape)
        x = self.conv_final(x)
        print (x.shape)
        x = x.view(x.size()[0], -1)
        x = self.linear_initial(x)
        for linear in self.linear_layers:
            x = F.leaky_relu(linear(x))
        x = F.softmax(self.linear_final(x))
        return x
        
    def calculate_padding(self, kernel, pad_type):
        if pad_type == 'valid':
            return 0
        if pad_type == 'same':
            return int((kernel - 1) / 2)