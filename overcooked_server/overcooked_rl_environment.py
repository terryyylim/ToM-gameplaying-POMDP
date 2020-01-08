import gym
import numpy as np

class Overcooked_Env(gym.Env):

    def __init__():
        pass

    def seed(self, seed = None):
        pass

    def step(self):
        #calculate_new_state
        #move_user_to_location
        # OR perform action
        return next_state, self.reward, self.done, #something here

    def calculate_new_state(self):
        pass

    def move_user_to_location(self):
        pass
    
    def perform_action(self):
        pass

    def update_reward(self):
        pass


    


    #gym.spaces.Discrete(8 + number of actions)