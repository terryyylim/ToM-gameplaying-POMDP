import click
import collections
import numpy as np
import time
import threading
import os

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.overcooked_map_env import OvercookedEnv
import helpers

class TaskThread(threading.Thread):
    def __init__(
        self,
        controller
    ) -> None:
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.controller = controller
    
    def run(self) -> None:
        # while not self.event.wait(1):
        self.controller.env.random_queue_order()

class Controller(object):
    def __init__(
        self,
        env_name='cleanup'
    ) -> None:
        self.env_name = env_name
        if env_name == 'overcooked':
            print('Initializing Overcooked environment')
            self.env = OvercookedEnv()
            print(self.env)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            # self.env = CleanupEnv(num_agents=5, render=True)
        else:
            print('Error! Not a valid environment type')

    # undone: lacking rewards, observation updates
    def rollout(self, best_goals, horizon=50, save_path=None):
        print('rollout@rollout')
        rewards = []
        observations = []
        shape = self.env.world_map.shape
        full_obs = [np.zeros(
            (shape[0], shape[1], 3), dtype=np.uint8) for i in range(horizon)]
        
        action_mapping = {}
        for agent in best_goals:
            action_mapping[agent] = (
                best_goals[agent][0],
                best_goals[agent][1]['steps'][0]
            )

        print(action_mapping)
        print('@rollout - Starting step function')
        self.env.step(action_mapping)
        print(f'@rollout - Currently at horizon - {horizon}')
        self.env.render('./ipomdp/images/timestep'+str(horizon))
        
    def check_all_except_one_done(self, agent, agent_cur_step, agent_max_steps):
        temp_agent_cur_step = {key for key in agent_cur_step if key != agent}
        temp_agent_max_steps = {key for key in agent_max_steps if key != agent}

        if temp_agent_cur_step == temp_agent_max_steps:
            return True
        return False


@click.command()
@click.option("--env", default='overcooked', help="input the env you want to initialize here")
@click.option("--timer", default=10, help="input the number of seconds you want the game to run")
def main(env: str, timer: int) -> None:
    c = Controller(env_name=env)

    thread = TaskThread(c)
    thread.start()

    end_time = time.time() + 30
    time_step_execution = False
    while time.time() < end_time:

        # If goal space exist, else do nothing
        if not time_step_execution and c.env.world_state['goal_space']:
            best_goals = c.env.find_agents_best_goal()
            print('best_goals')
            print(best_goals)
            time_step_execution = True

        # TO-DO Given best_goals: perform 1 time-step
        else:
            continue

        if time.time() == end_time - 8:
            print(c.env.world_state)

    thread.event.set()

if __name__ == "__main__":
    main()
