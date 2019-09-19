import click
import time
import threading

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.overcooked_map_env import OvercookedEnv

class TaskThread(threading.Thread):
    def __init__(
        self,
        controller
    ) -> None:
        threading.Thread.__init__(self)
        self.event = threading.Event()
        self.controller = controller
    
    def run(self) -> None:
        while not self.event.wait(10):
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


@click.command()
@click.option("--env", default='overcooked', help="input the env you want to initialize here")
@click.option("--timer", default=10, help="input the number of seconds you want the game to run")
def main(env: str, timer: int) -> None:
    c = Controller(env_name=env)

    thread = TaskThread(c)
    thread.start()

    end_time = time.time() + 30
    while time.time() < end_time:

        # action_1 = c.env.agents[1].find_best_goal()
        # action_2 = c.env.agents[2].find_best_goal()
        # TO-DO Given action_1 and action_2: perform 1 time-step

        if time.time() == end_time - 8:
            print(c.env.world_state)

    thread.event.set()

if __name__ == "__main__":
    main()
