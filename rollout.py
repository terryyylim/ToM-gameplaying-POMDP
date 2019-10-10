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
    counter = 1
    make_video = False

    # Remove the previous images and video
    images_dir = 'ipomdp/images'
    videos_dir = 'videos'
    for file in os.listdir(images_dir):
        if file.endswith('.png'):
            os.remove(images_dir+'/'+file)
    os.remove(videos_dir+'/trajectory.mp4')

    c.env.render('./ipomdp/images/timestep0')
    while time.time() < end_time:

        # If goal space exist, else do nothing
        if not time_step_execution and c.env.world_state['goal_space']:
            if counter < 120:
                print(f'============= Executing next timestep {counter} @ {time.time()} =============')
                for agent in c.env.world_state['agents']:
                    agent.location = tuple(agent.location)
                    print(f'Agent {id(agent)} at {agent.location}, holding {agent.holding}')
                    if agent.holding:
                        print(f'Agent {id(agent)} is holding {agent.holding}, of state {agent.holding.state}\n')
                time_step_execution = True
                best_goals = c.env.find_agents_best_goal()
                print(f'Agent Best Goals\n {best_goals}\n')

                c.rollout(best_goals, counter)

                # Mini-hack: remove all cells to get to items in world_state
                # This hack causes agent to get stuck sometimes
                hack_cells = [(1,3), (1,8), (3,11), (5,1), (6,1), (7,1), (7,11), (7,3), (7,5)]
                for valid_cell in hack_cells:
                    c.env.world_state['valid_cells'].append(valid_cell)

                c.env.world_state['valid_cells'] = list(set(c.env.world_state['valid_cells']))

                print()
                print(f'============= Summary after timestep {counter} =============')
                print(f'Current world_state: \n{c.env.world_state}\n\n')
                ingredient_loc = [ingredient.location for ingredient in c.env.world_state['ingredients']]
                chopping_boards_state = {cb: (cb.state, cb.location) for cb in c.env.world_state['chopping_board']}
                goal_space = c.env.world_state['goal_space']
                goal_info = [(id(goal), goal.head, id(goal.head), goal.head.state, goal.head.task) for goal in c.env.world_state['goal_space']]
                agent_holding_status = {agent: agent.holding for agent in c.env.world_state['agents']}
                agent_can_update_status = {agent: agent.can_update for agent in c.env.world_state['agents']}
                print(f'Current ingredient locations: \n{ingredient_loc}\n')
                print(f'Current chopping board states: \n{chopping_boards_state}\n')
                print(f'Current goal space: \n{goal_space}\n')
                print(f'Current goal info (goal_id; goal_head; goal_head_id; goal_head_state, goal_head_task): \n{goal_info}\n')
                print(f'Current agents holding status:\n')
                print(agent_holding_status)
                print(f'Current agents can_update status:\n')
                print(agent_can_update_status)

                if counter == 80 or counter == 83:
                    print('@rollout - Making video now')
                    make_video = True
                counter += 1
                time_step_execution = False
        else:
            continue

        if make_video:
            video_path = os.path.abspath(os.path.dirname(__file__)) + '/videos'
            image_path = os.path.abspath(os.path.dirname(__file__)) + '/ipomdp/images'
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            fps = 1
            video_name = 'trajectory'
            helpers.make_video_from_image_dir(
                video_path,
                image_path,
                video_name,
                fps
            )

    thread.event.set()

if __name__ == "__main__":
    main()
