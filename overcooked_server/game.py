from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import os
import click
import copy
import sys
import pandas as pd
import pygame as pg
from datetime import datetime

from map_env import MapEnv
from overcooked_env import OvercookedEnv
from sprites import *
from settings import *

import helpers

class Game:
    def __init__(
        self,
        num_ai_agents: int=1,
        num_rl_agents: int=0,
        RLTrainer= None,
        is_simulation: bool=False,
        simulation_episodes: int=500,
        is_tom: bool=False,
        experiment_id: str='1'
    ) -> None:
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.is_simulation = is_simulation
        self.is_tom = is_tom
        self.experiment_id = experiment_id
        self.RLTrainer = RLTrainer
        self.TERMINATING_EPISODE = simulation_episodes
        AI_AGENTS_TO_INITIALIZE = {}
        RL_AGENTS_TO_INITIALIZE = {}
        agent_id = 0
        for idx in range(1, num_ai_agents+1):
            idx_str = str(idx)
            AI_AGENTS_TO_INITIALIZE[idx_str] = AI_AGENTS[idx_str]
            agent_id = idx
        
        for idx in range(agent_id+1, num_rl_agents+1):
            idx = str(idx)
            RL_AGENTS_TO_INITIALIZE[idx] = AI_AGENTS[idx]

        # Logs saving
        game_folder = os.path.dirname(__file__)
        self.experiment_folder = os.path.join(game_folder, experiment_id)
        helpers.check_dir_exist(self.experiment_folder)
        self.images_folder = os.path.join(self.experiment_folder, 'images')
        helpers.check_dir_exist(self.images_folder)

        if self.is_simulation:
            self.env = OvercookedEnv(
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                rl_agents=RL_AGENTS_TO_INITIALIZE,
                rl_trainer= self.RLTrainer,
                queue_episodes=QUEUE_EPISODES
            )
            self.load_data()
            self.results_filename = 'results/' + self.env.results_filename + '.csv'
            self.results = defaultdict(int)
            self.results_col = []
            for i in range(self.TERMINATING_EPISODE+1):
                if i%50 == 0:
                    self.results[str(i)] = 0
                    self.results_col.append(str(i))

            self.run_simulation(simulation_episodes)
        else:
            final_HUMAN_AGENTS = {k:v for k,v in HUMAN_AGENTS.items() if k == '1'}
            self.env = OvercookedEnv(
                human_agents=final_HUMAN_AGENTS,
                ai_agents=AI_AGENTS_TO_INITIALIZE,
                queue_episodes=QUEUE_EPISODES
            )
            self.info_df = pd.DataFrame(
                columns=[
                    'episode', 'player_coords', 'agent_coords', 'player_action', 'agent_action', 'available_orders', 'score', 'total_score'
                ]
            )
            self.load_data()
            self.results_filename = self.experiment_folder + '/' + self.env.results_filename + '.csv'
        self.results = defaultdict(int)
        self.results_col = []
        for i in range(self.TERMINATING_EPISODE+1):
            if i%50 == 0:
                self.results[str(i)] = 0
                self.results_col.append(str(i))

    def _deep_copy(self, obj):
        return copy.deepcopy(obj)

    def load_data(self):
        episode = self.env.world_state

        self.PLAYERS, self.PLATES, self.POTS = {}, {}, {}
        self.INGREDIENTS = []
        self.TABLE_TOPS = self._deep_copy(TABLE_TOPS)
        
        RETURN_STATION_OCCUPIED = False
        self.RETURN_STATION = {
            'state': 'empty',
            'coords': episode['return_counter']
        }

        for agent in episode['agents']:
            self.PLAYERS[agent.id] = {
                'holding': agent.holding,
                'coords': agent.location
            }
        for plate in episode['plate']:
            if plate.location in self.TABLE_TOPS:
                self.TABLE_TOPS.remove(plate.location)
            if plate.location == episode['return_counter']:
                RETURN_STATION_OCCUPIED = True
            self.PLATES[plate.plate_id] = {
                'state': plate.state,
                'coords': plate.location
            }
        for pot in episode['pot']:
            if pot.ingredient_count:
                pot_ingredient_count = pot.ingredient_count
            else:
                pot_ingredient_count = defaultdict(int)
            self.POTS[pot.pot_id] = {
                'ingredient_count': pot_ingredient_count,
                'coords': pot.location
            }

        for ingredient in episode['ingredients']:
            if ingredient.location in self.TABLE_TOPS:
                self.TABLE_TOPS.remove(ingredient.location)
            self.INGREDIENTS.append([
                ingredient.name,
                ingredient.state,
                ingredient.location
            ])
        
        if RETURN_STATION_OCCUPIED:
            self.RETURN_STATION = {
                'state': 'filled',
                'coords': episode['return_counter']
            }
        self.CHOPPING_BOARDS = [chopping_board.location for chopping_board in episode['chopping_board'] if chopping_board.state != 'taken']
        self.INGREDIENTS_STATION = INGREDIENTS_STATION
        self.SERVING_STATION = SERVING_STATION
        self.WALLS = WALLS

    def new(
        self,
        players: Dict[int, Tuple[int,int]],
        table_tops: List[Tuple[int,int]],
        ingredients,
        chopping_boards: List[Tuple[int,int]],
        plates: Dict[int, List[Any]],
        pots: Dict[int, List[Any]],
        ingredient_stations: Dict[str, Tuple[int,int]],
        serving_stations: List[Tuple[int,int]],
        return_station: List[Tuple[int,int]],
        walls: List[Tuple[int,int]]=WALLS,
        score: Tuple[int,int]=SCOREBOARD_SCORE,
        orders: Tuple[int,int]=SCOREBOARD_ORDERS,
        timer: Tuple[int,int]=SCOREBOARD_TIMER,
        scoreboard: List[Tuple[int,int]]=SCOREBOARD
    ) -> None:
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.table_tops = pg.sprite.Group()
        self.ingredients = pg.sprite.Group()
        self.chopping_boards = pg.sprite.Group()
        self.plates = pg.sprite.Group()
        self.pot_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.serving_stations = pg.sprite.Group()
        self.return_station = pg.sprite.Group()
        self.score = pg.sprite.Group()
        self.orders = pg.sprite.Group()
        self.timer = pg.sprite.Group()
        self.scoreboard = pg.sprite.Group()
        self.walls = walls
        self.player_count = len(players)

        for idx in range(1, self.player_count+1):
            idx = str(idx)
            if idx == '1':
                self.player_1 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '2':
                self.player_2 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '3':
                self.player_3 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '4':
                self.player_4 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
        for table_top_coord in table_tops:
            TableTop(self, table_top_coord[1], table_top_coord[0])
        for ingredient in ingredients:
            ingredient_name = ingredient[0]
            ingredient_state = ingredient[1]
            ingredient_coords = ingredient[2]
            Ingredients(self, ingredient_name, ingredient_state, ingredient_coords[1], ingredient_coords[0])
        for chopping_board_coord in chopping_boards:
            ChoppingBoardStation(self, chopping_board_coord[1], chopping_board_coord[0])
        for key, val in plates.items():
            plate_state = val['state']
            plate_coord = val['coords']
            PlateStation(self, plate_state, plate_coord[1], plate_coord[0])
        for key, val in pots.items():
            pot_ingredient_count = val['ingredient_count']
            pot_coord = val['coords']
            PotStation(self, pot_ingredient_count, pot_coord[1], pot_coord[0])
        for key, val in ingredient_stations.items():
            ingredient = key
            ingredient_station_coord = val
            for coords in ingredient_station_coord:
                IngredientStation(self, ingredient, coords[1], coords[0])
        for serving_station_coord in serving_stations:
            ServingStation(self, serving_station_coord[1], serving_station_coord[0])
            
        ReturnStation(self, return_station['state'], return_station['coords'][1], return_station['coords'][0])
        Score(self, score[1], score[0])
        Orders(self, orders[1], orders[0])
        Timer(self, timer[1], timer[0])
        for scoreboard_coord in scoreboard:
            ScoreBoard(self, scoreboard_coord[1], scoreboard_coord[0])

    def save_results(self):
        try:
            results_df = pd.read_csv(self.results_filename)
            new_row = pd.DataFrame([self.results], columns=self.results.keys())
            results_df = pd.concat([results_df, new_row], axis=0).reset_index()
        except FileNotFoundError:
            results_df = pd.DataFrame([self.results], columns=self.results.keys())
        results_df = results_df[self.results_col]
        results_df.to_csv(self.results_filename, index=False)

        if not self.is_simulation:
            self.info_df.to_csv(self.experiment_folder + '/experiments_' + self.env.results_filename + '.csv', index=False)

            # agent_types = [agent.is_inference_agent for agent in self.env.world_state['agents']]
            video_name_ext = helpers.get_video_name_ext(self.env.world_state['agents'], self.TERMINATING_EPISODE, MAP)
            # video_name_ext = helpers.get_video_name_ext(agent_types, self.TERMINATING_EPISODE, MAP)
            helpers.make_video_from_image_dir(
                self.experiment_folder,
                self.images_folder,
                video_name_ext
            )

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        
        while self.playing:
            if self.env.episode%50 == 0:
                self.results[str(self.env.episode)] = self.env.world_state['total_score']
            if self.env.episode >= self.TERMINATING_EPISODE:
                self.save_results()
                pg.display.quit()
                self.quit()
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.draw_grid()
        self.all_sprites.draw(self.screen)

        # Score Display (On top of sprites)
        font = pg.font.Font('freesansbold.ttf', 20)
        # current_score = self.env.world_state['explicit_rewards']['serve']
        current_score = self.env.world_state['total_score']
        current_order = self.env.world_state['order_count']
        episodes_left = self.TERMINATING_EPISODE - self.env.episode
        score = font.render(str(current_score), True, GREEN, SCOREBOARD_BG)
        order = font.render(str(current_order), True, GREEN, SCOREBOARD_BG)
        ep_countdown = font.render(str(episodes_left), True, GREEN, SCOREBOARD_BG)
        scoreRect = score.get_rect()
        orderRect = order.get_rect()
        countdownRect = ep_countdown.get_rect()
        scoreRect.center = (
            (SCOREBOARD_SCORE[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2
        )
        orderRect.center = (
            (SCOREBOARD_ORDERS[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2    
        )
        countdownRect.center = (
            (SCOREBOARD_TIMER[1]+1)*TILESIZE+TILESIZE//2,
            HEIGHT-TILESIZE//2    
        )
        self.screen.blit(score, scoreRect)
        self.screen.blit(order, orderRect)
        self.screen.blit(ep_countdown, countdownRect)
        pg.display.flip()

    def events(self):    
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYUP:
                if event.key in [
                    pg.K_LEFT, pg.K_RIGHT, pg.K_UP, pg.K_DOWN,
                    pg.K_COMMA, pg.K_PERIOD, pg.K_SLASH, pg.K_RSHIFT, pg.K_m,
                    pg.K_z, pg.K_x, pg.K_c, pg.K_v, pg.K_b, pg.K_n
                ]:
                    print(f'\nStart of episode {self.env.episode}')
                    goal_space = self.env.world_state['goal_space']
                    goal_info = self.env.world_state['goal_space_count']
                    print(f'Current goal space: \n{goal_space}\n')
                    print(f'Current goal info: \n{goal_info}\n')
                    print([agent.holding for agent in self.env.world_state['agents']])

                    player_action_validity, action_type, action_task, goal_id = self.env._check_action_validity(1, event.key)
                    player_object = [agent for agent in self.env.world_state['agents'] if agent.id == '1'][0]
                    best_goals = self.env.find_agents_best_goal()
                    print('found best goals')
                    print(best_goals)

                    if not player_action_validity:
                        best_goals[player_object] = [-1, {'steps': [8], 'rewards': -2}]
                    else:
                        action_mapping, reward_mapping = self._get_action_mapping_info(event.key)
                        if action_type == 'movement':
                            best_goals[player_object] = [-1, {'steps': [action_mapping], 'rewards': reward_mapping}]
                        else:
                            # its a task_action being taken
                            # goal_id = player_object._find_suitable_goal(action_mapping, action_task)
                            best_goals[player_object] = [goal_id, {'steps': action_task, 'rewards': reward_mapping}]

                    print('best goals')
                    print(best_goals)
                    print('before rolling out')
                    print([agent.location for agent in self.env.world_state['agents']])

                    self.rollout(best_goals, self.env.episode)
                    self.load_data()
                    self.update()
                    self.draw()

                    self.new(
                        self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                        self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
                    )
                    if event.key == pg.K_ESCAPE:
                        self.quit()
                    print(f'Just completed episode {self.env.episode}')
                    print([agent.location for agent in self.env.world_state['agents']])
                    print([agent.holding for agent in self.env.world_state['agents']])
                    self.env.update_episode()
                    # pg.image.save(self.screen, f'episodes/episode_{self.env.episode}.png')
                else:
                    print('Not valid key press')

    def _get_action_mapping_info(self, action):
        action_reward_mapping = {
            pg.K_LEFT: [0, -1],
            pg.K_RIGHT: [1, -1],
            pg.K_UP: [2, -1],
            pg.K_DOWN: [3, -1],
            pg.K_COMMA: [4, -2], # DIAG-UP-LEFT
            pg.K_PERIOD: [5, -2], # DIAG-UP-RIGHT
            pg.K_SLASH: [6, -2], # DIAG-DOWN-LEFT
            pg.K_RSHIFT: [7, -2], # DIAG-DOWN-RIGHT
            pg.K_m: [8, -2], # STAY
            pg.K_z: [9, 10],
            pg.K_x: [10, 30],
            pg.K_c: [11, 45],
            pg.K_v: [12, 50],
            pg.K_b: [13, 100],
            pg.K_n: [14, 0], # DROP
        }
        return action_reward_mapping[action][0], action_reward_mapping[action][1]
    


    def update_experiment_results(self, info_df):
        agent_1_info = [(agent.location, agent.last_action) for agent in self.env.world_state['agents'] if agent.id == '1'][0]
        agent_2_info = [(agent.location, agent.last_action) for agent in self.env.world_state['agents'] if agent.id == '2'][0]
        temp_info_df = info_df.append({
            'episode': self.env.episode,
            'player_coords': agent_1_info[0],
            'agent_coords': agent_2_info[0],
            'player_action': agent_1_info[1],
            'agent_action': agent_2_info[1],
            'available_orders': self.env.world_state['order_count'],
            'score': self.env.world_state['score'],
            'timer': self.TERMINATING_EPISODE - self.env.episode,
            'total_score': self.env.world_state['total_score']
        }, ignore_index=True)
        return temp_info_df

    def rollout(self, best_goals, horizon=50, save_path=None):
        """
        Save deep copy of current world state to be used as previous world state in next timestep.
        Deep copy to be used for inference calculations.

        Deep copy constructs a new compound object and then, recursively, inserts copies into it
        of the objects found in the original.
        This helps prevent any weird occurrences affecting the true world state.
        """
        temp_copy = copy.deepcopy(self.env.world_state)
        # only require historical world_state 1 timestep ago
        temp_copy['historical_world_state'] = {}
        self.env.world_state['historical_world_state'] = temp_copy
        
        reward_mapping = {}
        action_mapping = {}
        for agent in best_goals:
            reward_mapping[agent] = best_goals[agent][1]['rewards']
            action_mapping[agent] = (
                best_goals[agent][0],
                best_goals[agent][1]['steps'][0]
            )
            if type(best_goals[agent][1]['steps'][0]) == int:
                agent.last_action = str(best_goals[agent][1]['steps'][0])
            else:
                agent.last_action = best_goals[agent][1]['steps'][0][0]
        for agent in action_mapping:
            self.env.world_state['historical_actions'][agent.id] = [action_mapping[agent][1]]

        print(action_mapping)
        print('@rollout - Starting step function')
        print([agent.last_action for agent in self.env.world_state['agents']])
        final_rewards = self.env.step(action_mapping, reward_mapping)
        if self.RLTrainer:
            self.env.rl_trainer.receive_rewards(final_rewards)
        # print(f'Historical World State')
        # print(self.env.world_state['historical_world_state'])

        explicit_chop_rewards = self.env.world_state['explicit_rewards']['chop']
        explicit_cook_rewards = self.env.world_state['explicit_rewards']['cook']
        explicit_serve_rewards = self.env.world_state['explicit_rewards']['serve']
        print(f'Current EXPLICIT chop rewards: {explicit_chop_rewards}')
        print(f'Current EXPLICIT cook rewards: {explicit_cook_rewards}')
        print(f'Current EXPLICIT serve rewards: {explicit_serve_rewards}')

        if not self.is_simulation:
            self.info_df = self.update_experiment_results(self.info_df)
            pg.image.save(self.screen, self.images_folder+f'/episode_{self.env.episode}.png')

    def run_simulation(self, episodes:int=500):
        game_folder = os.path.dirname(__file__)
        simulations_folder = os.path.join(game_folder, 'simulations')
        video_folder = os.path.join(game_folder, 'videos')

        if self.RLTrainer:
            map_folder = os.path.join(*[self.RLTrainer.config.results_filepath, 'videos', MAP])
        else:
            map_folder = os.path.join(*[game_folder, 'videos', MAP])

        helpers.check_dir_exist(simulations_folder)
        helpers.check_dir_exist(video_folder)
        helpers.check_dir_exist(map_folder)

        helpers.clean_dir(simulations_folder)
        
        start_time = datetime.now()
        for episode in range(episodes):
            if episode == 0:
                self.new(
                    self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                    self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
                )

            print(f'================ Episode {episode} best goals ================')
            print(f'\nStart of episode {self.env.episode}')
            best_goals = self.env.find_agents_best_goal()
            goal_space = self.env.world_state['goal_space']
            goal_info = self.env.world_state['goal_space_count']
            print(f'Current goal space: \n{goal_space}\n')
            print(f'Current goal info: \n{goal_info}\n')
            print(f'Best goals')
            print(best_goals)
            print(f'Agent locations')
            print([agent.location for agent in self.env.world_state['agents']])

            self.rollout(best_goals, self.env.episode)
            self.load_data()
            self.update()
            self.draw()

            self.new(
                self.PLAYERS, self.TABLE_TOPS, self.INGREDIENTS, self.CHOPPING_BOARDS, self.PLATES, self.POTS,
                self.INGREDIENTS_STATION, self.SERVING_STATION, self.RETURN_STATION
            )

            print(f'Just completed episode {self.env.episode}')
            goal_space = self.env.world_state['goal_space']
            goal_info = self.env.world_state['goal_space_count']
            print(f'Current goal space: \n{goal_space}\n')
            print(f'Current goal info: \n{goal_info}\n')
            print([agent.location for agent in self.env.world_state['agents']])
            print([agent.holding for agent in self.env.world_state['agents']])
            self.env.update_episode()
            if self.env.rl_trainer:
                    if self.env.rl_trainer.episode_number % 100 == 0:
                        pg.image.save(self.screen, simulations_folder+f'/episode_{self.env.episode}.png')

            if self.env.episode == 0:
                self.results[str(self.env.episode)] = self.env.world_state['total_score']
            if (self.env.episode+1)%50 == 0:
                self.results[str(self.env.episode+1)] = self.env.world_state['total_score']
            if self.env.episode == self.TERMINATING_EPISODE:
                print('saving results')
                self.save_results()
                if self.RLTrainer:
                    self.env.rl_trainer.end_episode()

        print(f'======================= Done with simulation =======================')
        explicit_chop_rewards = self.env.world_state['explicit_rewards']['chop']
        explicit_cook_rewards = self.env.world_state['explicit_rewards']['cook']
        explicit_serve_rewards = self.env.world_state['explicit_rewards']['serve']
        print(f'Current EXPLICIT chop rewards: {explicit_chop_rewards}')
        print(f'Current EXPLICIT cook rewards: {explicit_cook_rewards}')
        print(f'Current EXPLICIT serve rewards: {explicit_serve_rewards}')

        end_time = datetime.now()
        experiment_runtime = (end_time - start_time).seconds
        experiment_runtime_min = experiment_runtime//60
        experiment_runtime_sec = experiment_runtime%60
        print(f'Simulation Experiment took {experiment_runtime_min} mins, {experiment_runtime_sec} secs to run.')

        # agent_types = [agent.is_inference_agent for agent in self.env.world_state['agents']]
        # video_name_ext = helpers.get_video_name_ext(agent_types, episodes, MAP)
        if self.env.rl_trainer:
            self.env.rl_trainer.log_explicit_results(explicit_chop_rewards, explicit_cook_rewards, explicit_serve_rewards)
            self.env.rl_trainer.logger.info(f'Simulation Experiment took {experiment_runtime_min} mins, {experiment_runtime_sec} secs to run.')
            if self.env.rl_trainer.episode_number % 1 == 0:
                self.env.rl_trainer.logger.info(f'Saving video at {map_folder}')
                video_name_ext = helpers.get_video_name_ext(self.env.world_state['agents'], 
                                                            self.env.rl_trainer.episode_number, MAP)
                helpers.make_video_from_image_dir(
                    map_folder,
                    simulations_folder,
                    video_name_ext,
                    0.5
                    )
        #sys.exit()

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

@click.command()
@click.option('--num_ai_agents', default=1, help='Number of AI agents to initialize')
@click.option('--num_rl_agents', default=0, help='Number of RL agents to initialize')
@click.option('--is_simulation', default=False, help='Run Simulation or Human Experiment?')
@click.option('--episodes', default=1, help='Number of episodes to run')
@click.option('--simulation_episodes', default=500, help='Number of timesteps to run')
@click.option('--is_tom', default=False, help='Is agent ToM-based?')
@click.option('--experiment_id', default='1', help='ID of the experiment')
@click.option('--gui', default=False, help='To use GUI of Pygame')
def main(num_ai_agents, num_rl_agents, is_simulation, episodes, simulation_episodes, is_tom, experiment_id, gui):
    # create the game object
    if not gui:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    if num_rl_agents > 0:
        from rl_trainer import PPOTrainer
        from rl_config import config
        RLTrainer = PPOTrainer(config)
        RLTrainer.logger.info(str(config.hyperparameters))
    else: RLTrainer = None

    for episode in range(episodes):
        time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        RLTrainer.logger.info("======== STARTING EPISODE {} AT {} =======".format(episode+1, time_now))
        g = Game(num_ai_agents, num_rl_agents, RLTrainer, is_simulation, simulation_episodes, is_tom, experiment_id)
    time_now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    RLTrainer.logger.info("============= A TOTAL OF {} COMPLETED at {} ==================".format(episodes, time_now))
    sys.exit()
    g.show_start_screen()
    while True:
        g.new(
            g.PLAYERS, g.TABLE_TOPS, g.INGREDIENTS, g.CHOPPING_BOARDS, g.PLATES, g.POTS,
            g.INGREDIENTS_STATION, g.SERVING_STATION, g.RETURN_STATION
        )
        g.run()
        g.show_go_screen()


if __name__ == "__main__":
    main()
