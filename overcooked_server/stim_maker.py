# program to create the stimuli for the moral lines experiments
import pygame as pygame
from moral_line_sprites import *
from moral_line_settings import *
import moral_line_helpers as helpers

import click
import sys

import os  
import shutil

from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

# first, get the background and physical sprites of the environment on the gridworld
# then, encode an array with coordinates indicating the trajectory of each agent from start to finish

# draw sprites at the coordinates,
# this part will require drawing the appropriate facing agent based on movement from the last coordinate
# at certain parts, the sprites need to change

# save images at each time step, stitch them into a video

class Game:
    def __init__(self, num_ai_agents: int = 8, experiment_id: str = '1') -> None:
        pg.init()
        self.current_episode = 1
        self.simulation_episodes = len(EP_DATA)
        self.episode_data = EP_DATA
        self.current_score = 0

        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()

        self.experiment_id = experiment_id

        self.players = {}
        self.player = [None] * 8  

        self.all_sprites = pg.sprite.Group()
        self.cactus = pg.sprite.Group()
        self.movable = pg.sprite.Group()
        self.objects = pg.sprite.Group()
        self.palm_trees = pg.sprite.Group()
        self.rocks = pg.sprite.Group()
        self.river_water = pg.sprite.Group()
        self.still_water = pg.sprite.Group()
        self.well = pg.sprite.Group()
        self.barrel = pg.sprite.Group()

        self.score = pg.sprite.Group()
        self.timer = pg.sprite.Group()
        self.scoreboard = pg.sprite.Group()

    # loads in and creates players
    def load_players(self):
        self.players = selected_map.EP_DATA[self.current_episode]

        #randomizing player IDs
        #mapping = random.sample(range(1, 9), 8)

        for idx in range(1, len(self.players)+1):
            self.player[idx-1] = Player(self, idx, self.players[idx]['coords'][1],
                                          self.players[idx]['coords'][0],
                                          self.players[idx]['holding'],
                                          self.players[idx]['orientation'],
                                          self.players[idx]['completed'])

        # for player in self.player:
        #     self.movable.add(player)

    # updates all player locations and holding states
    def update_players(self, data, index):
        self.movable.empty()
        for i, player in enumerate(self.player):
            if type(player) == int:
                return
            player.move_to(data[i + 1]['coords'][1], data[i + 1]['coords'][0])
            if index + 1 in selected_map.HOLDING_TIMES[i]:
                if player.update_holding():
                    self.current_score += 1

            self.movable.add(player)

    # sorta obsolete, can just call load_players method.
    def load_data(self):
        # made into a method as its used two places
        self.load_players()

    def new(
            self,
            players: Dict[int, Tuple[int, int]],
            cactus: List[Tuple[int, int]] = CACTUS,
            palm_trees: List[Tuple[int, int]] = PALM_TREES,
            rocks: List[Tuple[int, int]] = ROCKS,
            river_water: List[Tuple[int, int]] = RIVER_WATER,
            still_water: List[Tuple[int, int]] = STILL_WATER,
            well: List[Tuple[int, int]] = WELL,
            barrel: List[Tuple[int, int]] = BARREL,
            score: Tuple[int, int] = SCOREBOARD_SCORE,
            timer: Tuple[int, int] = SCOREBOARD_TIMER,
            scoreboard: List[Tuple[int, int]] = SCOREBOARD
    ) -> None:
        # initialize all variables and do all the setup for a new game
        # changed to empty rather than creating a new. more efficient.
        self.objects.empty()
        self.movable.empty()
        self.cactus.empty()
        self.palm_trees.empty()
        self.rocks.empty()
        self.river_water.empty()
        self.still_water.empty()
        self.well.empty()
        self.barrel.empty()

        self.score.empty()
        self.timer.empty()
        self.scoreboard.empty()

        # replaced by method
        self.load_players()

        for cactus_coord in cactus:
            Cactus(self, cactus_coord[1], cactus_coord[0])
        for palm_trees_coord in palm_trees:
            PalmTrees(self, palm_trees_coord[1], palm_trees_coord[0])
        for rocks_coord in rocks:
            Rocks(self, rocks_coord[1], rocks_coord[0])
        for river_water_coord in river_water:
            RiverWater(self, river_water_coord[1], river_water_coord[0])
        for still_water_coord in still_water:
            StillWater(self, still_water_coord[1], still_water_coord[0])
        for well_coord in well:
            Well(self, well_coord[1], well_coord[0])
        for barrel_coord in barrel:
            Barrel(self, barrel_coord[1], barrel_coord[0])


    def save_results(self):

        # can not use undefined variables. changed to a hard coding.
        helpers.make_video_from_image_dir(
            "../maps",
            "../simulations",
            MAP
        )

    def quit(self):
        pg.quit()
        sys.exit()

    # def update(self):
    #     # update portion of the game loop
    #     self.all_sprites.update()

    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))
            if y == HEIGHT - TILESIZE:
                pg.draw.rect(self.screen, LIGHTGREY, (0, y, WIDTH, HEIGHT))

    def draw(self):
        self.screen.fill(BGCOLOR)
        self.draw_grid()
        self.all_sprites.update()
        self.objects.draw(self.screen)
        self.movable.draw(self.screen)
        # Score Display (On top of sprites)
        font = pg.font.Font('freesansbold.ttf', 20)
        score = font.render(f"Score: {self.current_score}", True, GREEN, LIGHTGREY)

        # current episode print.
        curr_ep = font.render(f"Current Episode: {self.current_episode}", True, GREEN, LIGHTGREY)
        scoreRect = score.get_rect()
        curr_ep_rect = curr_ep.get_rect()
        scoreRect.center = (
            (SCOREBOARD_SCORE[1] + 1) * TILESIZE + TILESIZE // 2,
            HEIGHT - TILESIZE // 2
        )
        # curr ep is TILESIZE * 4 items to the right of score.
        curr_ep_rect.center = (
            (SCOREBOARD_SCORE[1] + 1) * TILESIZE + TILESIZE * 4,
            HEIGHT - TILESIZE // 2
        )
        self.screen.blit(score, scoreRect)
        self.screen.blit(curr_ep, curr_ep_rect)
        pg.display.flip()

    def run_simulation(self):

        game_folder = os.path.dirname(__file__)
        simulations_folder = os.path.join(game_folder, 'simulations')
        video_folder = os.path.join(game_folder, 'videos')
        map_folder = os.path.join(*[game_folder, 'videos', MAP])

        helpers.check_dir_exist(simulations_folder)
        helpers.check_dir_exist(video_folder)
        helpers.check_dir_exist(map_folder)

        helpers.clean_dir(simulations_folder)

        for index, episode in enumerate(self.episode_data):
            if index == 0:
                self.new(
                    self.players, CACTUS, PALM_TREES,
                    ROCKS, RIVER_WATER, STILL_WATER, WELL, BARREL
                )
            #self.update()
            self.draw()
            # update method call caused all sorts of repeat characters.
            # had t- change order, and detect when the last episode is
            if len(self.episode_data) >= index + 2:
                PLAYERS = selected_map.EP_DATA[index + 2]
                self.update_players(PLAYERS, index)

            pg.image.save(self.screen, simulations_folder + f'/episode_{str(index+1)}.png')
            #index += 1
            self.current_episode += 1

        helpers.make_video_from_image_dir(
            map_folder,
            simulations_folder,
            MAP
        )

        sys.exit()


@click.command()
@click.option('--num_ai_agents', default=8, help='Number of AI agents to initialize')
@click.option('--experiment_id', default='1', help='ID of the experiment')
def main(num_ai_agents, experiment_id):
    # create the game object
    g = Game(num_ai_agents, experiment_id)
    #g.new(g.players, CACTUS, PALM_TREES, ROCKS, RIVER_WATER, STILL_WATER, WELL, BARREL)
    #g.load_data()
    g.run_simulation()


if __name__ == "__main__":
    main()
