from typing import Any
from typing import Tuple
from typing import Optional

import click
import logging
from tkinter import *

import helpers
from create_agent import Agent
from config import MapConfig
from config import map_configurations


class EnvMap:
    """
    The Map Environment to simulate reinforcement learning algorithms experiments using GUI interface tkinter.
    """

    def __init__(
        self,
        config: MapConfig,
        input_file: Optional[str] = None,
        input_agent: Optional[Agent] = None
    ) -> None:
        self.config = config.MAP_CONFIG
        self.reset = self.config['reset']
        self.total_rewards = self.config['total_rewards']
        self.width = self.config['map_width']
        self.max_reward = 0
        self.cell_scores = {}

        # Load Agent object if present else create Agent
        self.load_agent(input_file, input_agent)

        self.grid = Canvas(master=self.initialize_grid(), width=self.config['x_axis']*self.width, height=self.config['y_axis']*self.width)
        self.create_grid()
        self.agent_label = self.create_agent_presence()
        self.grid.grid(row=0, column=0)

    def load_agent(self, input_file: Optional[str], input_agent: Optional[Agent]) -> None:
        logging.info(f'Loading Agent...')
        if input_file is None and input_agent is None:
            self.input_file = helpers.find_latest_file(self.config.PATHS.agent)
            self.agent = helpers.load_input_file(self.input_file)  # type: ignore
        elif input_file is None and input_agent is not None:
            self.agent = input_agent
        elif input_file is not None and input_agent is not None:
            logging.warning('Both an input file and an input Agent object were provided. Using the object.')
            self.agent = input_agent
        elif input_file is not None and input_agent is None:
            self.input_file = click.format_filename(input_file)
            self.agent = helpers.load_input_file(self.input_file)

    def initialize_grid(self) -> Any:
        self.map = Tk(screenName=self.config['screen_name'])
        self.map.bind('<Up>', self.move_up)
        self.map.bind('<Down>', self.move_down)
        self.map.bind('<Left>', self.move_left)
        self.map.bind('<Right>', self.move_right)

        return self.map

    def create_grid(self) -> None:
        for x_coord in range(self.config['x_axis']):
            for y_coord in range(self.config['y_axis']):
                self.grid.create_rectangle(
                    x_coord*self.width,
                    y_coord*self.width,
                    (x_coord+1)*self.width,
                    (y_coord+1)*self.width,
                    fill=self.config['map_colour'],
                    width=self.config['grid_width']
                )
                temp = {}
                for action in self.agent.config['actions']:
                    temp[action] = self.create_triangle(x_coord, y_coord, action)
                self.cell_scores[(x_coord, y_coord)] = temp

        for (x_coord, y_coord, colour, _) in self.config['rewards']:
            self.grid.create_rectangle(
                x_coord*self.width,
                y_coord*self.width,
                (x_coord+1)*self.width,
                (y_coord+1)*self.width,
                fill=colour,
                width=self.config['grid_width']
            )
        for (x_coord, y_coord) in self.config['obstacles']:
            self.grid.create_rectangle(
                x_coord*self.width,
                y_coord*self.width,
                (x_coord+1)*self.width,
                (y_coord+1)*self.width,
                fill=self.config['obstacle_colour'],
                width=self.config['grid_width']
            )

    def create_triangle(self, x_coord: int, y_coord: int, action: str) -> None:
        if action == self.agent.config['actions'][0]:
            return self.grid.create_polygon((x_coord+0.5-self.config['triangle_size'])*self.width, (y_coord+self.config['triangle_size'])*self.width,
                                            (x_coord+0.5+self.config['triangle_size'])*self.width, (y_coord+self.config['triangle_size'])*self.width,
                                            (x_coord+0.5)*self.config['triangle_size'], y_coord*self.width,
                                            fill=self.config['map_colour'], width=self.config['grid_width'])
        elif action == self.agent.config['actions'][1]:
            return self.grid.create_polygon((x_coord+0.5-self.config['triangle_size'])*self.width, (y_coord+1-self.config['triangle_size'])*self.width,
                                            (x_coord+0.5+self.config['triangle_size'])*self.width, (y_coord+1-self.config['triangle_size'])*self.width,
                                            (x_coord+0.5)*self.width, (y_coord+1)*self.width,
                                            fill=self.config['map_colour'], width=self.config['grid_width'])
        elif action == self.agent.config['actions'][2]:
            return self.grid.create_polygon((x_coord+self.config['triangle_size'])*self.width, (y_coord+0.5-self.config['triangle_size'])*self.width,
                                            (x_coord+self.config['triangle_size'])*self.width, (y_coord+0.5+self.config['triangle_size'])*self.width,
                                            x_coord*self.width, (y_coord+0.5)*self.width,
                                            fill=self.config['map_colour'], width=self.config['grid_width'])
        elif action == self.agent.config['actions'][3]:
            return self.grid.create_polygon((x_coord+1-self.config['triangle_size'])*self.width, (y_coord+0.5-self.config['triangle_size'])*self.width,
                                            (x_coord+1-self.config['triangle_size'])*self.width, (y_coord+0.5+self.config['triangle_size'])*self.width,
                                            (x_coord+1)*self.width, (y_coord+0.5)*self.width,
                                            fill=self.config['map_colour'], width=self.config['grid_width'])

    def create_agent_presence(self) -> Any:
        return self.grid.create_rectangle(
            self.agent.config['pos'][0]*self.width+self.width*2/10,
            self.agent.config['pos'][1]*self.width+self.width*2/10,
            self.agent.config['pos'][0]*self.width+self.width*8/10,
            self.agent.config['pos'][1]*self.width+self.width*8/10,
            fill=self.agent.config['colour'],
            width=self.agent.config['size'],
            tag=self.agent.config['name']
        )

    def set_cell_score(self, state: Tuple[int, int], action: str, val: int):
        
        triangle = self.cell_scores[state][action]
        green_dec = int(min(255, max(0, (val - self.config['cell_score_min']) * 255.0 / (self.config['cell_score_max'] - self.config['cell_score_min']))))
        green = hex(green_dec)[2:]
        red = hex(255-green_dec)[2:]
        if len(red) == 1:
            red += "0"
        if len(green) == 1:
            green += "0"
        color = "#" + red + green + "00"
        self.grid.itemconfigure(triangle, fill=color)

    def move_up(self) -> None:
        self.move(0, -1)

    def move_down(self) -> None:
        self.move(0, 1)

    def move_left(self) -> None:
        self.move(-1, 0)

    def move_right(self) -> None:
        self.move(1, 0)
    
    def move(self, dx: int, dy: int) -> Any:

        # After terminal state is achieved
        if self.reset == True:
            self.reset_map()

        new_x = self.agent.config['pos'][0] + dx
        new_y = self.agent.config['pos'][1] + dy
        self.total_rewards += self.agent.config['move_reward']
        if (new_x >= 0) and (new_x < self.config['x_axis']) and (new_y >= 0)\
            and (new_y < self.config['y_axis']) and not ((new_x, new_y) in self.config['obstacles']):
            self.grid.coords(
                self.agent_label,
                new_x*self.width+self.width*2/10,
                new_y*self.width+self.width*2/10,
                new_x*self.width+self.width*8/10,
                new_y*self.width+self.width*8/10
            )
            self.agent.config['pos'] = (new_x, new_y)
        for (x, y, _, reward) in self.config['rewards']:
            if new_x == x and new_y == y:
                self.total_rewards -= self.agent.config['move_reward']
                self.total_rewards += reward
                if self.total_rewards > 0:
                    logging.info(f'Success! Total Rewards: {self.total_rewards}')
                    if self.total_rewards > self.max_reward:
                        self.max_reward = self.total_rewards
                else:
                    logging.info(f'Fail! Total Rewards: {self.total_rewards}')
                self.reset = True
                return

    def reset_map(self):
        self.agent.config['pos'] = (0, self.config['y_axis']-1)
        self.total_rewards = 1
        self.reset = False
        self.grid.coords(
            self.agent_label,
            self.agent.config['pos'][0]*self.width+self.width*2/10,
            self.agent.config['pos'][1]*self.width+self.width*2/10,
            self.agent.config['pos'][0]*self.width+self.width*8/10,
            self.agent.config['pos'][1]*self.width+self.width*8/10
        )

    def start_grid(self):
        self.map.mainloop()


@click.command()
@click.option('--config', default='production', help='the deployment target')
@click.option('--input_file', default=None, type=click.Path(exists=True, dir_okay=False))
def main(config: str, input_file: str) -> None:
    logging.info('Creating environment map...')

    configuration = helpers.get_configuration(config, map_configurations)

    env_map = EnvMap(config=configuration, input_file=input_file)
    helpers.save(env_map)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
