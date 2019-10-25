from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pygame as pg
import sys
from overcooked_pygame.settings import *
from overcooked_pygame.sprites import *

class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

        self.episode = 0

    def load_data(self):
        pass

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
        extinguisher: Tuple[int,int],
        trash_bin: Tuple[int,int]
    ) -> None:
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.table_tops = pg.sprite.Group()
        self.ingredients = pg.sprite.Group()
        self.chopping_boards = pg.sprite.Group()
        self.plates = pg.sprite.Group()
        self.pot_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.ingredient_stations = pg.sprite.Group()
        self.serving_stations = pg.sprite.Group()
        self.return_station = pg.sprite.Group()
        self.extinguisher = pg.sprite.Group()
        self.trash_bin = pg.sprite.Group()
        self.player_count = len(players)


        for idx in range(1, self.player_count+1):
            idx = str(idx)
            if idx == '1':
                self.player_1 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
            elif idx == '2':
                self.player_2 = Player(self, idx, players[idx]['coords'][1], players[idx]['coords'][0], players[idx]['holding'])
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
            pot_ingredient = val['ingredient']
            pot_ingredient_count = val['ingredient_count']
            pot_coord = val['coords']
            PotStation(self, pot_ingredient, pot_ingredient_count, pot_coord[1], pot_coord[0])
        for key, val in ingredient_stations.items():
            ingredient = key
            chopping_board_coord = val
            IngredientStation(self, ingredient, chopping_board_coord[1], chopping_board_coord[0])
        for serving_station_coord in serving_stations:
            ServingStation(self, serving_station_coord[1], serving_station_coord[0])
            
        ReturnStation(self, return_station['state'], return_station['coords'][1], return_station['coords'][0])
        ExtinguisherStation(self, extinguisher[1], extinguisher[0])
        TrashBin(self, trash_bin[1], trash_bin[0])


    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        
        while self.playing:
            if self.episode > 500:
                self.playing = False
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
        pg.display.flip()

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_LEFT:
                    print(self.player_1)
                    self.player_1.move(dx=-1)
                if event.key == pg.K_RIGHT:
                    self.player_1.move(dx=1)
                if event.key == pg.K_UP:
                    self.player_1.move(dy=-1)
                if event.key == pg.K_DOWN:
                    self.player_1.move(dy=1)
                self.episode += 1
                pg.image.save(self.screen, f'./episodes/episode_{self.episode}.png')

    def show_start_screen(self):
        pass

    def show_go_screen(self):
        pass

def main():
    # create the game object
    g = Game()
    g.show_start_screen()
    while True:
        g.new(
            PLAYERS, TABLE_TOPS, INGREDIENTS, CHOPPING_BOARDS, PLATES, POTS, INGREDIENTS_STATION,
            SERVING_STATION, RETURN_STATION, EXTINGUISHER, TRASH_BIN
        )
        g.run()
        g.show_go_screen()


if __name__ == "__main__":
    main()