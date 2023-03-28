import os
import pygame as pg

from settings import *
from overcooked_item_classes import *

# Set up Assets
game_folder = os.path.dirname(__file__)
assets_folder = os.path.join(game_folder, 'assets')


# Pure Player Class
# class Player(pg.sprite.Sprite):
#     def __init__(self, game, player_id, x, y, holding=None):
#         self.groups = game.all_sprites
#         pg.sprite.Sprite.__init__(self, self.groups)
#         self.game = game
#         self.image = pg.image.load(os.path.join(assets_folder, f'chef_{str(player_id)}.png')).convert()
#         self.image.set_colorkey(BACKGROUND_BLUE) # set background of image to transparent
#         self.rect = self.image.get_rect()
#         self.x = x
#         self.y = y

#     def move(self, dx=0, dy=0):
#         self.x += dx
#         self.y += dy

#     def update(self):
#         self.rect.x = self.x * TILESIZE
#         self.rect.y = self.y * TILESIZE

class Player(pg.sprite.Sprite):
    def __init__(self, game, player_id, x, y, holding=None):
        pg.sprite.Sprite.__init__(self, )
        self.game = game
        self.x = x
        self.y = y

        if holding:
            if isinstance(holding, Ingredient):
                self.image = pg.image.load(os.path.join(assets_folder,
                                                        f'chef_{str(player_id)}_holding_{holding.name}_{holding.state}.png')).convert()
            elif isinstance(holding, Plate):
                self.image = pg.image.load(
                    os.path.join(assets_folder, f'chef_{str(player_id)}_holding_plate_{holding.state}.png')).convert()
        else:
            self.image = pg.image.load(os.path.join(assets_folder, f'chef_{player_id}.png')).convert()
        self.image.set_colorkey(BACKGROUND_BLUE)  # set background of image to transparent
        self.rect = self.image.get_rect()

    def move(self, dx=0, dy=0):
        self.x += dx
        self.y += dy

    def update(self):
        self.rect.x = self.x * TILESIZE
        self.rect.y = self.y * TILESIZE


class TableTop(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.table_tops
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(BROWN)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Ingredients(pg.sprite.Sprite):
    def __init__(self, game, ingredient_name, ingredient_state, x, y):
        self.groups = game.all_sprites, game.ingredients
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.x = x
        self.y = y

        chopping_board_coords = [CHOPPING_BOARDS[chopping_board]['coords'] for chopping_board in CHOPPING_BOARDS]
        if (y, x) in chopping_board_coords:
            self.image = pg.image.load(
                os.path.join(assets_folder, f'chopping_board_{ingredient_name}_{ingredient_state}.png')).convert()
        else:
            self.image = pg.image.load(
                os.path.join(assets_folder, f'table_top_{ingredient_name}_{ingredient_state}.png')).convert()
        self.rect = self.image.get_rect()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class ChoppingBoardStation(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.chopping_boards
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'chopping_board_empty.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class PlateStation(pg.sprite.Sprite):
    def __init__(self, game, state, x, y):
        self.groups = game.all_sprites, game.plates
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'plate_{state}.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class PotStation(pg.sprite.Sprite):
    def __init__(self, game, pot_ingredients, x, y):
        self.groups = game.all_sprites, game.pot_stations
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.x = x
        self.y = y

        pot_ingredient_info = ""
        pot_ingredient_count = 0
        pot_ingredients = {ingredient_info[0]: ingredient_info[1] for ingredient_info in
                           sorted(pot_ingredients.items())}
        for ingredient in pot_ingredients:
            pot_ingredient_count += pot_ingredients[ingredient]
            if pot_ingredients[ingredient] > 0:
                pot_ingredient_info += ingredient + str(pot_ingredients[ingredient]) + '_'

        if pot_ingredient_count != 0:
            self.image = pg.image.load(os.path.join(assets_folder, f'pot_station_{pot_ingredient_info}.png')).convert()
            # self.image = pg.image.load(os.path.join(assets_folder, f'pot_station_{pot_ingredient}_{pot_ingredient_count}.png')).convert()
        else:
            self.image = pg.image.load(os.path.join(assets_folder, f'pot_station_empty.png')).convert()
        self.rect = self.image.get_rect()
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class IngredientStation(pg.sprite.Sprite):
    def __init__(self, game, ingredient, x, y):
        self.groups = game.all_sprites, game.ingredient_stations
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'ingredient_station_{ingredient}.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class ServingStation(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.serving_stations
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'serving_station.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class ReturnStation(pg.sprite.Sprite):
    def __init__(self, game, state, x, y):
        self.groups = game.all_sprites, game.return_station
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'return_station_{state}.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class ExtinguisherStation(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.extinguisher
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'extinguisher.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class TrashBin(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.trash_bin
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'trash_bin.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


# Scoreboard Sprites
class ScoreBoard(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.scoreboard
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(SCOREBOARD_BG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Score(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.score
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'scoreboard_score.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Orders(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.orders
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'scoreboard_orders.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Timer(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.orders
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.image.load(os.path.join(assets_folder, f'scoreboard_timer.png')).convert()
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE
