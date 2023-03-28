import os
import pygame as pg

from moral_line_settings import *
from random import randrange

# Set up Assets
game_folder = os.path.dirname(__file__)
assets_folder = os.path.join(game_folder, 'new_assets')

# set up a generic character class to reduce redundant code.
class Character(pg.sprite.Sprite):
    def __init__(self, game, x, y, path):
        self.groups = [game.all_sprites]
        pg.sprite.Sprite.__init__(self, game.all_sprites)
        self.game = game
        self.image = pg.image.load(path)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Player(Character):
    def __init__(self, game, player_id, x, y, holding=0, orientation='D', completed=False):
        self.id = player_id
        self.holding = holding
        self.completed = completed
        self.orientation = orientation
        path = self.get_path()
        Character.__init__(self, game, x, y, path)
        self.add(game.movable)

    # gets the required sprite image path
    def get_path(self):
        if self.holding == 0:
            if self.completed:
                path = os.path.join(assets_folder, "Characters/", str(self.id), "empty/", f'{self.orientation}.png')
            else:
                path = os.path.join(assets_folder, "Characters/", str(self.id), "idle/", f'{self.orientation}.png')
        elif self.holding == 1:
            path = os.path.join(assets_folder, "Characters/", str(self.id), "water-1-3/", f'{self.orientation}.png')
        elif self.holding == 2:
            path = os.path.join(assets_folder, "Characters/", str(self.id), "water-2-3/", f'{self.orientation}.png')
        elif self.holding == 3:
            path = os.path.join(assets_folder, "Characters/", str(self.id), "water-3-3/", f'{self.orientation}.png')    

        return path

    # determines the character orientation
    def check_orientation(self, x, y):
        # determines the orientation based on location
        if x == self.x:
            print(y - self.y)
            if y == self.y:
                self.update_image()
            elif y > self.y:
                self.orientation = 'D'
            else:
                self.orientation = 'U'
        elif x > self.x:
            self.orientation = 'R'
        else:
            self.orientation = 'L'
        self.update_image()

    # updates what the character is holding
    def outdated_update_holding(self):
        if self.holding:
            self.holding = False
            self.completed = True
            self.update_image()
            return True
        else:
            if self.completed:
                self.completed = False
            else:
                self.holding = True
            self.update_image()
            return False

    # updates what the character is holding
    def update_holding(self):
        if self.holding == 0 or self.holding == 1 or self.holding == 2:
            self.holding += 1
            self.completed = False
            self.update_image()
            return False
        else: #holding == 3
            self.holding = 0
            self.completed = True
            self.update_image()
            return True

    # updates the sprites image
    def update_image(self):
        path = self.get_path()
        self.image = pg.image.load(path)
        self.add(self.game.movable)

    # moves the character
    def move(self, dx=0, dy=0):
        self.check_orientation(self.x + dx, self.y + dx)
        self.x += dx
        self.y += dy

    # moves the character to a specific cell
    def move_to(self, x, y):
        self.check_orientation(x, y)
        self.x = x
        self.y = y
        self.rect = self.image.get_rect()

    # updates player rect
    def update(self):
        self.rect.x = self.x * TILESIZE
        self.rect.y = self.y * TILESIZE


# blocking objects below
class Cactus(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'Cactus/', f'C{str(randrange(1, 4))}.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.cactus, game.objects)


class PalmTrees(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'Palm_Trees/', f'PT{str(randrange(1, 4))}.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.palm_trees, game.objects)


class Rocks(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'Rocks/', f'R{str(randrange(1, 4))}.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.rocks, game.objects)


# functional objects below
class RiverWater(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'River_Water/BOX/RW-BOX-B2-C.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.river_water, game.objects)


class StillWater(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'River_Water/BOX/RW-BOX-B2-C.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.still_water, game.objects)


class Well(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'Well/W1.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.well, game.objects)


class Barrel(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, 'Barrel/1.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.barrel, game.objects)


# Scoreboard Sprites
class ScoreBoard(Character):
    def __init__(self, game, x, y):
        self.groups = game.all_sprites, game.scoreboard
        pg.sprite.Sprite.__init__(self, game.all_sprites, game.scoreboard, game.objects)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(SCOREBOARD_BG)
        self.rect = self.image.get_rect()
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Score(Character):
    def __init__(self, game, x, y):
        path = os.path.join(assets_folder, f'scoreboard_score.png')
        Character.__init__(self, game, x, y, path)
        self.add(game.score)

# class Timer(pg.sprite.Sprite):
#     def __init__(self, game, x, y):
#         self.groups = game.all_sprites, game.orders
#         pg.sprite.Sprite.__init__(self, self.groups)
#         self.game = game
#         self.image = pg.image.load(os.path.join(assets_folder, f'scoreboard_timer.png')).convert()
#         self.rect = self.image.get_rect()
#         self.x = x
#         self.y = y
#         self.rect.x = x * TILESIZE
#         self.rect.y = y * TILESIZE
