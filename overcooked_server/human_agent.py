from typing import List
from typing import Tuple

from collections import defaultdict
import random

from agent_configs import ACTIONS, REWARDS
from overcooked_item_classes import Ingredient, Dish, Plate
from settings import WALLS, INGREDIENTS_STATION, INGREDIENTS_INITIALIZATION, \
    RECIPES_INFO, RECIPE_ACTION_NAME, INGREDIENT_ACTION_NAME

class HumanAgent():
    def __init__(
        self,
        agent_id,
        location,
        holding=None,
        actions=ACTIONS,
        rewards=REWARDS,
        barriers=WALLS
    ) -> None:
        self.world_state = {}
        self.location = location
        self.id = agent_id
        self.holding = holding
        self.actions = actions
        self.rewards = rewards
        self.barriers = barriers

    def action_map(self, action_number: int) -> str:
        return ACTIONS[action_number]

    def return_valid_pos(self, new_pos):
        """
        Checks that the next pos is legal, if not return current pos
        """
        if new_pos in self.world_state['valid_cells']:
            return new_pos

        # you can't walk through walls nor another agent
        return self.location
    
    def update_agent_pos(self, new_coords: List[int]) -> None:
        self.location = new_coords
    
    def get_recipe_ingredient_count(self, recipe, ingredient):
        recipe_ingredient_count = RECIPES_INFO[recipe][ingredient]
        
        return recipe_ingredient_count

    def get_recipe_dish(self, ingredient):
        recipe_dish = None
        for recipe in RECIPES_INFO:
            if RECIPES_INFO[recipe]['ingredient'] == ingredient:
                recipe_dish = recipe
        
        return recipe_dish

    def get_ingredient_name(self, task_id):
        ingredient_name = None
        for ingredient in INGREDIENT_ACTION_NAME:
            if task_id in INGREDIENT_ACTION_NAME[ingredient]:
                ingredient_name = ingredient
        return ingredient_name

    def get_recipe_name(self, task_id):
        recipe_name = None
        for recipe in RECIPE_ACTION_NAME:
            if task_id in RECIPE_ACTION_NAME[recipe]:
                recipe_name = recipe
        return recipe_name

    def complete_cooking_check(self, recipe, ingredient_counts):
        return RECIPES_INFO[recipe] == ingredient_counts
    
    def get_general_goal_id(self, recipe, action):
        return RECIPES_ACTION_MAPPING[recipe]['general'][action]

    def pick(self, task_id:int, pick_info) -> None:
        print('human@pick')
        is_new = pick_info['is_new']
        is_last = pick_info['is_last']
        pick_type = pick_info['pick_type']
        task_coord = pick_info['task_coord']

        if pick_type == 'ingredient':
            ingredient_name = self.get_ingredient_name(task_id)
            if is_new:
                self.world_state['goal_space_count'][task_id] -= 1
                self.world_state['goal_space_count'][task_id+1] += 1
                state = 'unchopped'
                new_ingredient = Ingredient(
                    ingredient_name,
                    state,
                    'ingredient',
                    task_coord
                )
                new_ingredient.location = tuple(self.location)
                self.holding = new_ingredient
 
            if is_last:
                try:
                    self.world_state['goal_space'][task_id].pop(0)
                    self.world_state['goal_space'][task_id+1].append({
                        'state': 'unchopped',
                        'ingredient': ingredient_name
                    })
                except IndexError:
                    print(f'Picking beyond required amount')
                    self.world_state['goal_space'][task_id+1].append({
                        'state': 'unchopped',
                        'ingredient': ingredient_name
                    })
            else:
                print(f'IndexError when trying to pop goal_space - PICK')
                ingredient_location = [ingredient.location for ingredient in self.world_state['ingredients']]

                if task_coord in ingredient_location:
                    # if another agent took it already and is now missing; do nothing
                    old_ingredient = [
                        ingredient for ingredient in self.world_state['ingredients'] if ingredient.location == task_coord
                    ][0]

                    # check if ingredient to be picked is on chopping_board
                    cb = [chopping_board for chopping_board in self.world_state['chopping_board'] if chopping_board.location == old_ingredient.location]
                    if len(cb) == 1:
                        cb[0].state = 'empty'

                    # update ingredient to be held by agent and in agent's current position
                    old_ingredient.location = tuple(self.location)
                    self.holding = old_ingredient

                    # Need to remove from world state after agent has picked it
                    self.world_state['ingredients'] = [ingredient for ingredient in self.world_state['ingredients'] if id(ingredient) != id(old_ingredient)]
        
        elif pick_type == 'plate':
            plate_location = [plate.location for plate in self.world_state['plate']]
            if task_coord in plate_location:
                # if another agent took it already and is now missing; do nothing
                old_plate = [
                    plate for plate in self.world_state['plate'] if plate.location == task_coord
                ][0]

                # update plate to be held by agent and in agent's current position
                old_plate.location = tuple(self.location)
                self.holding = old_plate

                # Need to remove from world state after agent has picked it
                self.world_state['plate'] = [plate for plate in self.world_state['plate'] if id(plate) != id(old_plate)]

    def find_random_empty_cell(self) -> Tuple[int,int]:
        all_valid_surrounding_cells = []
        surrounding_cells_xy = [
            [-1,0], [0,1], [1,0], [0,-1]
        ]

        temp_valid_cells = self.world_state['valid_item_cells'].copy()
        for surrounding_cell_xy in surrounding_cells_xy:
            surrounding_cell = [sum(x) for x in zip(self.location, surrounding_cell_xy)]
            if tuple(surrounding_cell) in temp_valid_cells:
                all_valid_surrounding_cells.append(tuple(surrounding_cell))

        return random.choice(all_valid_surrounding_cells)

    def drop(self, task_id: int) -> None:
        """
        This action assumes agent is currently holding an item.
        Prerequisite
        ------------
        - Drop item at where agent is currently

        TO IMPROVE:
        - Prevent stacking of item
        - Dropping item blocks grid cell
        """
        print('human@drop - Drop item in-hand')
        random_empty_cell = self.find_random_empty_cell()

        if type(self.holding) == Ingredient:
            holding_ingredient = self.holding
            holding_ingredient.location = random_empty_cell
            self.world_state['ingredients'].append(holding_ingredient)
        # For now, just plate
        elif type(self.holding) == Plate:
            holding_plate = self.holding
            holding_plate.location = random_empty_cell
            self.world_state['plate'].append(holding_plate)
        self.holding = None

    def chop(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('human@chop')
        print(task_id)
        ingredient_name = self.get_ingredient_name(task_id)
        recipe_name = self.get_recipe_name(task_id)
        self.world_state['explicit_rewards']['chop'] += 1
        self.world_state['goal_space_count'][task_id] -= 1
        self.world_state['goal_space_count'][task_id+1] += 1
        if is_last:
            print('base_agent@chop - Remove chopping task')
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['goal_space'][task_id+1].append({
                'state': 'chopped',
                'ingredient': ingredient_name,
                'recipe': recipe_name
            })

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand
        holding_ingredient.location = task_coord

        # agent drops ingredient to chopping board
        self.holding = None
        holding_ingredient.state = 'chopped'
        self.world_state['ingredients'].append(holding_ingredient)
        used_chopping_board = [board for board in self.world_state['chopping_board'] if board.location == task_coord][0]

        # update chopping board to be 'taken'
        used_chopping_board.state = 'taken'
        
    def cook(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('human@cook')
        ingredient_name = self.get_ingredient_name(task_id)
        dish = self.get_recipe_name(task_id)
        ingredient_count = self.get_recipe_ingredient_count(dish, ingredient_name)

        # Find the chosen pot - useful in maps with more than 1 pot
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        if pot.ingredient_count[ingredient_name] == ingredient_count:
            # Maxed out already
            pass
        else:
            self.world_state['explicit_rewards']['cook'] += 1
            self.world_state['goal_space_count'][task_id] -= 1
            self.world_state['goal_space'][task_id].pop(0)

            holding_ingredient = self.holding
            # only update location after reaching, since ingredient is in hand
            holding_ingredient.location = task_coord

            # agent drops ingredient to pot
            pot.ingredient_count[ingredient_name] += 1
            pot.is_empty = False

            # remove ingredient from world state since used for cooking
            for idx, ingredient in enumerate(self.world_state['ingredients']):
                if id(ingredient) == id(holding_ingredient):
                    del self.world_state['ingredients'][idx]
                    break
            # remove ingredient from agent's hand since no longer holding
            self.holding = None

            # if pot.ingredient_count == ingredient_count:
            if self.complete_cooking_check(dish, pot.ingredient_count):
                print('human@cook - Add completed dish to pot')

                # Create new Dish Class object
                new_dish = Dish(dish, pot.location)
                self.world_state['cooked_dish'].append(new_dish)
                pot.dish = dish

                # Add Scoop to Goal Space
                self.world_state['goal_space_count'][task_id+1] += 1
                self.world_state['goal_space'][task_id+1].append({
                    'state': 'empty',
                    'ingredient': ingredient_name
                })

    def scoop(self, task_id: int, scoop_info):
        print('human@scoop')
        ingredient_name = self.get_ingredient_name(task_id)
        is_last = scoop_info['is_last']
        task_coord = scoop_info['task_coord']
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

         # Empty the pot as well
        pot.ingredient = None
        pot.ingredient_count = 0
        pot.is_empty = True
        pot.dish = None

        dish_to_plate = [dish for dish in self.world_state['cooked_dish'] if dish.location == task_coord][0]
        # let the plate which agent is holding, hold the completed dish
        self.world_state['cooked_dish_count'][dish_to_plate.name] -= 1
        self.holding.dish = dish_to_plate
        self.holding.state = 'plated'

        # remove dish from world state since used for plating
        for idx, cooked_dish in enumerate(self.world_state['cooked_dish']):
            if id(cooked_dish) == id(dish_to_plate):
                del self.world_state['cooked_dish'][idx]
                break

        self.world_state['goal_space_count'][task_id] -= 1
        if is_last:
            print('human@scoop - Remove scooping task')
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['goal_space_count'][task_id+1] += 1
            self.world_state['goal_space'][task_id+1].append({
                'state': 'plated',
                'ingredient': ingredient_name
            })
        
    def serve(self, task_id: int, serve_info):
        print('human@serve')
        self.world_state['explicit_rewards']['serve'] += 1
        is_last = serve_info['is_last']

        # plate returns to return point (in clean form for now)
        self.holding.dish = None
        self.holding.state = 'empty'
        self.holding.location = self.world_state['return_counter']
        self.world_state['plate'].append(self.holding)

        # remove dish from plate
        self.holding = None

        # remove order from TaskList
        if is_last:
            print('human@serve - Remove serve task')
            self.world_state['goal_space_count'][task_id] -= 1
            self.world_state['goal_space'][task_id].pop(0)
            self.world_state['order_count'] -= 1
