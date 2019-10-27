from typing import List
from typing import Tuple

from collections import defaultdict
import random

from agent import BaseAgent
from agent_configs import BARRIERS
from overcooked_classes import Ingredient, Dish, Plate, TaskList
from agent_configs import ACTIONS, REWARDS, INGREDIENTS_STATION, INGREDIENTS_INITIALIZATION, RECIPES_INGREDIENTS_COUNT, RECIPES_SERVE_TASK

class HumanAgent(BaseAgent):
    def __init__(
        self,
        agent_id,
        location,
        holding=None,
        actions=ACTIONS,
        rewards=REWARDS,
        barriers=BARRIERS
    ) -> None:
        super().__init__(agent_id, location)
        self.world_state = {}
        self.id = agent_id
        self.holding = holding
        self.actions = actions
        self.rewards = rewards
        self.barriers = barriers
    
    def _find_suitable_goal(self, action_number, task_info):
        print(f'Finding suitable goal now')
        action = self.action_map(action_number)
        # if more than 1 same goal (randomize it)
        # use goal.id if want to refactor
        print('debug goal est')
        print(action)
        print([id(goal) for goal in self.world_state['goal_space'] if goal.head.task == action.lower()])
        print([goal.head.task for goal in self.world_state['goal_space']])
        all_goals = [id(goal) for goal in self.world_state['goal_space'] if goal.head.task == action.lower()]

        try:
            if all_goals:
                goal_id = random.choice(all_goals)
            else:
                goal_id = -1
        except IndexError:
            # No goals
            goal_id = -1
        
        if goal_id == -1:
            if action == 'SCOOP':
                print(f'Edge case for SCOOP')
                # Edge case: human picks up plate before its available for serving
                temp_action = 'PLATE'
                all_goals = [id(goal) for goal in self.world_state['goal_space'] if goal.head.task == temp_action.lower()]

                task_id = random.choice(all_goals)

                task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
                task.head = task.head.next
                goal_id = task_id
        print(f'Human Agent going for goal {goal_id}.')
        return goal_id

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

    def pick(self, task_id: int, pick_info) -> None:
        """
        This action assumes agent has already done A* search and decided which goal state to achieve.
        Prerequisite
        ------------
        - Item object passed to argument should be item instance
        - At grid with accessibility to item [GO-TO]

        pick_info
        ---------
        'is_new': True/False - whether to take from box and create new Ingredient object or not
        'is_last': True/False - whether this action is last of everything - to update TaskList
        'task_coord': coordinate to perform task on eg. pick ingredient at task_coord
        'end_coord': coordinate where agent must be at before performing task at task_coord

        TO-CONSIDER: 
        Do we need to check if agent is currently holding something?
        Do we need to set item coord to agent coord when the item is picked up?
        """
        is_new = pick_info['is_new']
        is_last = pick_info['is_last']
        pick_type = pick_info['pick_type']
        task_coord = pick_info['task_coord']

        if task_id != -1:
            try:
                print('entered try')
                task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]

                if pick_type == 'ingredient':
                    if is_new:
                        print('base_agent@pick - is_new - create new ingredient')
                        ingredient_name = task.ingredient
                        state = task.head.state
                        new_ingredient = Ingredient(
                            ingredient_name,
                            state,
                            'ingredient',
                            INGREDIENTS_INITIALIZATION[ingredient_name]
                        )
                        new_ingredient.location = tuple(self.location)
                        new_ingredient.state = 'unchopped'
                        self.holding = new_ingredient

                    if is_last and task.head.task == 'pick':
                        print('@base_agent@pick - last picked ingredient')
                        task.head = task.head.next
                    else:
                        print('base_agent@pick - picking not is_new ingredient')
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

                            if is_last:
                                task.head = task.head.next
                
                elif pick_type == 'plate':
                    print('base_agent@pick - plate')
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

                        # Edge case: both agents pick a plate at the same time (prevent task from being updated twice)
                        if is_last and task.head.task == 'plate':
                            task.head = task.head.next
            except IndexError:
                # Edge-Case: AgentL about to serve, AgentR trying to pick plate for the same goal that is going to be removed
                print('entere index error')
                pass
        else:
            if pick_type == 'ingredient':
                if is_new:
                    print('base_agent@pick - is_new - create new ingredient')
                    ingredient_names = [ingredient for ingredient in INGREDIENTS_STATION]
                    ingredient_locs = [INGREDIENTS_STATION[ingredient] for ingredient in INGREDIENTS_STATION]
                    ingredient_name_idx = ingredient_locs.index(task_coord)
                    ingredient_name = ingredient_names[ingredient_name_idx]
                    new_ingredient = Ingredient(
                        ingredient_name,
                        'unchopped',
                        'ingredient',
                        INGREDIENTS_INITIALIZATION[ingredient_name]
                    )
                    new_ingredient.location = tuple(self.location)
                    self.holding = new_ingredient
                else:
                    print('base_agent@pick - picking not is_new ingredient')
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
                print('base_agent@pick - plate')
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
        
    def chop(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('base_agent@chop')
        self.world_state['explicit_rewards']['chop'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand (?)
        holding_ingredient.location = task_coord

        # agent drops ingredient to chopping board
        self.holding = None
        holding_ingredient.state = 'chopped'
        self.world_state['ingredients'].append(holding_ingredient)
        used_chopping_board = [board for board in self.world_state['chopping_board'] if board.location == task_coord][0]

        # update chopping board to be 'taken'
        used_chopping_board.state = 'taken'

        # Edge case: task.head.next can point to None if agentR reaches here after agentL already in midst of performing next task
        if is_last and task.head.next:
            print('base_agent@chop - Remove chopping task')
            task.head = task.head.next
        
    def cook(self, task_id: int, is_last: bool, task_coord: Tuple[int, int]):
        print('base_agent@cook')
        self.world_state['explicit_rewards']['cook'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        dish = task.dish
        ingredient = task.head.ingredient
        ingredient_count = RECIPES_INGREDIENTS_COUNT[dish]

        # Find the chosen pot - useful in maps with more than 1 pot
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        holding_ingredient = self.holding
        # only update location after reaching, since ingredient is in hand (?)
        holding_ingredient.location = task_coord

        # agent drops ingredient to pot
        pot.ingredient_count[ingredient] += 1

        # remove ingredient from world state since used for cooking
        for idx, ingredient in enumerate(self.world_state['ingredients']):
            if id(ingredient) == id(holding_ingredient):
                del self.world_state['ingredients'][idx]
                break
        # remove ingredient from agent's hand since no longer holding
        self.holding = None

        print('base_agent@cook - check for dish ingredients\' prep')
        if pot.ingredient_count == ingredient_count:
            print('base_agent@cook - Add completed dish to pot')

            # Create new Dish Class object
            new_dish = Dish(dish, pot.location)
            self.world_state['cooked_dish'].append(new_dish)
            self.world_state['task_id_count'] += 1
            self.world_state['goal_space'].append(TaskList(dish, RECIPES_SERVE_TASK[dish], dish, self.world_state['task_id_count']))
            self.world_state['cooked_dish_count'][dish] += 1
            for task_list in self.world_state['goal_space']:
                if task_list.id not in self.world_state['task_id_mappings']:
                    self.world_state['task_id_mappings'][task_list.id] = id(task_list)
            pot.dish = dish

        if is_last:
            print('base_agent@cook - Remove cooking task')
            task.head = task.head.next
            if task.head == None:
                for idx, task_no in enumerate(self.world_state['goal_space']):
                    if id(task_no) == task_id:
                        del self.world_state['goal_space'][idx]
                        break
    
    def scoop(self, task_id: int, scoop_info):
        """
        Current logic flaw
        ------------------
        Only works for dishes with 1 ingredient
        """
        print('base_agent@scoop')

        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        is_last = scoop_info['is_last']
        task_coord = scoop_info['task_coord']
        pot = [pot for pot in self.world_state['pot'] if pot.location == task_coord][0]

        # Empty the pot as well
        pot.ingredient_count = defaultdict(int)
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
        if is_last and task.head.next:
            print('base_agent@scoop - Update plate -> serve task')
            task.head = task.head.next
    
    def serve(self, task_id: int, serve_info):
        print('base_agent@serve')
        self.world_state['explicit_rewards']['serve'] += 1
        task = [task for task in self.world_state['goal_space'] if id(task) == task_id][0]
        is_last = serve_info['is_last']
        
        # plate returns to return point (in clean form for now)
        self.holding.dish = None
        self.holding.state = 'empty'
        self.holding.location = (5,0)
        self.world_state['plate'].append(self.holding)

        # remove dish from plate
        self.holding = None

        # remove order from TaskList
        if is_last:
            print('base_agent@serve - Remove serve task')
            task.head = task.head.next
            if task.head == None:
                for idx, task_no in enumerate(self.world_state['goal_space']):
                    if id(task_no) == task_id:
                        del self.world_state['goal_space'][idx]
                        break

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
        print(f'Found all valid surrounding cells')
        print(all_valid_surrounding_cells)

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
        print('base_agent@drop - Drop item in-hand')
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
