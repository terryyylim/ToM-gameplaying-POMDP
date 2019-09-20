from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from ipomdp.agents.agent_configs import *

class TaskNode:
    def __init__(
        self,
        task: str,
        ingredient: str,
        state: str
    ) -> None:
        self.task = task
        self.ingredient = ingredient
        self.state = state
        self.next = None

class TaskList:
    def __init__(
        self,
        dish: str,
        task: List[str],
        ingredient: str
    ) -> None:
        self.task = task
        self.ingredient = ingredient
        self.initialize_tasks(task, ingredient)

    def initialize_tasks(self, task: List[str], ingredient: str):
        task_list = TaskNode(task[0][0], ingredient, task[0][1])
        head = task_list
        temp_pointer = head
        for sub_task in task[1:]:
            temp_pointer.next = TaskNode(sub_task[0], ingredient, sub_task[1])
            temp_pointer = temp_pointer.next
        self.head = head

class Item:
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        state: str,
    ) -> None:
        self.id = id
        self.category = category
        self.location = location
        self.state = state

    def get_category(self) -> str:
        return self.category

    def get_state(self) -> str:
        return self.state

class ChoppingBoard(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        placed_item: str=None,
    ) -> None:
        super().__init__(id, category, location)
        self.placed_item = placed_item

class Extinguisher(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int]
    ) -> None:
        super().__init__(id, category, location)

class Plate(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        ready_to_serve: bool=False
    ) -> None:
        """
        Only start plating dish when ingredient has been prepared (Chopped/Cooked etc.)
        """
        super().__init__(id, category, location)
        self.ready_to_serve = ready_to_serve

class Pot(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        intermediate_state: Optional[str] = None,
        on_stove: Optional[bool] = True,
    ) -> None:
        """
        Paramters
        ---------
        intermediate_state: str
            keeps track of state based on RECIPES_COOKING_INTERMEDIATE_STATES_1 in `configs.py`
        state: str
            Whether pot is cooking or not

        TO-DO: Think of way to time the cooking process
        """
        super().__init__(id, category, location)
        self.intermediate_state = intermediate_state
        self.on_stove = on_stove

    def is_empty(self) -> bool:
        return self.intermediate_state == None

    def is_cooking(self) -> bool:
        if self.on_stove and self.intermediate_state:
            return True
        return False
    
    def get_location(self) -> Tuple[str,str]:
        return self.location

class Stove(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        has_pot: bool=True
    ) -> None:
        super().__init__(id, category, location)
        self.has_pot = has_pot

class Ingredient(Item):
    def __init__(
        self,
        name: str,
        state: str,
        category: str,
        is_raw: bool,
        is_new: bool=True
    ) -> None:
        """
        Parameters
        ----------
        state:
            Whether the ingredient is unchopped/chopped/cooking/cooked
        is_raw:
            Whether the ingredient is raw/fresh
        is_new:
            Whether the ingredient is just taken from storage
        """
        super().__init__(id, category, state)
        self.name = name
        self.is_raw = is_raw
        self.is_new = is_new
        if is_new:
            self.initialize_pos()

    def initialize_pos(self):
        if self.is_raw:
            self.location = WORLD_STATE['r_'+self.name][0]
        else:
            self.location = WORLD_STATE['f_'+self.name][0]
