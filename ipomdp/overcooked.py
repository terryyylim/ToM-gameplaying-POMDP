from typing import Optional
from typing import Tuple

from configs import *

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
