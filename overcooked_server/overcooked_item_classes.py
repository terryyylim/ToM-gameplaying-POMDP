from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from collections import defaultdict

from settings import WORLD_STATE

class Item:
    def __init__(
        self,
        category: str,
        location: Tuple[int,int]
    ) -> None:
        self.id = id
        self.category = category
        self.location = location

    def get_category(self) -> str:
        return self.category

class ChoppingBoard(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int],
        state: str,
    ) -> None:
        super().__init__(category, location)
        self.state = state

class Extinguisher(Item):
    def __init__(
        self,
        category: str,
        location: Tuple[int,int]
    ) -> None:
        super().__init__(category, location)

class Plate(Item):
    def __init__(
        self,
        plate_id: int,
        category: str,
        location: Tuple[int,int],
        state: str,
        ready_to_serve: bool=False
    ) -> None:
        """
        Only start plating dish when ingredient has been prepared (Chopped/Cooked etc.)
        """
        super().__init__(category, location)
        self.plate_id = plate_id
        self.ready_to_serve = ready_to_serve
        self.state = state

class Pot(Item):
    def __init__(
        self,
        pot_id: int,
        category: str,
        location: Tuple[int,int],
        ingredient_count: Dict[str,int],
        ingredient: str=None,
        is_empty: bool=True,
    ) -> None:
        """
        Paramters
        ---------
        state: str
            Whether pot is cooking or not

        TO-DO: Think of way to time the cooking process
        """
        super().__init__(category, location)
        self.pot_id = pot_id
        self.ingredient = ingredient
        self.ingredient_count = ingredient_count
        self.is_empty = is_empty
        self.dish = None
    
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
        super().__init__(id, category)
        self.name = name
        self.is_raw = is_raw
        self.is_new = is_new
        self.state = state
        if is_new:
            self.initialize_pos()

    def initialize_pos(self):
        if self.is_raw:
            self.location = WORLD_STATE['ingredient_'+self.name][0]
        else:
            # for fresh ingredient (eg. lettuce); currently not in use
            self.location = WORLD_STATE['f_'+self.name][0]

class Dish(Item):
    def __init__(
        self,
        name,
        location
    ) -> None:
        self.name = name
        self.location = location
