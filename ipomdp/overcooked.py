from typing import Optional
from typing import Tuple

class Pot:
    def __init__(
        self,
        location: Tuple[int,int],
        intermediate_state: Optional[str] = None,
        cooking_state: Optional[str] = None,
        on_stove: Optional[bool] = True,
    ) -> None:
        """
        Paramters
        ---------
        intermediate_state: str
            keeps track of state based on RECIPES_COOKING_INTERMEDIATE_STATES_1 in `configs.py`

        TO-DO: Think of way to time the cooking process
        """
        self.location = location
        self.intermediate_state = intermediate_state
        self.on_stove = on_stove
        self.cooking_state = cooking_state

    def is_empty(self) -> bool:
        return self.intermediate_state == None

    def is_cooking(self) -> bool:
        if self.on_stove and self.intermediate_state:
            return True
        return False

    def get_cooking_state(self) -> str:
        return self.cooking_state
    
    def get_location(self) -> Tuple[str,str]:
        return self.locationclass Ingredient(Item):
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
