"""
Overcooked (Stage: 1)
----------
Every stage of Overcooked will have different declarations of initial barriers and world state.
To get to respective subgoals, agent/observer has to move to grid before the subgoal.

Ingredients should be pre-determined before start of the simulation (we need to know if ingredient needs to be cooked).
"""
BARRIERS = [
    (0,8), (1,8), (2,8), (3,8), (4,8), (5,8), (6,8), (7,8), (8,8), (9,8), (10,8), (11,8), (12,8),
    (0,8), (0,7), (0,6), (0,5),
    (0,4), (1,4), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4),
    (0,4), (0,3), (0,2), (0,1),
    (0,0), (1, 0), (2, 0), (3,0), (4, 0), (5, 0), (6,0), (7, 0), (8, 0), (9,0), (10, 0), (11, 0),
    (12,0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8)
]

"""
WORLD_STATE
-----------
World state models where key stations are located. (Eg. chopping board - K1(3,1), K2(5,1) etc.)
World state models where agents/observers are located. (Eg. agent - A(6,8), O(6,4) etc.)
World state models where raw <ingredient>, fresh <ingredient> etc. are located.
World state models where all combinations of partially-cooked/fully-cooked dishes are located. (Eg. onion, onion-onion, onion-onion-onion)
At any one point, world state for pots are being tracked for location and burnt chance.
Dict[str: List[Optional[int, int]]]

Definitions
-----------
Following notation is in the form of dictionary key -> explanation

<To run loop to append ingredient(s) information to World state dictionary; because possibility of multiple ingredients>
collection_<ingredient> : Collection point for raw/fresh ingredient.
r_<ingredient> -> raw <ingredient>: Un-chopped ingredient that needs to be cooked. 
rc_<ingredient> -> raw chopped <ingredient>: Chopped ingredient that needs to be cooked.
rco_<ingredient> -> raw cooked <ingredient>: Ingredient that is cooked.
f_<ingredient> -> fresh <ingredient>: Ingredient that does not need to be cooked.
fc_<ingredient> -> fresh chopped <ingredient>: Chopped ingredient that does not need to be cooked.

d_plates -> dirty plates: Plates that require washing before use.
c_plates -> clean plates: Plates that do not require washing before use.
pp_plates -> partially plated plates: Plates that hold ingredients (in-preparation). 
fp_plates -> fully plated plates: Plates that hold ingredients (ready to-be-served).
e_boards -> empty chopping board: Chopping boards that are empty.
f_boards -> filled chopping board: Chopping boards that are filled.
e_stoves -> empty stoves: Stoves that are empty.
f_stoves -> filled stoves: Stoves that are filled.
e_pots -> empty pots: Pots that are empty.
pc1_pots -> partially cooked 1 <ingredient> pots: Pots with 1 cooking onion. (with timer countdown)
pc2_pots -> partially cooked 2 <ingredient> pots: Pots with 2 cooking onions. (with timer countdown)
pc3_pots -> partially cooked 3 <ingredient> pots: Pots with 3 cooking onions. (with timer countdown)
pco1_pots -> cooked 1 <ingredient> pot: Pot with 1 cooked onion.
pco2_pots -> cooked 2 <ingredient> pot: Pot with 2 cooked onions.
pco3_pots -> cooked 3 <ingredient> pot: Pot with 3 cooked onions.
on_pots -> pots on stove: Pots that are on the stove.
off_pots -> pots off stove: Pots that are off the stove.
burning_pots -> burning pots: Pots that are on the verge of burning.
burnt_pots -> burnt pots: Pots that are burnt.
sc_<X> -> Service counters to serve food.
r_plates -> Plates retuning venue after <X> seconds of food consumption.
basin: Basin to wash dirty plates.

agent: Location of agent
observer: Location of observer
(is it done?)

To clear chopping board space, find nearest empty tabletop grid and shift it there
"""
WORLD_STATE_1 = {
    'agent': (6,8),
    'observer': (6,4),
    'board_1': (3,1),
    'board_2': (5,1),
    'r_onion': (3,7),
    'stove_1': (8,7),
    'plate_1': (9,1),
    'plate_2': (10,1),
    'basin': (11,5),
    'bin': (12,1),
    'sc_1': (1,1),
    'sc_2': (1,2),
    'pot_1': (8,7),
    'r_plates': (1,3),
    'd_plates': [],
    'c_plates': [(9,1), (10,1)],
    'pp_plates': [],
    'fp_plates': [],
    'e_boards': [(3,1), (5,1)],
    'f_boards': [],
    'e_stoves': [],
    'f_stoves': [(8,7)],
    'e_pots': [(8,7)],
    'pc1_pots': [],
    'pc2_pots': [],
    'pc3_pots': [],
    'pco1_pots': [],
    'pco2_pots': [],
    'pco3_pots': [],
    'on_pots': [(8,7)],
    'off_pots': [],
    'burning_pots': [],
    'burnt_pots': []
}

"""
Actions
-------
TO-DO: Actions that can be performed.
"""

"""
Rewards
-------
TO-DO: Rewards that have floating values.
"""