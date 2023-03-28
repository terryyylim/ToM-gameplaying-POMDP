# ======================= World State =======================
# As of now: DO NOT ALLOW DROPPING ITEMS INFRONT OF TASK_PERFORMING CELL
MAP = 'modeling_no_line_1'

CACTUS = [(0,0), (0,11)]
PALM_TREES = [(1,1), (13,13)]
ROCKS = [(0,13)]
RIVER_WATER = [(2,13), (3,13), (4,13), (5,13), (6,13), (7,13), (8,13), (9,13), (10,13), (11,13)]
STILL_WATER = []
WELL = []
BARREL = [(4,0),(5,0),(6,0),(7,0),(8,0),(9,0)]

all_possible_cells = [
    (0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11), (0,12), (0,13),
    (1,0), (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11), (1,12), (1,13),
    (2,0), (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11), (2,12), (2,13),
    (3,0), (3,1), (3,2), (3,3), (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11), (3,12), (3,13),
    (4,0), (4,1), (4,2), (4,3), (4,4), (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (4,11), (4,12), (4,13),
    (5,0), (5,1), (5,2), (5,3), (5,4), (5,5), (5,6), (5,7), (5,8), (5,9), (5,10), (5,11), (5,12), (5,13),
    (6,0), (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), (6,7), (6,8), (6,9), (6,10), (6,11), (6,12), (6,13),
    (7,0), (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), (7,7), (7,8), (7,9), (7,10), (7,11), (7,12), (7,13),
    (8,0), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (8,8), (8,9), (8,10), (8,11), (8,12), (8,13),
    (9,0), (9,1), (9,2), (9,3), (9,4), (9,5), (9,6), (9,7), (9,8), (9,9), (9,10), (9,11), (9,12), (9,13),
    (10,0), (10,1), (10,2), (10,3), (10,4), (10,5), (10,6), (10,7), (10,8), (10,9), (10,10), (10,11), (10,12), (10,13),
    (11,0), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6), (11,7), (11,8), (11,9), (11,10), (11,11), (11,12), (11,13),
    (12,0), (12,1), (12,2), (12,3), (12,4), (12,5), (12,6), (12,7), (12,8), (12,9), (12,10), (12,11), (12,12), (12,13),
    (13,0), (13,1), (13,2), (13,3), (13,4), (13,5), (13,6), (13,7), (13,8), (13,9), (13,10), (13,11), (13,12), (13,13)
]


invalid_movement_cells = CACTUS + PALM_TREES + ROCKS + RIVER_WATER + STILL_WATER + WELL + BARREL
    #these are all possible cells minus all the objects listed above
valid_movement_cells = [x for x in all_possible_cells if x not in invalid_movement_cells]

#customize, or randomize spawn between all the valid movement cells, checking to make sure it's 
#not a duplicated location

HOLDING_TIMES = [
    [],
    [],
    [],
    [],
    [],
    [],
    [], 
    []
]

EP_DATA = {1:{1:{'coords': (4,11),'holding': False,'orientation':'D','completed': False},2:{'coords': (4,10),'holding': False,'orientation':'D','completed': False},3:{'coords': (4,9),'holding': False,'orientation':'D','completed': False},4:{'coords': (4,8),'holding': False,'orientation':'D','completed': False},5:{'coords': (4,7),'holding': False,'orientation':'D','completed': False},6:{'coords': (4,6),'holding': False,'orientation':'D','completed': False},7:{'coords': (4,5),'holding': False,'orientation':'D','completed': False},8:{'coords': (4,4),'holding': False,'orientation':'D','completed': False},},2:{1:{'coords': (3,11),'holding': False,'orientation':'D','completed': False},2:{'coords': (3,10),'holding': False,'orientation':'D','completed': False},3:{'coords': (3,9),'holding': False,'orientation':'D','completed': False},4:{'coords': (3,8),'holding': False,'orientation':'D','completed': False},5:{'coords': (3,7),'holding': False,'orientation':'D','completed': False},6:{'coords': (3,6),'holding': False,'orientation':'D','completed': False},7:{'coords': (3,5),'holding': False,'orientation':'D','completed': False},8:{'coords': (3,4),'holding': False,'orientation':'D','completed': False},},3:{1:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,5),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,4),'holding': False,'orientation':'D','completed': False},},4:{1:{'coords': (2,12),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,5),'holding': False,'orientation':'D','completed': False},},5:{1:{'coords': (1,12),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,12),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},},6:{1:{'coords': (1,12),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,12),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},},7:{1:{'coords': (1,12),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,12),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},},8:{1:{'coords': (1,12),'holding': False,'orientation':'D','completed': False},2:{'coords': (2,12),'holding': False,'orientation':'D','completed': False},3:{'coords': (2,11),'holding': False,'orientation':'D','completed': False},4:{'coords': (2,10),'holding': False,'orientation':'D','completed': False},5:{'coords': (2,9),'holding': False,'orientation':'D','completed': False},6:{'coords': (2,8),'holding': False,'orientation':'D','completed': False},7:{'coords': (2,7),'holding': False,'orientation':'D','completed': False},8:{'coords': (2,6),'holding': False,'orientation':'D','completed': False},},}

import random
#shuffle players
mapping = random.sample(range(1, 9), 8)
for episode in EP_DATA:
    newDict = {}
    for i in range(1,9):
        newKey = mapping[i-1]
        newDict[newKey] = EP_DATA[episode][i]
    EP_DATA[episode] = newDict
# #shuffle holding times
NEW_HOLDING = []
for i in range(8):
    NEW_HOLDING.append([])
for idx, num in enumerate(mapping):
    NEW_HOLDING[num-1] = HOLDING_TIMES[idx]
HOLDING_TIMES = NEW_HOLDING
