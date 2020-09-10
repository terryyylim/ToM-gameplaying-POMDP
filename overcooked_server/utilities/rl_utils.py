import logging
import os
import numpy as np

from settings import GRIDWIDTH, GRIDHEIGHT


def init_layers(num_agents):
    base_map_features = ['table_tops', 'pot_loc', 'chop_loc', 'dish_disp_loc',
                         'onion_disp_loc', 'serve_loc']
    variable_features = ['onions_fresh', 'onions_chopped', 'plates', 'soup']
    agent_features = ["player_{}".format(idx+1) for idx in range(num_agents)]
    LAYERS = agent_features + base_map_features + variable_features
    return LAYERS


def get_state_shape(world_state):
    x = int(GRIDWIDTH)
    y = int(GRIDHEIGHT - 1)
    return (y, x)  # pygame uses (y,x)


def get_loc(world_state, object_name):
    """ Obtains (x,y) coordinates of objects in Overcooked world_state"""
    coords = [game_object.location for game_object in world_state[object_name]]
    return coords


def vectorize_world_state(world_state, layers):
    """ Transforms overcooked_ai world_state into numpy array for CNN"""
    def make_layer(position, value):
        layer = np.zeros(shape)
        layer[position] = value
        return layer

    shape = get_state_shape(world_state)

    state_mask_dict = {layer: np.zeros(shape) for layer in layers}

    # MAP LAYERS
    for loc in world_state['table_tops']:
        state_mask_dict['table_tops'][loc] = 1

    for pot in world_state['pot']:
        state_mask_dict['pot_loc'][pot.location] = 1
        if 'onion' in pot.ingredient_count.keys():
            state_mask_dict['onions_chopped'] += make_layer(pot.location,
                                                            pot.ingredient_count['onion'])
    for loc in get_loc(world_state, 'chopping_board'):
        state_mask_dict['chop_loc'][loc] = 1

    if isinstance(world_state['return_counter'], list):
        for loc in world_state['return_counter']:
            state_mask_dict['dish_disp_loc'][loc] = 1
    else:
        state_mask_dict['dish_disp_loc'][world_state['return_counter']] = 1

    for loc in world_state['ingredient_onion']:
        state_mask_dict['onion_disp_loc'][loc] = 1

    for loc in world_state['service_counter']:
        state_mask_dict['serve_loc'][loc] = 1

    # VARIABLE LAYERS
    if 'ingredients' in world_state.keys():
        for ingredient in world_state['ingredients']:
            if ingredient.state == 'uncooked':
                state_mask_dict['onions_fresh'] += make_layer(ingredient.location, 1)
            elif ingredient.state == 'chopped':
                state_mask_dict['onions_chopped'] += make_layer(ingredient.location, 1)

    for plate in world_state['plate']:
        state_mask_dict['plates'][plate.location] = 1
        if plate.state == 'plated':
            state_mask_dict['soup'][plate.location] = 1

    if 'cooked_dish' in world_state.keys():
        for loc in get_loc(world_state, 'cooked_dish'):
            state_mask_dict['soup'][loc] = 1

    # AGENT LAYER
    for i, loc in enumerate(get_loc(world_state, 'agents')):
        state_mask_dict["player_{}".format(i+1)][loc] = 1

    state_mask_stack = np.array([[state_mask_dict[layer] for layer in layers]])
    return np.array(state_mask_stack).astype(float)


def flip_array(agent_id, vec_world_state, layers):
    MAIN_AGENT_IDX = 0
    agent_idx = layers.index('player_{}'.format(agent_id))
    vec_world_state[0][MAIN_AGENT_IDX], vec_world_state[0][agent_idx] = vec_world_state[0][agent_idx], vec_world_state[0][MAIN_AGENT_IDX]
    return vec_world_state


def setup_logger(filepath):
    """Sets up the logger"""
    filename = os.path.join(filepath, "Training.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # create a file handler
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(handler)
    return logger
