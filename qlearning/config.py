import pprint


class Config:
    def __repr__(self) -> str:
        output = ''
        attributes = [a for a in dir(self) if not a.startswith('__')]
        for attribute in attributes:
            output += attributes + '\n'  # type: ignore
            output += pprint.pformat(getattr(self, attribute))
            output += '\n'

        return output


class Paths:
    agent = './agent'
    env = './env'
    learner = './learner'


MAP_CONFIG = {
    'x_axis': 5,
    'y_axis': 5,
    'total_rewards': 1,
    'map_width': 100,
    'grid_width': 1,
    'triangle_size': 0.1,
    'cell_score_min': -0.2,
    'cell_score_max': 0.2,
    'reset': False,
    'screen_name': 'EnvMap',
    'map_colour': 'white',
    'obstacle_colour': 'black',
    'obstacles': [(1,1), (2,3), (3,3)],
    'rewards': [(4, 0, 'green', 1), (4, 1, 'red', -1)]
}

AGENT_CONFIG = {
    'name': 'agent',
    'colour': 'orange',
    'size': 1,
    'actions': ['up', 'down', 'left', 'right'],
    'move_reward': -0.02,
    'pos': (0, MAP_CONFIG['y_axis']-1)
}

LEARNER_CONFIG = {
    'lr': 1,
    'discount': 0.3
}


class AgentConfig(Config):
    AGENT_CONFIG = AGENT_CONFIG
    PATHS = Paths


class MapConfig(Config):
    MAP_CONFIG = MAP_CONFIG
    PATHS = Paths


class LearnerConfig(Config):
    LEARNER_CONFIG = LEARNER_CONFIG
    PATHS = Paths


agent_configurations = {
    'production': AgentConfig
}

map_configurations = {
    'production': MapConfig
}

learner_configurations = {
    'production': LearnerConfig
}