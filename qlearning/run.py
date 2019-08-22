import click
import logging
import threading
import time

import helpers
from config import agent_configurations
from config import map_configurations
from config import learner_configurations
from create_agent import Agent
from create_env import EnvMap
from create_learner import QLearner


@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:
    if config not in ('production'):
        raise ValueError(f'Unknown deployment environment "{config}"')

    try:
        # Create Agent
        logging.info('Creating Agent')
        agent_configuration = helpers.get_configuration(config, agent_configurations)
        agent = Agent(agent_configuration)

        # Create Environment Map
        logging.info('Creating Environment Map')
        map_configuration = helpers.get_configuration(config, map_configurations)
        env_map = EnvMap(map_configuration, input_agent=agent)

        # Create QLearner
        logging.info('Creating Learner')
        learner_configuration = helpers.get_configuration(config, learner_configurations)
        qlearner = QLearner(learner_configuration, input_env=env_map)

        logging.info('Starting EnvMap!')
        t = threading.Thread(target=qlearner.run)
        t.daemon = True
        t.start()
        qlearner.env.start_grid()
    except Exception as e:
        logging.exception(e)
    else:
        logging.info('Success @run.py')


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
