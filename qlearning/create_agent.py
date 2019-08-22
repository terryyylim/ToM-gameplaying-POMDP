import click
import logging

import helpers
from config import AgentConfig
from config import agent_configurations

class Agent:
    """
    The Agent to navigate the Map Environment created using GUI interface tkinter.
    """

    def __init__(
        self,
        config: AgentConfig
    ) -> None:
        self.config = config.AGENT_CONFIG


@click.command()
@click.option('--config', default='production', help='the deployment target')
def main(config: str) -> None:
    logging.info('Creating agent...')

    configuration = helpers.get_configuration(config, agent_configurations)

    agent = Agent(config=configuration)
    helpers.save(agent)


if __name__ == "__main__":
    logger = helpers.get_logger()

    main()
