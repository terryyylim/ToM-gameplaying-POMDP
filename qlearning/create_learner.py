from typing import Any
from typing import Tuple
from typing import Optional

import click
import logging
import time

import helpers
from create_env import EnvMap
from config import LearnerConfig
from config import learner_configurations


class QLearner:
    """
    The Learner to learn the Map Environment created using GUI interface tkinter
    and optimize the rewards attained by the Agent initialized in the map.
    """
    def __init__(
        self,
        config: LearnerConfig,
        input_file: Optional[str] = None,
        input_env: Optional[EnvMap] = None
    ) -> None:
        self.config = config.LEARNER_CONFIG
        self.states = []
        self.q_table = {}

        # Load EnvMap object if present else create EnvMap
        self.load_env(input_file, input_env)

        self.initialize_q_table()

    def load_env(self, input_file: Optional[str], input_env: Optional[EnvMap]) -> None:
        logging.info(f'Loading Map...')
        if input_file is None and input_env is None:
            self.input_file = helpers.find_latest_file(self.config.PATHS.env)
            self.env = helpers.load_input_file(self.input_file)  # type: ignore
        elif input_file is None and input_env is not None:
            self.env = input_env
        elif input_file is not None and input_env is not None:
            logging.warning('Both an input file and an input Env object were provided. Using the object.')
            self.env = input_env
        elif input_file is not None and input_env is None:
            self.input_file = click.format_filename(input_file)
            self.env = helpers.load_input_file(self.input_file)

    def initialize_q_table(self) -> None:
        for x_coord in range(self.env.config['x_axis']):
            for y_coord in range(self.env.config['y_axis']):
                self.states.append((x_coord, y_coord))

        for state in self.states:
            temp = {}
            for action in self.env.agent.config['actions']:
                temp[action] = 0.1
                self.env.set_cell_score(state, action, temp[action])
            self.q_table[state] = temp

        for (x_coord, y_coord, _, reward) in self.env.config['rewards']:
            for action in self.env.agent.config['actions']:
                self.q_table[(x_coord, y_coord)][action] = reward
                self.env.set_cell_score((x_coord, y_coord), action, reward)

    def take_action(self, action: str) -> Any:
        """
        Perform the maximum reward action.

        Parameters
        ----------
        action: str
            The action to take.
        
        Returns
        -------
        Tuple[int, int]
            The x-coord and y-coord of the agent's previous position.
        str
            The action taken.
        int
            The total rewards attained thus far.
        Tuple[int, int]
            The x-coord and y-coord of the agent's current position.
        """
        cur_pos = self.env.agent.config['pos']
        exp_future_reward = -self.env.total_rewards # after implementation, try to make it positive and set config to neg
        if action == self.env.agent.config['actions'][0]:
            self.env.move_up()
        elif action == self.env.agent.config['actions'][1]:
            self.env.move_down()
        elif action == self.env.agent.config['actions'][2]:
            self.env.move_left()
        elif action == self.env.agent.config['actions'][3]:
            self.env.move_right()
        else:
            logging.info('Invalid action was specified!')
            return
        new_pos = self.env.agent.config['pos']
        exp_future_reward += self.env.total_rewards
        return cur_pos, action, exp_future_reward, new_pos

    def max_Q(self, cur_state: Tuple[int, int]) -> Any:
        """
        Exploitation step: Choose the action that returns the maximum reward.

        Parameters
        ----------
        cur_state: Tuple[int, int]
            The x-coord and y-coord of the agent's current position.
        
        Returns
        -------
        str, int
            The maximum reward - action, reward to take and receive respectively.
        """
        val = None
        action = None

        # At every grid, there are up to 4 actions available for consideration
        # First action of every episode is always 'up'
        for a, q_value in self.q_table[cur_state].items():
            if val is None or (q_value > val):
                val = q_value
                action = a
        return action, val
    
    def increment_Q(self, cur_state: Tuple[int, int], action: str, alpha: float, inc: float) -> None:
        """
        Exploitation step: Choose the action that returns the maximum reward.

        Parameters
        ----------
        cur_state: Tuple[int, int]
            The x-coord and y-coord of the agent's current position.
        action:
            The maximum reward - action to take.
        alpha:
            The learning rate - how much we accept the new value vs the old value.
            A factor of 0 makes the agent learn nothing (exclusively exploiting prior knowledge).
            A factor of 1 makes the agent consider only the most recent information (ignoring
            prior knowledge to explore possibilities).
        inc:

        """
        self.q_table[cur_state][action] *= 1 - alpha
        self.q_table[cur_state][action] += alpha * inc
        self.env.set_cell_score(cur_state, action, self.q_table[cur_state][action])

    def run(self) -> None:
        time.sleep(1)
        alpha = self.config['lr']
        discount = self.config['discount']
        time_step = 1
        all_rewards = []
        ep_count = 0
        convergence_state = False
        while not convergence_state:
            cur_pos = self.env.agent.config['pos']
            # Pick the action which gives the maximum reward (exploitation)
            max_action, max_val = self.max_Q(cur_pos)
            (cur_pos, a, exp_future_reward, new_pos) = self.take_action(max_action)

            # Update Q-table
            max_action, max_val = self.max_Q(new_pos)
            self.increment_Q(cur_pos, a, alpha, exp_future_reward+discount*max_val)

            # After 1 timestep, check if terminal state has been reached
            time_step += 1
            if self.env.reset:
                # Keep track of converging rewards
                ep_count += 1
                all_rewards.append(self.env.total_rewards)
                if len(all_rewards) > 10 and all(reward == all_rewards[-1] for reward in all_rewards[:-11:-1]):
                    convergence_state = True
                self.env.reset_map()
                time.sleep(0.01)
                time_step = 1

            # Update learning rate (Take smaller steps as we progress)
            alpha = pow(time_step, -0.1)
            
            # Convergence has been reached
            if convergence_state:
                logging.info(f'After {ep_count} episodes, maximum reward of {all_rewards[-1]} has been attained!')
                logging.info(f'Final Q-table: \n{self.q_table}')
                return

            time.sleep(0.1)

@click.command()
@click.option('--config', default='production', help='the deployment target')
@click.option('--input_file', default=None, type=click.Path(exists=True, dir_okay=False))
def main(config: str, input_file: str) -> None:
    logging.info('Creating learner...')

    configuration = helpers.get_configuration(config, learner_configurations)

    learner = QLearner(config=configuration, input_file=input_file)
    helpers.save(learner)



if __name__ == "__main__":
    logger = helpers.get_logger()

    main()

