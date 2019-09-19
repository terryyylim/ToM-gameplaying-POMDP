import click

from ipomdp.envs.map_env import MapEnv
from ipomdp.envs.overcooked_map_env import OvercookedEnv

class Controller(object):
    def __init__(
        self,
        env_name='cleanup'
    ) -> None:
        self.env_name = env_name
        if env_name == 'overcooked':
            print('Initializing Overcooked environment')
            self.env = OvercookedEnv()
            print(self.env)
        elif env_name == 'cleanup':
            print('Initializing Cleanup environment')
            # self.env = CleanupEnv(num_agents=5, render=True)
        else:
            print('Error! Not a valid environment type')


@click.command()
@click.option("--env", default='overcooked', help="input the env you want to initialize here")
def main(env: str) -> None:
    c = Controller(env_name=env)

if __name__ == "__main__":
    main()
