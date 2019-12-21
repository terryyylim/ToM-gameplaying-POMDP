# Modelling Theory of Mind in MultiAgent Cooperative Game Playing using Inverse-planning nested POMDPs

## Content
* [Summary](#Summary)
* [Installation](#Installation)
* [Gameplay](#Gameplay)
* [Simulation](#Simulation)

### Summary
This repository contains the code for modeling agents through inverse planning with POMDPs on the agent to infer goals and subsequently predict the actions of the other agents in a cooperative setting.

### Installation
Clone this repository:
```
git clone https://github.com/terryyylim/ToM-gameplaying-POMDP.git
```

To prevent complications to the global environment variables, I suggest creating a virtual environment for tracking and using the libraries required for the project.

1. Create and activate a virtual environment (run `pip3 install virtualenv` first if you don't have Python virtualenv installed):
```
virtualenv -p python3 <desired-path>
source <desired-path>/bin/activate
```

2. Install the requirements:
```
pip install -r requirements.txt
```

### Gameplay
To start the game, run the following.
```
python overcooked_server/game.py
```

#### Controls
```Human Agent 1```

| Actions | PICK | CHOP | COOK | SCOOP | SERVE |
| :-----: | :---: | :---: | :---: | :---: | :---: |
| Key-Map |  Z  |  X  |  C  |  V  |  B  |

| Movement | UP  | DOWN | LEFT | RIGHT | 
| :------: | :-: | :--: | :--: | :---: |
| Key-Map  | UP  | DOWN | LEFT | RIGHT |

| Movement | DIAG-UP-LEFT | DIAG-UP-RIGHT | DIAG-DOWN-LEFT | DIAG-DOWN-RIGHT |
| :------: | :----------: | :-----------: | :------------: | :-------------: |
| Key-Map  |       ,      |        .      |        /       |      RSHIFT     |

```Human Agent 2```

| Actions | PICK | CHOP | COOK | SCOOP | SERVE |
| :-----: | :---: | :---: | :---: | :---: | :---: |
| Key-Map |   1  |   2  |   3  |   4   |   5   |

| Movement | UP  | DOWN | LEFT | RIGHT | 
| :------: | :-: | :--: | :--: | :---: |
| Key-Map  |  Y  |  H   |  G   |   J   |

| Movement | DIAG-UP-LEFT | DIAG-UP-RIGHT | DIAG-DOWN-LEFT | DIAG-DOWN-RIGHT |
| :------: | :----------: | :-----------: | :------------: | :-------------: |
| Key-Map  |       Q      |        W      |        E       |         R       |

### Simulation
To run a simulation, run the following with desired parameters in CLI.
```
python overcooked_server/game.py --num_ai_agents=2 --is_simulation=True --simulation_episodes=500
```