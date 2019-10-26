# Modelling Theory of Mind in MultiAgent Cooperative Game Playing using Inverse-planning nested POMDPs

## Content
* [Summary](#Summary)
* [Installation](#Installation)
* [Gameplay](#Gameplay)

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

3. Run the file
```
python rollout.py
```

### Gameplay
To start the game, run the following.
```
python overcooked_server/overcooked_game.py
```

#### Controls
| Actions | PICK | CHOP | COOK | SCOOP | SERVE |
| :-----: | :--: | :--: | :--: | :---: | :---: |
| Key-Map |   Z  |   X  |   C  |   V   |   B   |

| Movement | UP  | DOWN | LEFT | RIGHT | 
| :------: | :-: | :--: | :--: | :---: |
| Key-Map  | UP  | DOWN | LEFT | RIGHT |

| Movement | DIAG-UP-LEFT | DIAG-UP-RIGHT | DIAG-DOWN-LEFT | DIAG-DOWN-RIGHT |
| :------: | :----------: | :-----------: | :------------: | :-------------: |
| Key-Map  |       Q      |        W      |        E       |         R       |