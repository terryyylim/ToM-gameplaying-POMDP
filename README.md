# Modelling Theory of Mind in MultiAgent Cooperative Game Playing using Inverse-planning nested POMDPs

## Content
* [Summary](#Summary)
* [Installation](#Installation)

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
