# Q-learning

## Content
* [Summary](#Summary)
* [Exploitation vs Exploration](#Exploitation-vs-Exploration)

### Summary
Q-learning is an off-policy reinforcement learning algorithm that seeks to find the best action to take given the current state.

In Q-learning, a Q-table is initialized to store the q-values which will be updated after every episode, and it becomes a reference
table for the agent to select the best action based on the q-value. The updates will occur after each action and ends when a terminal
state is reached.

### Exploitation vs Exploration
When interacting with the environment, the agent can choose an action to optimize Exploitation or Exploration.

Exploitation first uses the Q-table as a reference and then selects the action based on the max value of those actions.
Essentially, it is making use of the information we have available to make a decision.

Exploration is taking the action randomly instead of making use of the information we have available. This is important
so that our agent can explore and discover new states that otherwise would not be selected during the exploitation process.

The balance between exploitation and exploration can be achieved by using epsilon(ε).
Eg. ε=0.2 represents 20% of exploration