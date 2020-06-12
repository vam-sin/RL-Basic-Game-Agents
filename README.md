# RL-Basic-Game-Agents
Basic Reinforcement Learning based Game Agents which make use of Q-Learning

The agents are built using the OpenAi Gym library. Making use of Q-Learning the agent learns to navigate through the environemnt.

# Q Learning

A model free learning technique that can be used to find the optimal action-selection policy using the Q function.

Q [number of states, number of actions] (Matrix, called the Q Table data structure, telling the best possible action at each state)
 
We learn a function Q that basically tells us how to act in each state.

## Process:

1. Initialize Q randomly
2. Choose option from Q
3. Perform that action
4. Measure the reward of that action
5. Update the estimate of Q it originally had in the Q Table 
6. Go to step 2

But this makes the algorithm greedy, the agent once it receives a lesser reward, wont ever choose that
choice again. Makes it less exploratory, it will always choose according the optimal policy that it currenty has.

To make it exploratory, we make sure that at a certain probability, it chooses a random action, so that the agent
can explore all the possible rewards in the space.

## Code Steps:

Bellman Equation:
New_Q[s, a] = curr_Q[s, a] + alpha * [R[s, a] + (gamma * max(new_Q[s', a']) - curr_Q[s, a]]

R[s, a]: Reward for taking action a at state s
1. Make an environment
2. Initialize Q randomly
3. Choose hyperparameters 
	alpha: learning rate
	gamma: discount rate
4. Loop through episode
