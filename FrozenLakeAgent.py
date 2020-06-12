'''
Q Learning

A model free learning technique that can be used to find the optimal action-selection policy using the Q function.

Q [number of states, number of actions] (Matrix, called the Q Table data structure, telling the best possible action at each state)
 
We learn a function Q that basically tells us how to act in each state.

Process:

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

Steps:

Bellman Equation:
New_Q[s, a] = curr_Q[s, a] + alpha * [R[s, a] + (gamma * max(new_Q[s', a']) - curr_Q[s, a]]

R[s, a]: Reward for taking action a at state s
1. Make an environment
2. Initialize Q randomly
3. Choose hyperparameters 
	alpha: learning rate
	gamma: discount rate
4. Loop through episode


'''

# Q Learning agent for Hill Climbing

import gym
import numpy as np 

# make the env
env = gym.make('FrozenLake8x8-v0') 

# init Q
Q = np.random.rand(env.observation_space.n, env.action_space.n)

# hyperparam
alpha = 0.9 # learning rate
gamma = 0.90  # discount rate
epis = 1000
steps_per_ep = 500

total_r = [] # list of rewards over the episodes
# Iterate through all the episodes
for i in range(epis):
	print("Epsiode: ", i)

	d = False # boolean to check if the task is done
	s = env.reset() # start state
	r_epis = 0 # Reward for the episode
	for j in range(steps_per_ep):
		# action to take at that step (index of the max value in the row of s)
		a = np.argmax(Q[s,:]) 

		# perform the action
		new_s, reward, done, info = env.step(a)

		# update Q Table
		Q[s, a] = Q[s, a] + alpha * (reward + (gamma * np.max(Q[new_s,:])) - Q[s, a])
		r_epis += reward

		if done:
			break

	total_r.append(r_epis)
	env.render()

print("Total Average Reward: ", np.sum(total_r)/epis)
env.close()


















