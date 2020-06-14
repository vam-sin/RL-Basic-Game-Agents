# Q Learning agent for Hill Climbing
# Iteration 1: Only Exploitation, No Exploration.

import gym
import numpy as np 
import time
import random 

def run_solution_policy(env, Q):
	# Q is the solution policy
	steps_per_ep = 500
	render_time = 10
	d = False
	s = env.reset()
	r_epis = 0.0

	for j in range(steps_per_ep):
		env.render()
		time.sleep(0.01)
		a = np.argmax(Q[s,:])  
		new_s, reward, done, info = env.step(a)

		r_epis += reward 

		s = new_s

		if done and reward == 0:
			for j in range(render_time):
				env.render()
			print(done)
			print("HOLE/ICE")
			return 
		elif done and reward == 1:
			env.render()
			print("GOAL")
			return 

	print("Not Done")
	env.render()

# make the env
env = gym.make('FrozenLake8x8-v0') 
env.seed(42)
np.random.seed(42)

# init Q
Q = np.zeros([env.observation_space.n, env.action_space.n])

# hyperparam
alpha = 0.128 # learning rate
gamma = 1  # discount rate
epis = 1000
eps = 0.99
steps_per_ep = 500

total_r = [] # list of rewards over the episodes
# Iterate through all the episodes
for i in range(epis):
	print("Epsiode: ", i)

	d = False # boolean to check if the task is done
	s = env.reset() # start state
	# holes = 0
	r_epis = 0 # Reward for the episode
	for j in range(steps_per_ep):
		# action to take at that step (index of the max value in the row of s)
		
		val = abs(np.random.uniform(0, 1))

		# Only Exploitation
		# a = np.argmax(Q[s,:])

		# Exploration and Exploitation
		if abs(val) <= eps:
			# Exploit 
			a = np.argmax(Q[s,:])
		else:
			# Exploration
			a = random.randint(0, env.action_space.n-1)

		# perform the action
		new_s, reward, done, info = env.step(a)
		# print(j, reward)

		if reward == 0.0 and done == True: # F
			# print("HOLE")
			new_r = -100
		elif reward == 0.0 and done == False: # F
			# print("ICE")
			new_r = 0
		elif reward == 1.0:
			# print("GOAL")
			new_r = 100

		r_epis += new_r

		# update Q Table
		Q[s, a] = Q[s, a] + alpha * (new_r + (gamma * np.max(Q[new_s,:])) - Q[s, a])

		s = new_s

		if done == True and reward == 1: # Hole
			print("GOAL BREAK")
			break
		elif done and reward == 0: # Goal
			print("HOLE BREAK")
			break

	total_r.append(r_epis)
	# env.render()

run_solution_policy(env, Q)
print("Total Average Reward: ", np.sum(total_r)/epis)
env.close()


















