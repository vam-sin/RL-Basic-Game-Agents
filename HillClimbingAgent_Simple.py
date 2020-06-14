# Libraries
import numpy as np 
import gym
from gym import wrappers
import random
import time

# params
n_states = 40
num_episodes = 10000
steps_per_ep = 50000
alpha = 1.0 # LR
gamma = 1.0 # discount factor
eps = 0.02 # prob for random event

def bucket_state(env, s1, s2):
	# places states into the number of buckets we have
	low = list(env.observation_space.low)
	high = list(env.observation_space.high) 
	bucket_size1 = (high[0]-low[0])/n_states
	bucket_size2 = (high[1]-low[1])/n_states
	c1 = int((s1-low[0])/bucket_size1)
	c2 = int((s2-low[1])/bucket_size2)

	return c1, c2

# Final policy
def run_solution_policy(env, Q):
	# Q is the solution policy
	steps_per_ep = 5000
	render_time = 10
	d = False
	s1, s2 = env.reset()
	s1, s2 = bucket_state(env, s1, s2)
	r_epis = 0.0

	for j in range(steps_per_ep):
		env.render()
		time.sleep(0.01)
		a = np.argmax(Q[s1,s2,:])
		new_s, reward, done, info = env.step(a)
		new_s1, new_s2 = bucket_state(env, new_s[0], new_s[1])

		r_epis += reward 

		s1, s2 = new_s1, new_s2

		if done and reward == -1:
			print("Nope: ", reward)
			return
		elif done and reward != -1:
			print("New: ", reward)
			return

	print("Not Done")
	env.render()

# Environment
env = gym.make('MountainCar-v0')

# Reproducible results 
env.seed(0)
np.random.seed(0)

# states are defined by tuple of two float values
Q = np.zeros([n_states, n_states, env.action_space.n])

total_reward = []
for i in range(num_episodes):
	print("Episode: ", i)

	s1, s2 = env.reset() # Current state
	s1, s2 = bucket_state(env, s1, s2)

	ep_reward = 0

	for j in range(steps_per_ep):

		# Exploration vs. Exploitation
		if np.random.uniform(0, 1) < eps:
			a = random.randint(0, env.action_space.n-1)
		else:
			a = np.argmax(Q[s1,s2,:])

		new_s, reward, done, info = env.step(a)
		new_s1, new_s2 = bucket_state(env, new_s[0], new_s[1])

		ep_reward += reward

		# Q update
		Q[s1, s2, a] = Q[s1, s2, a] + alpha * (reward + (gamma * np.max(Q[new_s1, new_s2, :])) - Q[s1, s2, a])

		s1, s2 = new_s1, new_s2

		if done:
			print(reward)
			break

	total_reward.append(ep_reward)

run_solution_policy(env, Q)
print("Total Average Reward: ", np.sum(total_reward)/num_episodes)
env.close()
