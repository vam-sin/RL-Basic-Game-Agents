# A neural network to learn the Q function
# instead of just updating on our own.

from torch import nn 
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np 
import pandas as pd 
import gym
import time

# env
env = gym.make('MountainCar-v0')
env.seed(0)
torch.manual_seed(0)
np.random.seed(1)

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.state_space = env.observation_space.shape[0]
		self.action_space = env.action_space.n 
		self.hidden1 = 200
		self.hidden2 = 200
		self.l1 = nn.Linear(self.state_space, self.hidden1, bias = False)
		self.l2 = nn.Linear(self.hidden1, self.hidden2, bias = False)
		self.l3 = nn.Linear(self.hidden2, self.action_space, bias = False)
		# state space (coordinates) -> hidden -> action space (prob for each of the actions)

	def forward(self, x):
		x = self.l1(x)
		x = F.relu(x)
		x = self.l2(x)
		x = F.relu(x)
		x = self.l3(x)
		x = F.softmax(x)

		return x

def run_solution_policy(env, Q):
		steps_per_ep = 2000
		d = False
		s = env.reset()
		r_epis = 0.0

		for j in range(steps_per_ep):
			env.render()
			time.sleep(0.01)
			Q = policy(Variable(torch.from_numpy(s).type(torch.FloatTensor)))
			_, a = torch.max(Q, -1)
			a = a.item()
			new_s, reward, done, info = env.step(a)

			r_epis += reward 

			s = new_s

			if done:
				if s[0] >= 0.5:
					print("Completion")
				else:
					print("Need more work")
				env.render()
				return

		print("Not Done")
		env.render()

# hyperparam
num_epis = 3000
steps_per_ep = 2000
eps = 0.01
total_reward = []
total_loss = []
gamma = 0.99
alpha = 0.001
max_pos = -0.4

# Initialize model
policy = Policy()
loss_func = nn.MSELoss()
opt = torch.optim.Adam(policy.parameters())
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size = 1, gamma = 0.9)

# train the policy
for i in range(num_epis):
	print("Episode: ", i)

	reward_ep = 0
	loss_ep = 0
	# print(loss_ep)
	s = env.reset()

	for j in range(steps_per_ep):

		# prob for actions in that state s
		Q = policy(Variable(torch.from_numpy(s).type(torch.FloatTensor)))

		# eps greedy
		# if np.random.rand(1) < eps:
		# 	a = np.random.randint(0, 3) # There are only 3 actions to choose from
		# else:
		_, a = torch.max(Q, -1)
		a = a.item()

		new_s, reward, done, info = env.step(a)


		if new_s[0] >= 0.5:
			print("Task Completed YAY")
			reward = 10
		elif new_s[0] > -0.4:
			reward = (1+new_s[0]) ** 2

		# action space probs from new state new_s
		Q1 = policy(Variable(torch.from_numpy(new_s).type(torch.FloatTensor)))
		maxQ1, _ = torch.max(Q1, -1)

		Q_tar = Q.clone()
		Q_tar = Variable(Q_tar.data)
		Q_tar[a] = reward + torch.mul(maxQ1.detach(), gamma)

		loss = loss_func(Q, Q_tar)
		# print(loss.item())

		policy.zero_grad()
		loss.backward()
		opt.step()

		loss_ep += loss.item()
		reward_ep += reward

		if done:
			if new_s[0] >= 0.5:
				print("Completion")
				eps *= 0.99
			else:
				print("Not Completed")
			total_reward.append(reward_ep)
			total_loss.append(loss_ep)

			break

		else:
			s = new_s

	# print(loss_ep)
	if (i+1) % 100 == 0:
		print("Intermediate Average Reward: ", np.sum(total_reward)/(i+1))
		print("Intermediate Average Loss: ", np.sum(total_loss)/(i+1))
		run_solution_policy(env, policy)

print("Total Average Reward: ", np.sum(total_reward)/num_epis)
print("Total Average Loss: ", np.sum(total_loss)/num_epis)
env.close()
