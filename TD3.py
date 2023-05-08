# This file contain the actor and criric network structure and TD3 algorithms
# The TD3 algorihtms is reference from: 
# D. Byrne, “TD3: Learning to run with ai,” Medium, https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93 (accessed May 8, 2023). 


import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta,Normal
import math

# This is used to allocated and move data between CPU and GPU for memory allocation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor network
class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, net_width, maxaction):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, action_dim)

		self.maxaction = maxaction

	# Create a feed forward neural network
	def forward(self, state):
		a = torch.tanh(self.l1(state))
		a = torch.tanh(self.l2(a))
		a = torch.tanh(self.l3(a)) * self.maxaction
		return a

# Critic network
class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, net_width):
		super(Q_Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, net_width)
		self.l2 = nn.Linear(net_width, net_width)
		self.l3 = nn.Linear(net_width, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, net_width)
		self.l5 = nn.Linear(net_width, net_width)
		self.l6 = nn.Linear(net_width, 1)

	# Create a feed forward neural network
	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2

	# Q-value used to update the actor network
	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


# TD3 algorihtms function
class TD3(object):
	# Initialize the variable for the algorithms
	def __init__(
		self,
		env_with_Dead,
		state_dim,
		action_dim,
		max_action,
		gamma=0.99,
		net_width=128,
		a_lr=1e-4,
		c_lr=1e-4,
		Q_batchsize = 256
	):
		self.rmse_q_loss = []

		# Set up the actor network
		self.actor = Actor(state_dim, action_dim, net_width, max_action).to(device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		# Set up the crtitic network
		self.q_critic = Q_Critic(state_dim, action_dim, net_width).to(device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		# Set up the variable
		self.env_with_Dead = env_with_Dead
		self.action_dim = action_dim
		self.max_action = max_action
		self.gamma = gamma
		self.policy_noise = 0.2*max_action
		self.noise_clip = 0.5*max_action
		self.tau = 0.005
		self.Q_batchsize = Q_batchsize
		self.delay_counter = -1
		self.delay_freq = 1

	# This function will return the value about the prediction of the next action
	def select_action(self, state):#only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			a = self.actor(state)
		return a.cpu().numpy().flatten()

	# Return the calculated rmse critic loss
	def return_rmse_q_loss(self):
		return self.rmse_q_loss
	
	# Reset the rmse of critic loss after each episode
	def reset_rmse_q_loss(self):
		self.rmse_q_loss = []

	# Function to train the TD3 agent
	def train(self,replay_buffer):
		self.delay_counter += 1
		with torch.no_grad():
			s, a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
			noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
			smoothed_target_a = (
					self.actor_target(s_prime) + noise  # Noisy on target action
			).clamp(-self.max_action, self.max_action)

		# Compute the target Q value
		target_Q1, target_Q2 = self.q_critic_target(s_prime, smoothed_target_a)
		target_Q = torch.min(target_Q1, target_Q2)
		'''DEAD OR NOT'''
		if self.env_with_Dead:
			target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
		else:
			target_Q = r + self.gamma * target_Q  # env without dead


		# Get current Q estimates
		current_Q1, current_Q2 = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
		
		# Record average root mean square error of Q-value
		self.rmse_q_loss.append(math.sqrt(q_loss.detach().numpy()))
		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Section to determine when to udpate the actor network
		if self.delay_counter == self.delay_freq:
			# Update Actor
			a_loss = -self.q_critic.Q1(s,self.actor(s)).mean()
			self.actor_optimizer.zero_grad()
			a_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			self.delay_counter = -1

	# Function to save the actor and critic weight
	def save(self,episode):
		torch.save(self.actor.state_dict(), "models/ppo_actor{}.pth".format(episode))
		torch.save(self.q_critic.state_dict(), "models/ppo_q_critic{}.pth".format(episode))

	#  Load the model trained
	def load(self,episode):

		# This is for load the model that is new generated after training done
		self.actor.load_state_dict(torch.load("models/ppo_actor{}.pth".format(episode), map_location=torch.device('cpu')))
		self.q_critic.load_state_dict(torch.load("models/ppo_q_critic{}.pth".format(episode), map_location=torch.device('cpu')))

		# you can modify the path to load model from specific directory by adding the file path such as "file_path/models/ppo_actor{}.pth" and "file_path/models/ppo_q_critic{}.pth"
		# This will load the model that trained with large experience data
		# Uncomment this and comment the above part if run the models from other file
		# self.actor.load_state_dict(torch.load("exp/exp_1/models/ppo_actor{}.pth".format(episode), map_location=torch.device('cpu')))
		# self.q_critic.load_state_dict(torch.load("exp/exp_1/models/ppo_q_critic{}.pth".format(episode), map_location=torch.device('cpu')))
		print("****load checkpoint****")




