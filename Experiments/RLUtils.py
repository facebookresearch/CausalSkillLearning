#!/usr/bin/env python
from headers import *

def resample(original_trajectory, desired_number_timepoints):
	original_traj_len = len(original_trajectory)
	new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
	return original_trajectory[new_timepoints]

class Transition():

	def __init__(self, state, action, next_state, onestep_reward, terminal, success):
		# Now that we're doing 1step TD, and AC architectures rather than MC, 
		# Don't need an explicit value of return.
		self.state = state
		self.action = action
		self.next_state = next_state
		self.onestep_reward = onestep_reward
		self.terminal = terminal
		self.success = success

class Episode_TransitionList():

	def __init__(self, transition_list):
		self.episode = transition_list	

	def length(self):
		return len(self.episode)

# Alternate way of implementing an episode... 
# Make it a class that has state_list, action_list, etc. over the episode..
class Episode():

	def __init__(self, state_list=None, action_list=None, reward_list=None, terminal_list=None):
		self.state_list = state_list
		self.action_list = action_list
		self.reward_list = reward_list
		self.terminal_list = terminal_list
		self.episode_lenth = len(self.state_list)

	def length(self):
		return self.episode_lenth

class HierarchicalEpisode(Episode):

	def __init__(self, state_list=None, action_list=None, reward_list=None, terminal_list=None, latent_z_list=None, latent_b_list=None):

		super(HierarchicalEpisode, self).__init__(state_list, action_list, reward_list, terminal_list)

		self.latent_z_list = latent_z_list
		self.latent_b_list = latent_b_list		

class ReplayMemory():

	def __init__(self, memory_size=10000):
		
		# Implementing the memory as a list of EPISODES.
		# This acts as a queue. 
		self.memory = []

		# Accessing the memory with indices should be constant time, so it's okay to use a list. 
		# Not using a priority either. 
		self.memory_len = 0
		self.memory_size = memory_size

		print("Setup Memory.")

	def append_to_memory(self, episode):

		if self.check_full():
			# Remove first episode in the memory (queue).
			self.memory.pop(0)
			# Now push the episode to the end of hte queue. 
			self.memory.append(episode)
		else:
			self.memory.append(episode)

		self.memory_len+=1

	def sample_batch(self, batch_size=25):

		self.memory_len = len(self.memory)

		indices = np.random.randint(0,high=self.memory_len,size=(batch_size))

		return indices

	def retrieve_batch(self, batch_size=25):
		# self.memory_len = len(self.memory)

		return np.arange(0,batch_size)

	def check_full(self):

		self.memory_len = len(self.memory)

		if self.memory_len<self.memory_size:
			return 0 
		else:
			return 1 

# Refer: https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
"""
Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
"""
class OUNoise(object):
    def __init__(self, action_space_size, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.2, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space_size
        self.low          = -np.ones((self.action_dim))
        self.high         = np.ones((self.action_dim))
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0): 
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)