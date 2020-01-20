#!/usr/bin/env python
from headers import *

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
