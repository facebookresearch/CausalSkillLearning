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
		self.cummulative_reward = 0.	
		self.terminal = terminal
		self.success = success

	def set_cumm_reward(self, cumm_reward):
		self.cummulative_reward = cumm_reward

class ReplayMemory():

	def __init__(self, memory_size=100000):
		
		# Implementing the memory as a list of transitions. 
		# This acts as a queue. 
		self.memory = []

		# Accessing the memory with indices should be constant time, so it's okay to use a list. 
		# Not using a priority either. 
		self.memory_len = 0
		self.memory_size = memory_size

		print("Setup Memory.")

	def append_to_memory(self, transition):

		if self.check_full():
			# Remove first transition in the memory (queue).
			self.memory.pop(0)
			# Now push the transition to the end of hte queue. 
			self.memory.append(transition)
		else:
			self.memory.append(transition)

		self.memory_len+=1

	def append_list_to_memory(self, transition_list):
		for trans in transition_list:
			self.append_to_memory(trans)

	def sample_batch(self, batch_size=25):

		self.memory_len = len(self.memory)

		indices = npy.random.randint(0,high=self.memory_len,size=(batch_size))

		return indices

	def retrieve_batch(self, batch_size=25):
		# self.memory_len = len(self.memory)

		return npy.arange(0,batch_size)

	def check_full(self):

		self.memory_len = len(self.memory)

		if self.memory_len<self.memory_size:
			return 0 
		else:
			return 1 
