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
