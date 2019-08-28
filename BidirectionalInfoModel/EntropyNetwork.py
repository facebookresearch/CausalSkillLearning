#!/usr/bin/env python
from headers import *

class PolicyNetwork(torch.nn.Module):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super().__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = 1
		
		# In the __init__ function, define the "Operations" that you're going to apply. 
		# In the forward function, use these ops to define the network. 
		# The point of this is so that you can "Instantiate" multiple hidden layers. 
		# Say choose number of hidden layers, then iterate and keep calling self.hidden_layer. 

		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers,bidirectional=True)
		# Define output layers for the LSTM, and activations for this output layer. 
		self.output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.probability_value = torch.nn.Sigmoid()
		self.logprobability_value = torch.nn.LogSigmoid()

	def forward(self, input, hidden=None, return_log_probabilities=False):		
		# The argument hidden_input here is the initial hidden state we want to feed to the LSTM. 				
		# Assume inputs is the trajectory sequence.

		# Input Format must be: Sequence_Length x 1 x Input_Size. 
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float()
		outputs, hidden = self.lstm(format_input)

		# Takes softmax of last output. 
		if return_log_probabilities:
			# Computes log probabilities, needed for loss function and log likelihood. 
			# log_probabilities = self.pertime_step_logsoftmax_layer(self.output_layer(outputs[-1]))
			log_probability = self.logprobability_value(self.output_layer(outputs[-1])).squeeze(1)
			return outputs, hidden, log_probability
		else:
			# Compute action probabilities for sampling. 
			probability = self.probability_value(self.output_layer(outputs[-1])).squeeze(1)
			return outputs, hidden, probability