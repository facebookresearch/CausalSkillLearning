#!/usr/bin/env python
from headers import *

class PolicyNetwork_BaseClass(torch.nn.Module):
	
	def __init__(self):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetwork_BaseClass, self).__init__()

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample().squeeze(0)
		return sample_action

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		return action_probabilities.argmax()

	def select_epsilon_greedy_action(self, action_probabilities):
		epsilon = 0.1
		if np.random.random()<epsilon:
			return self.sample_action(action_probabilities)
		else:
			return self.select_greedy_action(action_probabilities)

class PolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetwork, self).__init__()

		self.input_size = input_size+number_subpolicies+1
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = number_layers
		
		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.softmax_layer = torch.nn.Softmax(dim=1)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)		

	def forward(self, input, hidden=None, return_log_probabilities=False):		
		# The argument hidden_input here is the initial hidden state we want to feed to the LSTM. 				
		# Assume inputs is the trajectory sequence.

		# Input Format must be: Sequence_Length x 1 x Input_Size. 

		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float().cuda()
		outputs, hidden = self.lstm(format_input)

		# Takes softmax of last output. 
		if return_log_probabilities:
			# Computes log probabilities, needed for loss function and log likelihood. 

			preprobability_outputs = self.output_layer(outputs)
			log_probabilities = self.batch_logsoftmax_layer(preprobability_outputs).squeeze(1)
			probabilities = self.batch_softmax_layer(preprobability_outputs).squeeze(1)
			return outputs, hidden, log_probabilities, probabilities
		else:
			# Compute action probabilities for sampling. 
			softmax_output = self.softmax_layer(self.output_layer(outputs[-1]))
			return outputs, hidden, softmax_output

class ContinuousPolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousPolicyNetwork, self).__init__()

		self.input_size = input_size+number_subpolicies+1
		self.hidden_size = hidden_size
		# The output size here must be mean+variance for each dimension. 
		# This is output_size*2. 
		self.output_size = output_size
		self.num_layers = number_layers
		
		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)		

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()
		self.softplus_activation_layer = torch.nn.Softplus()

	def forward(self, input, action_sequence):
		# Input is the trajectory sequence of shape: Sequence_Length x 1 x Input_Size. 
		# Here, we also need the continuous actions as input to evaluate their logprobability / probability. 		
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float().cuda()

		hidden = None
		format_action_seq = torch.from_numpy(action_sequence).cuda().float().view(action_sequence.shape[0],1,self.output_size)
		lstm_outputs, hidden = self.lstm(format_input)


		# Predict Gaussian means and variances. 
		mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		variance_outputs = self.softplus_activation_layer(self.variances_output_layer(lstm_outputs))

		# Remember, because of Pytorch's dynamic construction, this distribution can have it's own batch size. 
		# It doesn't matter if batch sizes changes over different forward passes of the LSTM, because we're only going
		# to evaluate this distribution (instance)'s log probability with the same sequence length. 
		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))
		log_probabilities = dist.log_prob(format_action_seq)
		# log_probabilities = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs)).log_prob(format_action_seq)
		entropy = dist.entropy()
		return log_probabilities, entropy

class LatentPolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional Information model, this is going to be evaluated for log-probabilities alone. 
	# THIS IS STILL A SINGLE DIRECTION LSTM!!

	# This still needs to be written separately from the normal sub-policy network(s) because it also requires termination probabilities. 
	# Must change forward pass back to using lstm() directly on the entire sequence rather than iterating.
	# Now we have the whole input sequence beforehand. 

	# Policy Network inherits from torch.nn.Module.
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, b_exploration_bias=0.):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(LatentPolicyNetwork, self).__init__()

		# Input size is actually input_size + number_subpolicies +1 
		self.input_size = input_size+number_subpolicies+1
		self.offset_for_z = input_size+1
		self.hidden_size = hidden_size
		self.number_subpolicies = number_subpolicies
		self.output_size = number_subpolicies
		self.num_layers = number_layers
		self.b_exploration_bias = b_exploration_bias

		# Define LSTM. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers).cuda()

		# # Try initializing the network to something, so that we can escape the stupid constant output business.
		for name, param in self.lstm.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=5)

		# Transform to output space - Latent z and Latent b. 
		self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(self.hidden_size,2)
		
		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
	
	def forward(self, input):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float().cuda()
		hidden = None
		outputs, hidden = self.lstm(format_input)

		latent_z_preprobabilities = self.subpolicy_output_layer(outputs)
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		latent_z_probabilities = self.batch_softmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)

		latent_z_logprobabilities = self.batch_logsoftmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_logprobabilities = self.batch_logsoftmax_layer(latent_b_preprobabilities).squeeze(1)
			
		# Return log probabilities. 
		return latent_z_logprobabilities, latent_b_logprobabilities, latent_b_probabilities, latent_z_probabilities

class VariationalPolicyNetwork(PolicyNetwork_BaseClass):
	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(VariationalPolicyNetwork, self).__init__()
	
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.number_subpolicies = number_subpolicies
		self.output_size = number_subpolicies
		self.num_layers = number_layers	
		self.z_exploration_bias = z_exploration_bias
		self.b_exploration_bias = b_exploration_bias

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Transform to output space - Latent z and Latent b. 
		# THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional. 
		self.subpolicy_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(2*self.hidden_size,2)

		# Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)
			
	def sample_latent_variables(self, subpolicy_outputs, termination_output_layer):
		# Run sampling layers. 
		sample_z = self.sample_action(subpolicy_outputs)		
		sample_b = self.sample_action(termination_output_layer)
		return sample_z, sample_b 

	def sample_latent_variables_epsilon_greedy(self, subpolicy_outputs, termination_output_layer, epsilon):
		sample_z = self.select_epsilon_greedy_action(subpolicy_outputs, epsilon)
		sample_b = self.select_epsilon_greedy_action(termination_output_layer, epsilon)
		return sample_z, sample_b

	def forward(self, input, epsilon):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float().cuda()
		hidden = None
		outputs, hidden = self.lstm(format_input)

		variational_z_preprobabilities = self.subpolicy_output_layer(outputs) + self.z_exploration_bias
		variational_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		variational_z_probabilities = self.batch_softmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_probabilities = self.batch_softmax_layer(variational_b_preprobabilities).squeeze(1)

		variational_z_logprobabilities = self.batch_logsoftmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_logprobabilities = self.batch_logsoftmax_layer(variational_b_preprobabilities).squeeze(1)
		
		# sampled_z_index, sampled_b = self.sample_latent_variables(variational_z_probabilities, variational_b_probabilities)
		sampled_z_index, sampled_b = self.sample_latent_variables_epsilon_greedy(variational_z_probabilities, variational_b_probabilities, epsilon)

		return sampled_z_index, sampled_b, variational_b_logprobabilities,\
		 variational_z_logprobabilities, variational_b_probabilities, variational_z_probabilities

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		# Sampling can handle batched action_probabilities. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample()
		return sample_action

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

	def select_epsilon_greedy_action(self, action_probabilities, epsilon=0.1):
		epsilon = epsilon
		# if np.random.random()<epsilon:
		# 	# return(np.random.randint(0,high=len(action_probabilities)))
		# 	return self.sample_action(action_probabilities)
		# else:
		# 	return self.select_greedy_action(action_probabilities)

		# Issue with the current implementation is that it selects either sampling or greedy selection identically across the entire batch. 
		# This is stupid, use a toch.where instead? 
		# Sample an array of binary variables of size = batch size. 
		# For each, use greedy or ... 

		whether_greedy = torch.rand(action_probabilities.shape[0]).cuda()
		sample_actions = torch.where(whether_greedy<epsilon, self.sample_action(action_probabilities), self.select_greedy_action(action_probabilities))

		return sample_actions

	def sample_termination(self, termination_probability):
		sample_terminal = torch.distributions.Bernoulli(termination_probability).sample().squeeze(0)
		return sample_terminal

class EncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(EncoderNetwork, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_subpolicies = number_subpolicies
		self.num_layers = 5

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Define output layers for the LSTM, and activations for this output layer. 

		# Because it's bidrectional, once we compute <outputs, hidden = self.lstm(input)>, we must concatenate: 
		# From reverse LSTM: <outputs[0,:,hidden_size:]> and from the forward LSTM: <outputs[-1,:,:hidden_size]>.
		# (Refer - https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 )
		# Because of this, the output layer must take in size 2*hidden.
		self.hidden_layer = torch.nn.Linear(2*self.hidden_size, 2*self.hidden_size)
		self.output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)		

		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

	def forward(self, input):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).float().cuda()
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,1,-1))
		
		# Calculate preprobs.
		preprobabilities = self.output_layer(self.hidden_layer(concatenated_outputs))
		probabilities = self.batch_softmax_layer(preprobabilities)
		logprobabilities = self.batch_logsoftmax_layer(preprobabilities)

		latent_z = self.select_greedy_action(probabilities)

		# Return latentz_encoding as output layer of last outputs. 
		return latent_z, logprobabilities