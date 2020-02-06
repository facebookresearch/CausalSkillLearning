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

	# def select_epsilon_greedy_action(self, action_probabilities):
	# 	epsilon = 0.1
	# 	if np.random.random()<epsilon:
	# 		return self.sample_action(action_probabilities)
	# 	else:
	# 		return self.select_greedy_action(action_probabilities)

	def select_epsilon_greedy_action(self, action_probabilities, epsilon=0.1):
		epsilon = epsilon

		whether_greedy = torch.rand(action_probabilities.shape[0]).cuda()
		sample_actions = torch.where(whether_greedy<epsilon, self.sample_action(action_probabilities), self.select_greedy_action(action_probabilities))

		return sample_actions

class PolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional model, this is going to be evaluated for log-probabilities alone. 
	# Forward pass set up for evaluating this already. 

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4, batch_size=1, whether_latentb_input=False):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetwork, self).__init__()

		if whether_latentb_input:
			self.input_size = input_size+number_subpolicies+1
		else:
			self.input_size = input_size+number_subpolicies
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = number_layers
		self.batch_size = batch_size
		
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

		# Input Format must be: Sequence_Length x Batch_Size x Input_Size. 

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
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

	# def __init__(self, input_size, hidden_size, output_size, number_subpolicies, number_layers=4, batch_size=1):
	# def __init__(self, input_size, hidden_size, output_size, z_space_size, number_layers=4, batch_size=1, whether_latentb_input=False):
	def __init__(self, input_size, hidden_size, output_size, args, number_layers=4, whether_latentb_input=False, zero_z_dim=False, small_init=False):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousPolicyNetwork, self).__init__()

		self.hidden_size = hidden_size
		# The output size here must be mean+variance for each dimension. 
		# This is output_size*2. 
		self.args = args
		self.output_size = output_size
		self.num_layers = number_layers
		self.batch_size = self.args.batch_size
		
		if whether_latentb_input:
			self.input_size = input_size+self.args.z_dimensions+1
		else:
			if zero_z_dim:
				self.input_size = input_size
			else:
				self.input_size = input_size+self.args.z_dimensions
		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)		

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		# # Try initializing the network to something, so that we can escape the stupid constant output business.
		if small_init:
			for name, param in self.mean_output_layer.named_parameters():
				if 'bias' in name:
					torch.nn.init.constant_(param, 0.0)
				elif 'weight' in name:
					torch.nn.init.xavier_normal_(param,gain=0.0001)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.

		self.variance_factor = 0.01

	def forward(self, input, action_sequence, epsilon=0.001):
		# Input is the trajectory sequence of shape: Sequence_Length x 1 x Input_Size. 
		# Here, we also need the continuous actions as input to evaluate their logprobability / probability. 		
		# format_input = torch.tensor(input).view(input.shape[0], self.batch_size, self.input_size).float().cuda()
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		format_action_seq = torch.from_numpy(action_sequence).cuda().float().view(action_sequence.shape[0],1,self.output_size)
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)
		# variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias) + epsilon

		# Remember, because of Pytorch's dynamic construction, this distribution can have it's own batch size. 
		# It doesn't matter if batch sizes changes over different forward passes of the LSTM, because we're only going
		# to evaluate this distribution (instance)'s log probability with the same sequence length. 
		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))
		log_probabilities = dist.log_prob(format_action_seq)
		# log_probabilities = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs)).log_prob(format_action_seq)
		entropy = dist.entropy()

		if self.args.debug:
			print("Embedding in the policy network.")		
			embed()
			
		return log_probabilities, entropy

	def get_actions(self, input, greedy=False):
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		if greedy:
			return mean_outputs
		else:

			# Remember, because of Pytorch's dynamic construction, this distribution can have it's own batch size. 
			# It doesn't matter if batch sizes changes over different forward passes of the LSTM, because we're only going
			# to evaluate this distribution (instance)'s log probability with the same sequence length. 
			dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

			return dist.sample()

	def reparameterized_get_actions(self, input, greedy=False, action_epsilon=0.):
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			action = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = mean_outputs + variance_outputs * noise

		return action

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0., hidden=None):
		
		# Input should be a single timestep input here. 
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		# Instead of feeding in entire input sequence, we are feeding in current timestep input and previous hidden state.
		lstm_outputs, hidden = self.lstm(format_input, hidden)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(lstm_outputs))
		else:
			mean_outputs = self.mean_output_layer(lstm_outputs)
		variance_outputs = (self.variance_activation_layer(self.variances_output_layer(lstm_outputs))+self.variance_activation_bias)

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			action = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = mean_outputs + variance_outputs * noise

		return action, hidden

	def get_regularization_kl(self, input_z1, input_z2):
		# Input is the trajectory sequence of shape: Sequence_Length x 1 x Input_Size. 
		# Here, we also need the continuous actions as input to evaluate their logprobability / probability. 		
		format_input_z1 = input_z1.view(input_z1.shape[0], self.batch_size, self.input_size)
		format_input_z2 = input_z2.view(input_z2.shape[0], self.batch_size, self.input_size)

		hidden = None
		# format_action_seq = torch.from_numpy(action_sequence).cuda().float().view(action_sequence.shape[0],1,self.output_size)
		lstm_outputs_z1, _ = self.lstm(format_input_z1)
		# Reset hidden? 
		lstm_outputs_z2, _ = self.lstm(format_input_z2)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs_z1 = self.activation_layer(self.mean_output_layer(lstm_outputs_z1))
			mean_outputs_z2 = self.activation_layer(self.mean_output_layer(lstm_outputs_z2))
		else:
			mean_outputs_z1 = self.mean_output_layer(lstm_outputs_z1)
			mean_outputs_z2 = self.mean_output_layer(lstm_outputs_z2)
		variance_outputs_z1 = self.variance_activation_layer(self.variances_output_layer(lstm_outputs_z1))+self.variance_activation_bias
		variance_outputs_z2 = self.variance_activation_layer(self.variances_output_layer(lstm_outputs_z2))+self.variance_activation_bias

		dist_z1 = torch.distributions.MultivariateNormal(mean_outputs_z1, torch.diag_embed(variance_outputs_z1))
		dist_z2 = torch.distributions.MultivariateNormal(mean_outputs_z2, torch.diag_embed(variance_outputs_z2))

		kl_divergence = torch.distributions.kl_divergence(dist_z1, dist_z2)

		return kl_divergence

class LatentPolicyNetwork(PolicyNetwork_BaseClass):

	# REMEMBER, in the Bi-directional Information model, this is going to be evaluated for log-probabilities alone. 
	# THIS IS STILL A SINGLE DIRECTION LSTM!!

	# This still needs to be written separately from the normal sub-policy network(s) because it also requires termination probabilities. 
	# Must change forward pass back to using lstm() directly on the entire sequence rather than iterating.
	# Now we have the whole input sequence beforehand. 

	# Policy Network inherits from torch.nn.Module.
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, b_exploration_bias=0., batch_size=1):

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
		self.batch_size = batch_size

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
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
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

	def get_actions(self, input, greedy=False):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		latent_z_preprobabilities = self.subpolicy_output_layer(outputs)
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		latent_z_probabilities = self.batch_softmax_layer(latent_z_preprobabilities).squeeze(1)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)

		if greedy==True:
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = self.select_greedy_action(latent_z_probabilities)
		else:
			selected_b = self.sample_action(latent_b_probabilities)
			selected_z = self.sample_action(latent_z_probabilities)
		
		return selected_b, selected_z

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

class ContinuousLatentPolicyNetwork(PolicyNetwork_BaseClass):

	# def __init__(self, input_size, hidden_size, z_dimensions, number_layers=4, b_exploration_bias=0., batch_size=1):
	def __init__(self, input_size, hidden_size, args, number_layers=4):		

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousLatentPolicyNetwork, self).__init__()

		self.args = args
		# Input size is actually input_size + number_subpolicies +1 
		self.input_size = input_size+self.args.z_dimensions+1
		self.offset_for_z = input_size+1
		self.hidden_size = hidden_size
		# self.number_subpolicies = number_subpolicies
		self.output_size = self.args.z_dimensions
		self.num_layers = number_layers
		self.b_exploration_bias = self.args.b_exploration_bias
		self.batch_size = self.args.batch_size

		# Define LSTM. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers).cuda()

		# Transform to output space - Latent z and Latent b. 
		# self.subpolicy_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.termination_output_layer = torch.nn.Linear(self.hidden_size,2)
		
		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.
			
		self.variance_factor = 0.01

		# # # Try initializing the network to something, so that we can escape the stupid constant output business.
		for name, param in self.lstm.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.001)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=5)

		# Also initializing mean_output_layer to something large...
		for name, param in self.mean_output_layer.named_parameters():
			if 'bias' in name:
				torch.nn.init.constant_(param, 0.)
			elif 'weight' in name:
				torch.nn.init.xavier_normal_(param,gain=2)

	def forward(self, input, epsilon=0.001):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
		latent_b_logprobabilities = self.batch_logsoftmax_layer(latent_b_preprobabilities).squeeze(1)
			
		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		if self.args.debug:
			print("Embedding in Latent Policy.")
			embed()
		# Return log probabilities. 
		return latent_b_logprobabilities, latent_b_probabilities, self.dists

	def get_actions(self, input, greedy=False, epsilon=0.001):
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
			
		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		if greedy==True:
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = mean_outputs
		else:
			# selected_b = self.sample_action(latent_b_probabilities)
			selected_b = self.select_greedy_action(latent_b_probabilities)
			selected_z = self.dists.sample()

		return selected_b, selected_z

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None, previous_z=None):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		outputs, hidden = self.lstm(format_input, hidden)

		latent_b_preprobabilities = self.termination_output_layer(outputs)
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	
		# Greedily select b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)

		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + action_epsilon

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			selected_z = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			selected_z = mean_outputs + variance_outputs * noise

		# If single input and previous_Z is None, this is the first timestep. So set b to 1, and don't do anything to z. 
		if input.shape[0]==1 and previous_z is None:
			selected_b[0] = 1
		# If previous_Z is not None, this is not the first timestep, so don't do anything to z. If b is 0, use previous. 
		elif input.shape[0]==1 and previous_z is not None:
			if selected_b==0:
				selected_z = previous_z
		elif input.shape[0]>1:
			# Now modify z's as per New Z Selection. 
			# Set initial b to 1. 
			selected_b[0] = 1
			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if selected_b[t]==0:
					selected_z[t] = selected_z[t-1]

		return selected_z, selected_b, hidden

	def reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None):

		# Wraps incremental 
		# MUST MODIFY INCREMENTAL ONE TO HANDLE NEW_Z_SELECTION (i.e. only choose new one if b is 1....)

			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]

	def select_greedy_action(self, action_probabilities):
		# Select action with max probability for test time. 
		# NEED TO USE DIMENSION OF ARGMAX. 
		return action_probabilities.argmax(dim=-1)

class ContinuousLatentPolicyNetwork_ConstrainedBPrior(ContinuousLatentPolicyNetwork):

	def __init__(self, input_size, hidden_size, args, number_layers=4):		

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousLatentPolicyNetwork_ConstrainedBPrior, self).__init__(input_size, hidden_size, args, number_layers)

		# We can inherit the forward function from the above class... we just need to modify get actions.	

		self.min_skill_time = 12
		self.max_skill_time = 16

	def get_prior_value(self, elapsed_t, max_limit=5):

		skill_time_limit = max_limit-1

		if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				lens = np.array([12,13,14,15,16])
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 16
				skill_time_limit = 12

			else:
				max_limit = 20
				skill_time_limit = max_limit-1	

		prior_value = torch.zeros((1,2)).cuda().float()
		# If at or over hard limit.
		if elapsed_t>=max_limit:
			prior_value[0,1]=1.

		# If at or more than typical, less than hard limit:
		elif elapsed_t>=skill_time_limit:
	
			if self.args.var_skill_length:
				prior_value[0] = torch.tensor(prob_biases[elapsed_t-skill_time_limit]).cuda().float()
			else:
				# Random
				prior_value[0,1]=0. 

		# If less than typical. 
		else:
			# Continue.
			prior_value[0,0]=1.

		return prior_value

	def get_actions(self, input, greedy=False, epsilon=0.001, delta_t=0):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)
	
		latent_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias		
			
		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))	

		############################################
		prior_value = self.get_prior_value(delta_t)

		# Now... add prior value.
		# Only need to do this to the last timestep... because the last sampled b is going to be copied into a different variable that is stored.
		latent_b_preprobabilities[-1, :, :] += prior_value		
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	

		# Sample b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)
		############################################

		# Now implementing hard constrained b selection.
		if delta_t < self.min_skill_time:
			# Continue. Set b to 0.
			selected_b[-1] = 0.

		elif (self.min_skill_time <= delta_t) and (delta_t < self.max_skill_time):
			pass

		else: 
			# Stop and select a new z. Set b to 1. 
			selected_b[-1] = 1.

		# Also get z... assume higher level funciton handles the new z selection component. 
		if greedy==True:
			selected_z = mean_outputs
		else:
			selected_z = self.dists.sample()

		return selected_b, selected_z

	def incremental_reparam_get_actions(self, input, greedy=False, action_epsilon=0.001, hidden=None, previous_z=None, delta_t=0):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		outputs, hidden = self.lstm(format_input, hidden)

		latent_b_preprobabilities = self.termination_output_layer(outputs)
		
		############################################
		# GET PRIOR AND ADD. 
		prior_value = self.get_prior_value(delta_t)
		latent_b_preprobabilities[-1, :, :] += prior_value			
		############################################
		latent_b_probabilities = self.batch_softmax_layer(latent_b_preprobabilities).squeeze(1)	

		# Greedily select b. 
		selected_b = self.select_greedy_action(latent_b_probabilities)

		# Predict Gaussian means and variances. 		
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# We should be multiply by self.variance_factor.
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + action_epsilon

		noise = torch.randn_like(variance_outputs)

		if greedy: 
			selected_z = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			selected_z = mean_outputs + variance_outputs * noise

		# If single input and previous_Z is None, this is the first timestep. So set b to 1, and don't do anything to z. 
		if input.shape[0]==1 and previous_z is None:
			selected_b[0] = 1
		# If previous_Z is not None, this is not the first timestep, so don't do anything to z. If b is 0, use previous. 
		elif input.shape[0]==1 and previous_z is not None:
			if selected_b==0:
				selected_z = previous_z
		elif input.shape[0]>1:
			# Now modify z's as per New Z Selection. 
			# Set initial b to 1. 
			selected_b[0] = 1
			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if selected_b[t]==0:
					selected_z[t] = selected_z[t-1]

		return selected_z, selected_b, hidden

class VariationalPolicyNetwork(PolicyNetwork_BaseClass):
	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	# def __init__(self, input_size, hidden_size, number_subpolicies, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.,  batch_size=1):
	def __init__(self, input_size, hidden_size, number_subpolicies, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(VariationalPolicyNetwork, self).__init__()
		
		self.args = args
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.number_subpolicies = number_subpolicies
		self.output_size = number_subpolicies
		self.num_layers = number_layers	
		self.z_exploration_bias = self.args.z_exploration_bias
		self.b_exploration_bias = self.args.b_exploration_bias
		self.z_probability_factor = self.args.z_probability_factor
		self.b_probability_factor = self.args.b_probability_factor
		self.batch_size = self.args.batch_size

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

	def forward(self, input, epsilon, new_z_selection=True):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_z_preprobabilities = self.subpolicy_output_layer(outputs)*self.z_probability_factor + self.z_exploration_bias
		# variational_b_preprobabilities = self.termination_output_layer(outputs) + self.b_exploration_bias

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor
		# Add b continuation bias to the continuing option at every timestep. 
		variational_b_preprobabilities[:,0,0] += self.b_exploration_bias

		variational_z_probabilities = self.batch_softmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_probabilities = self.batch_softmax_layer(variational_b_preprobabilities).squeeze(1)

		variational_z_logprobabilities = self.batch_logsoftmax_layer(variational_z_preprobabilities).squeeze(1)
		variational_b_logprobabilities = self.batch_logsoftmax_layer(variational_b_preprobabilities).squeeze(1)
		
		# sampled_z_index, sampled_b = self.sample_latent_variables(variational_z_probabilities, variational_b_probabilities)
		sampled_z_index, sampled_b = self.sample_latent_variables_epsilon_greedy(variational_z_probabilities, variational_b_probabilities, epsilon)

		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# # Trying cheeky thing to see if we can learn in this setting.
			# sampled_b[1:] = 0

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]

		return sampled_z_index, sampled_b, variational_b_logprobabilities,\
		 variational_z_logprobabilities, variational_b_probabilities, variational_z_probabilities, None

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

class ContinuousVariationalPolicyNetwork(PolicyNetwork_BaseClass):

	# def __init__(self, input_size, hidden_size, z_dimensions, number_layers=4, z_exploration_bias=0., b_exploration_bias=0.,  batch_size=1):
	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):
		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		# super().__init__()
		super(ContinuousVariationalPolicyNetwork, self).__init__()
	
		self.args = args	
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = z_dimensions
		self.num_layers = number_layers	
		self.z_exploration_bias = self.args.z_exploration_bias
		self.b_exploration_bias = self.args.b_exploration_bias
		self.z_probability_factor = self.args.z_probability_factor
		self.b_probability_factor = self.args.b_probability_factor
		self.batch_size = self.args.batch_size

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Transform to output space - Latent z and Latent b. 
		# THIS OUTPUT LAYER TAKES 2*HIDDEN SIZE as input because it's bidirectional. 
		self.termination_output_layer = torch.nn.Linear(2*self.hidden_size,2)

		# Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=-1)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()

		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.
			
		self.variance_factor = 0.01

	def forward(self, input, epsilon, new_z_selection=True, var_epsilon=0.001):
		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Add b continuation bias to the continuing option at every timestep. 
		variational_b_preprobabilities[:,0,0] += self.b_exploration_bias
		variational_b_probabilities = self.batch_softmax_layer(variational_b_preprobabilities).squeeze(1)		
		variational_b_logprobabilities = self.batch_logsoftmax_layer(variational_b_preprobabilities).squeeze(1)

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + var_epsilon

		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		sampled_b = self.select_epsilon_greedy_action(variational_b_probabilities, epsilon)

		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		else:

			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.reparam:

				if self.args.train:
					noise = torch.randn_like(variance_outputs)

					# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
					sampled_z_index = mean_outputs + variance_outputs*noise
					# Ought to be able to pass gradients through this latent_z now.

					sampled_z_index = sampled_z_index.squeeze(1)

				# If evaluating, greedily get action.
				else:
					sampled_z_index = mean_outputs.squeeze(1)
			else:
				sampled_z_index = self.dists.sample().squeeze(1)
		
		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = standard_distribution.log_prob(sampled_z_index)

		# if self.args.debug:
		# 	print("#################################")
		# 	print("Embedding in Variational Network.")
		# 	embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities,\
		 variational_z_logprobabilities, variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood

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

class ContinuousVariationalPolicyNetwork_BPrior(ContinuousVariationalPolicyNetwork):
	
	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousVariationalPolicyNetwork_BPrior, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

	def get_prior_value(self, elapsed_t, max_limit=5):

		skill_time_limit = max_limit-1

		if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				lens = np.array([12,13,14,15,16])
				# probabilities = np.array([0.1,0.2,0.4,0.2,0.1])
				prob_biases = np.array([[0.8,0.],[0.4,0.],[0.,0.],[0.,0.4]])				

				max_limit = 16
				skill_time_limit = 12

			else:
				max_limit = 20
				skill_time_limit = max_limit-1	

		prior_value = torch.zeros((1,2)).cuda().float()
		# If at or over hard limit.
		if elapsed_t>=max_limit:
			prior_value[0,1]=1.

		# If at or more than typical, less than hard limit:
		elif elapsed_t>=skill_time_limit:
	
			if self.args.var_skill_length:
				prior_value[0] = torch.tensor(prob_biases[elapsed_t-skill_time_limit]).cuda().float()
			else:
				# Random
				prior_value[0,1]=0. 

		# If less than typical. 
		else:
			# Continue.
			prior_value[0,0]=1.

		return prior_value

	def forward(self, input, epsilon, new_z_selection=True):

		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon
		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		prev_time = 0
		# Create variables for prior and probs.
		prior_values = torch.zeros_like(variational_b_preprobabilities).cuda().float()
		variational_b_probabilities = torch.zeros_like(variational_b_preprobabilities).cuda().float()
		variational_b_logprobabilities = torch.zeros_like(variational_b_preprobabilities).cuda().float()
		sampled_b = torch.zeros(input.shape[0]).cuda().int()
		sampled_b[0] = 1
		
		for t in range(1,input.shape[0]):

			# Compute prior value. 
			delta_t = t-prev_time

			# if self.args.debug:
			# 	print("##########################")
			# 	print("Time: ",t, " Prev Time:",prev_time, " Delta T:",delta_t)

			prior_values[t] = self.get_prior_value(delta_t, max_limit=self.args.skill_length)

			# Construct probabilities.
			variational_b_probabilities[t,0,:] = self.batch_softmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			variational_b_logprobabilities[t,0,:] = self.batch_logsoftmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			
			sampled_b[t] = self.select_epsilon_greedy_action(variational_b_probabilities[t:t+1], epsilon)

			if sampled_b[t]==1:
				prev_time = t				

			# if self.args.debug:
			# 	print("Sampled b:",sampled_b[t])

		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		else:

			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.reparam:

				if self.args.train:
					noise = torch.randn_like(variance_outputs)

					# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
					sampled_z_index = mean_outputs + variance_outputs*noise
					# Ought to be able to pass gradients through this latent_z now.

					sampled_z_index = sampled_z_index.squeeze(1)

				# If evaluating, greedily get action.
				else:
					sampled_z_index = mean_outputs.squeeze(1)
			else:
				sampled_z_index = self.dists.sample().squeeze(1)
		
		if new_z_selection:
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = standard_distribution.log_prob(sampled_z_index)

		if self.args.debug:
			print("#################################")
			print("Embedding in Variational Network.")
			embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities.squeeze(1), \
		 variational_z_logprobabilities, variational_b_probabilities.squeeze(1), variational_z_probabilities, kl_divergence, prior_loglikelihood

class ContinuousVariationalPolicyNetwork_ConstrainedBPrior(ContinuousVariationalPolicyNetwork_BPrior):

	def __init__(self, input_size, hidden_size, z_dimensions, args, number_layers=4):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousVariationalPolicyNetwork_ConstrainedBPrior, self).__init__(input_size, hidden_size, z_dimensions, args, number_layers)

		self.min_skill_time = 12
		self.max_skill_time = 16

	def forward(self, input, epsilon, new_z_selection=True):

		# Input Format must be: Sequence_Length x 1 x Input_Size. 	
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		hidden = None
		outputs, hidden = self.lstm(format_input)

		# Damping factor for probabilities to prevent washing out of bias. 
		variational_b_preprobabilities = self.termination_output_layer(outputs)*self.b_probability_factor

		# Predict Gaussian means and variances. 
		if self.args.mean_nonlinearity:
			mean_outputs = self.activation_layer(self.mean_output_layer(outputs))
		else:
			mean_outputs = self.mean_output_layer(outputs)
		# Still need a softplus activation for variance because needs to be positive. 
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(outputs))+self.variance_activation_bias) + epsilon
		# This should be a SET of distributions. 
		self.dists = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		# Create variables for prior and probabilities.
		prior_values = torch.zeros_like(variational_b_preprobabilities).cuda().float()
		variational_b_probabilities = torch.zeros_like(variational_b_preprobabilities).cuda().float()
		variational_b_logprobabilities = torch.zeros_like(variational_b_preprobabilities).cuda().float()

		#######################################
		################ Set B ################
		#######################################

		# Set the first b to 1, and the time b was == 1. 		
		sampled_b = torch.zeros(input.shape[0]).cuda().int()
		sampled_b[0] = 1
		prev_time = 0

		for t in range(1,input.shape[0]):
			
			# Compute time since the last b occurred. 			
			delta_t = t-prev_time
			# Compute prior value. 
			prior_values[t] = self.get_prior_value(delta_t, max_limit=self.args.skill_length)

			# Construct probabilities.
			variational_b_probabilities[t,0,:] = self.batch_softmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
			variational_b_logprobabilities[t,0,:] = self.batch_logsoftmax_layer(variational_b_preprobabilities[t,0] + prior_values[t,0])
	
			# Now Implement Hard Restriction on Selection of B's. 
			if delta_t < self.min_skill_time:
				# Set B to 0. I.e. Continue. 
				# variational_b_probabilities[t,0,:] = variational_b_probabilities[t,0,:]*0
				# variational_b_probabilities[t,0,0] += 1
				
				sampled_b[t] = 0.

			elif (self.min_skill_time <= delta_t) and (delta_t < self.max_skill_time):		
				# Sample b. 			
				sampled_b[t] = self.select_epsilon_greedy_action(variational_b_probabilities[t:t+1], epsilon)

			elif self.max_skill_time <= delta_t:
				# Set B to 1. I.e. select new z. 
				sampled_b[t] = 1.

			# If b is 1, set the previous time to now. 
			if sampled_b[t]==1:
				prev_time = t				

		#######################################
		################ Set Z ################
		#######################################

		# Now set the z's. If greedy, just return the means. 
		if epsilon==0.:
			sampled_z_index = mean_outputs.squeeze(1)
		# If not greedy, then reparameterize. 
		else:
			# Whether to use reparametrization trick to retrieve the latent_z's.
			if self.args.train:
				noise = torch.randn_like(variance_outputs)

				# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
				sampled_z_index = mean_outputs + variance_outputs*noise
				# Ought to be able to pass gradients through this latent_z now.

				sampled_z_index = sampled_z_index.squeeze(1)

			# If evaluating, greedily get action.
			else:
				sampled_z_index = mean_outputs.squeeze(1)
		
		# Modify z's based on whether b was 1 or 0. This part should remain the same.		
		if new_z_selection:
			
			# Set initial b to 1. 
			sampled_b[0] = 1

			# Initial z is already trivially set. 
			for t in range(1,input.shape[0]):
				# If b_t==0, just use previous z. 
				# If b_t==1, sample new z. Here, we've cloned this from sampled_z's, so there's no need to do anything. 
				if sampled_b[t]==0:
					sampled_z_index[t] = sampled_z_index[t-1]		

		# Also compute logprobabilities of the latent_z's sampled from this net. 
		variational_z_logprobabilities = self.dists.log_prob(sampled_z_index.unsqueeze(1))
		variational_z_probabilities = None

		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(self.dists, standard_distribution)

		# Prior loglikelihood
		prior_loglikelihood = standard_distribution.log_prob(sampled_z_index)

		if self.args.debug:
			print("#################################")
			print("Embedding in Variational Network.")
			embed()

		return sampled_z_index, sampled_b, variational_b_logprobabilities.squeeze(1), \
		 variational_z_logprobabilities, variational_b_probabilities.squeeze(1), variational_z_probabilities, kl_divergence, prior_loglikelihood

class EncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, number_subpolicies=4, batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(EncoderNetwork, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_subpolicies = number_subpolicies
		self.num_layers = 5
		self.batch_size = batch_size

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

	def forward(self, input, epsilon):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,1,-1))
		
		# Calculate preprobs.
		preprobabilities = self.output_layer(self.hidden_layer(concatenated_outputs))
		probabilities = self.batch_softmax_layer(preprobabilities)
		logprobabilities = self.batch_logsoftmax_layer(preprobabilities)

		latent_z = self.select_epsilon_greedy_action(probabilities, epsilon=epsilon)

		# Return latentz_encoding as output layer of last outputs. 
		return latent_z, logprobabilities, None, None

class ContinuousEncoderNetwork(PolicyNetwork_BaseClass):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, input_size, hidden_size, output_size, args, batch_size=1):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(ContinuousEncoderNetwork, self).__init__()

		self.args = args
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = 5
		self.batch_size = batch_size

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers, bidirectional=True)

		# Define output layers for the LSTM, and activations for this output layer. 

		# # Because it's bidrectional, once we compute <outputs, hidden = self.lstm(input)>, we must concatenate: 
		# # From reverse LSTM: <outputs[0,:,hidden_size:]> and from the forward LSTM: <outputs[-1,:,:hidden_size]>.
		# # (Refer - https://towardsdatascience.com/understanding-bidirectional-rnn-in-pytorch-5bd25a5dd66 )
		# # Because of this, the output layer must take in size 2*hidden.
		# self.hidden_layer = torch.nn.Linear(2*self.hidden_size, self.hidden_size)
		# self.output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)		

		# Sigmoid and Softmax activation functions for Bernoulli termination probability and latent z selection .
		self.batch_softmax_layer = torch.nn.Softmax(dim=2)
		self.batch_logsoftmax_layer = torch.nn.LogSoftmax(dim=2)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.mean_output_layer = torch.nn.Linear(2*self.hidden_size,self.output_size)
		self.variances_output_layer = torch.nn.Linear(2*self.hidden_size, self.output_size)

		self.activation_layer = torch.nn.Tanh()
		self.variance_activation_layer = torch.nn.Softplus()
		self.variance_activation_bias = 0.

		self.variance_factor = 0.01

	def forward(self, input, epsilon=0.001, z_sample_to_evaluate=None):
		# This epsilon passed as an argument is just so that the signature of this function is the same as what's provided to the discrete encoder network.

		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array.
		format_input = input.view((input.shape[0], self.batch_size, self.input_size))
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input)

		concatenated_outputs = torch.cat([outputs[0,:,self.hidden_size:],outputs[-1,:,:self.hidden_size]],dim=-1).view((1,1,-1))

		# Predict Gaussian means and variances. 
		# if self.args.mean_nonlinearity:
		# 	mean_outputs = self.activation_layer(self.mean_output_layer(concatenated_outputs))
		# else:
		mean_outputs = self.mean_output_layer(concatenated_outputs)
		variance_outputs = self.variance_factor*(self.variance_activation_layer(self.variances_output_layer(concatenated_outputs))+self.variance_activation_bias) + epsilon

		dist = torch.distributions.MultivariateNormal(mean_outputs, torch.diag_embed(variance_outputs))

		# Whether to use reparametrization trick to retrieve the 
		if self.args.reparam:
			noise = torch.randn_like(variance_outputs)

			# Instead of *sampling* the latent z from a distribution, construct using mu + sig * eps (random noise).
			latent_z = mean_outputs + variance_outputs * noise
			# Ought to be able to pass gradients through this latent_z now.

		else:
			# Retrieve sample from the distribution as the value of the latent variable.
			latent_z = dist.sample()
		# calculate entropy for training.
		entropy = dist.entropy()
		# Also retrieve log probability of the same.
		logprobability = dist.log_prob(latent_z)

		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(dist, standard_distribution)

		if self.args.debug:
			print("###############################")
			print("Embedding in Encoder Network.")
			embed()

		if z_sample_to_evaluate is None:
			return latent_z, logprobability, entropy, kl_divergence

		else:
			logprobability = dist.log_prob(z_sample_to_evaluate)
			return logprobability

class CriticNetwork(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(CriticNetwork, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.number_layers = number_layers
		self.batch_size = 1

		# Create LSTM Network. 
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.number_layers)		

		self.output_layer = torch.nn.Linear(self.hidden_size,self.output_size)		

	def forward(self, input):

		format_input = input.view((input.shape[0], self.batch_size, self.input_size))

		hidden = None
		lstm_outputs, hidden = self.lstm(format_input)

		# Predict critic value for each timestep. 
		critic_value = self.output_layer(lstm_outputs)		

		return critic_value

class ContinuousMLP(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(ContinuousMLP, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
		self.relu_activation = torch.nn.ReLU()
		self.variance_activation_layer = torch.nn.Softplus()

	def forward(self, input, greedy=False, action_epsilon=0.0001):

		# Assumes input is Batch_Size x Input_Size.
		h1 = self.relu_activation(self.input_layer(input))
		h2 = self.relu_activation(self.hidden_layer(h1))
		h3 = self.relu_activation(self.hidden_layer(h2))
		h4 = self.relu_activation(self.hidden_layer(h3))

		mean_outputs = self.output_layer(h4)
		variance_outputs = self.variance_activation_layer(self.output_layer(h4))
		
		noise = torch.randn_like(variance_outputs)

		if greedy: 
			action = mean_outputs
		else:
			# Instead of *sampling* the action from a distribution, construct using mu + sig * eps (random noise).
			action = mean_outputs + variance_outputs * noise

		return action

	def reparameterized_get_actions(self, input, greedy=False, action_epsilon=0.0001):
		return self.forward(input, greedy, action_epsilon)

class CriticMLP(torch.nn.Module):

	def __init__(self, input_size, hidden_size, output_size, args=None, number_layers=4):

		super(CriticMLP, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = 1

		self.input_layer = torch.nn.Linear(self.input_size, self.hidden_size)
		self.hidden_layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)
		self.relu_activation = torch.nn.ReLU()

	def forward(self, input):

		# Assumes input is Batch_Size x Input_Size.
		h1 = self.relu_activation(self.input_layer(input))
		h2 = self.relu_activation(self.hidden_layer(h1))
		h3 = self.relu_activation(self.hidden_layer(h2))
		h4 = self.relu_activation(self.hidden_layer(h3))

		# Predict critic value for each timestep. 
		critic_value = self.output_layer(h4)		

		return critic_value
