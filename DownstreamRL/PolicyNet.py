# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from ..SkillNetwork.headers import *
from ..SkillNetwork.LSTMNetwork import LSTMNetwork, LSTMNetwork_Fixed

class PolicyNetwork(torch.nn.Module):

	def __init__(self, opts, input_size, hidden_size, output_size, fixed=True):
		
		super(PolicyNetwork, self).__init__()

		self.opts = opts 
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size

		if fixed:
			self.lstmnet = LSTMNetwork_Fixed(input_size=input_size, hidden_size=hidden_size, output_size=output_size).cuda()
		else:
			self.lstmnet = LSTMNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size).cuda()

		# Create linear layer to split prediction into mu and sigma. 
		self.mu_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)
		self.sig_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)

		# Stopping probability predictor. (Softmax, not sigmoid)
		self.stopping_probability_layer = torch.nn.Linear(self.hidden_size, 2)	
		self.softmax_layer = torch.nn.Softmax(dim=-1)

	def forward(self, input):

		format_input = torch.tensor(input).view(1,1,self.input_size).cuda().float()
		predicted_Z_preparam, stop_probabilities = self.lstmnet.forward(format_input)

		predicted_Z_preparam = predicted_Z_preparam.squeeze(1)

		self.latent_z_seq = []
		self.latent_mu_seq = []
		self.latent_log_sigma_seq = []
		self.kld_loss = 0.

		t = 0

		# Remember, the policy is Gaussian (so we can implement VAE-KLD on it).
		latent_z_mu_seq = self.mu_linear_layer(predicted_Z_preparam)
		latent_z_log_sig_seq = self.sig_linear_layer(predicted_Z_preparam)	

		# Compute standard deviation. 
		std = torch.exp(0.5*latent_z_log_sig_seq).cuda()
		# Sample random variable. 
		eps = torch.randn_like(std).cuda()

		self.latent_z_seq = latent_z_mu_seq+eps*std

		# Compute KL Divergence Loss term here, so we don't have to return mu's and sigma's. 
		self.kld_loss = torch.zeros(1)
		for t in range(latent_z_mu_seq.shape[0]):
			# Taken from mime_plan_skill.py Line 159 - KL Divergence for Gaussian prior and Gaussian prediction. 
			self.kld_loss += -0.5 * torch.sum(1. + latent_z_log_sig_seq[t] - latent_z_mu_seq[t].pow(2) - latent_z_log_sig_seq[t].exp())
	
		# Create distributions so that we can evaluate log probability. 	
		self.dists = [torch.distributions.MultivariateNormal(loc = latent_z_mu_seq[t], covariance_matrix = std[t]*torch.eye((self.opts.nz)).cuda()) for t in range(latent_z_mu_seq.shape[0])]

		# Evaluate log probability in forward so we don't have to do it elswhere. 
		self.log_probs = [self.dists[i].log_prob(self.latent_z_seq[i]) for i in range(self.latent_z_seq.shape[0])]

		return self.latent_z_seq, stop_probabilities

class PolicyNetworkSingleTimestep(torch.nn.Module):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, opts, input_size, hidden_size, output_size):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(PolicyNetworkSingleTimestep, self).__init__()

		self.opts = opts
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = 4
		self.maximum_length = 15

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)		
		# Create linear layer to split prediction into mu and sigma. 
		self.mu_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)
		self.sig_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)

		# Stopping probability predictor. (Softmax, not sigmoid)
		self.stopping_probability_layer = torch.nn.Linear(self.hidden_size, 2)	

		self.softmax_layer = torch.nn.Softmax(dim=-1)
		self.logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

	def forward(self, input, hidden=None):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).cuda().float()
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input, hidden)

		# Predict parameters	
		latentz_preparam = self.output_layer(outputs[-1])
		# Remember, the policy is Gaussian (so we can implement VAE-KLD on it).
		latent_z_mu = self.mu_linear_layer(latentz_preparam)
		latent_z_log_sig = self.sig_linear_layer(latentz_preparam)	

		# Predict stop probability. 
		preact_stop_probs = self.stopping_probability_layer(outputs[-1])
		stop_probability = self.softmax_layer(preact_stop_probs)

		stop = self.sample_action(stop_probability)

		# Remember, the policy is Gaussian (so we can implement VAE-KLD on it).
		latent_z_mu = self.mu_linear_layer(latentz_preparam)
		latent_z_log_sig = self.sig_linear_layer(latentz_preparam)	

		# Compute standard deviation. 
		std = torch.exp(0.5*latent_z_log_sig).cuda()
		# Sample random variable. 
		eps = torch.randn_like(std).cuda()

		latent_z = latent_z_mu+eps*std

		# Compute KL Divergence Loss term here, so we don't have to return mu's and sigma's. 
		# Taken from mime_plan_skill.py Line 159 - KL Divergence for Gaussian prior and Gaussian prediction. 
		kld_loss = -0.5 * torch.sum(1. + latent_z_log_sig - latent_z_mu.pow(2) - latent_z_log_sig.exp())
	
		# Create distributions so that we can evaluate log probability. 	
		dist = torch.distributions.MultivariateNormal(loc = latent_z_mu, covariance_matrix = std*torch.eye((self.opts.nz)).cuda())

		# Evaluate log probability in forward so we don't have to do it elswhere. 
		log_prob = dist.log_prob(latent_z)

		return latent_z, stop_probability, stop, log_prob, kld_loss, hidden

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample().squeeze(0)
		return sample_action

class AltPolicyNetworkSingleTimestep(torch.nn.Module):

	# Policy Network inherits from torch.nn.Module. 
	# Now we overwrite the init, forward functions. And define anything else that we need. 

	def __init__(self, opts, input_size, hidden_size, output_size):

		# Ensures inheriting from torch.nn.Module goes nicely and cleanly. 	
		super(AltPolicyNetworkSingleTimestep, self).__init__()

		self.opts = opts
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = 4
		self.maximum_length = 15

		# Define a bidirectional LSTM now.
		self.lstm = torch.nn.LSTM(input_size=self.input_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

		# Define output layers for the LSTM, and activations for this output layer. 
		self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)		
		# Create linear layer to split prediction into mu and sigma. 
		self.mu_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)
		self.sig_linear_layer = torch.nn.Linear(self.opts.nz, self.opts.nz)
		self.softplus_activation_layer = torch.nn.Softplus()

		# Stopping probability predictor. (Softmax, not sigmoid)
		self.stopping_probability_layer = torch.nn.Linear(self.hidden_size, 2)	

		self.softmax_layer = torch.nn.Softmax(dim=-1)
		self.logsoftmax_layer = torch.nn.LogSoftmax(dim=-1)

	def forward(self, input, hidden=None):
		# Input format must be: Sequence_Length x 1 x Input_Size. 
		# Assuming input is a numpy array. 		
		format_input = torch.tensor(input).view(input.shape[0],1,self.input_size).cuda().float()
		
		# Instead of iterating over time and passing each timestep's input to the LSTM, we can now just pass the entire input sequence.
		outputs, hidden = self.lstm(format_input, hidden)

		# Predict parameters	
		latentz_preparam = self.output_layer(outputs[-1])
		# Remember, the policy is Gaussian (so we can implement VAE-KLD on it).
		latent_z_mu = self.mu_linear_layer(latentz_preparam)
		latent_z_log_sig = self.sig_linear_layer(latentz_preparam)	
		latent_z_sig = self.softplus_activation_layer(self.sig_linear_layer(latentz_preparam))

		# Predict stop probability. 
		preact_stop_probs = self.stopping_probability_layer(outputs[-1])
		stop_probability = self.softmax_layer(preact_stop_probs)

		stop = self.sample_action(stop_probability)

		# Create distributions so that we can evaluate log probability. 	
		dist = torch.distributions.MultivariateNormal(loc = latent_z_mu, covariance_matrix = torch.diag_embed(latent_z_sig))

		latent_z = dist.sample()
		
		# Evaluate log probability in forward so we don't have to do it elswhere. 
		log_prob = dist.log_prob(latent_z)


		# Set standard distribution for KL. 
		standard_distribution = torch.distributions.MultivariateNormal(torch.zeros((self.output_size)).cuda(),torch.eye((self.output_size)).cuda())
		# Compute KL.
		kl_divergence = torch.distributions.kl_divergence(dist, standard_distribution)

		return latent_z, stop_probability, stop, log_prob, kl_divergence, hidden

	def sample_action(self, action_probabilities):
		# Categorical distribution sampling. 
		sample_action = torch.distributions.Categorical(probs=action_probabilities).sample().squeeze(0)
		return sample_action