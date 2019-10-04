#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, EncoderNetwork, ContinuousEncoderNetwork

class PolicyManager():

	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample trajectory segment from dataset. 
	# 		# Encode trajectory segment into latent z. 
	# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		self.args = args
		self.data = self.args.data
		# Not used if discrete_z is false.
		self.number_policies = number_policies
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.

		# Model size parameters
		if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero':
			self.state_size = 2
			self.input_size = 2*self.state_size
			self.hidden_size = 20
			# Number of actions
			self.output_size = 2		
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = 4
			self.traj_length = 5
			self.number_epochs = 200

		elif self.args.data=='MIME':
			self.state_size = 16			
			self.input_size = 2*self.state_size
			self.hidden_size = 64
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = 5
			self.traj_length = 10
			self.number_epochs = 200

		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self.learning_rate = 1e-4
		
		# Entropy regularization weight.
		self.entropy_regularization_weight = self.args.ent_weight
		# self.variational_entropy_regularization_weight = self.args.var_ent_weight
		self.variational_b_ent_reg_weight = 0.5
		self.variational_z_ent_reg_weight = 0.5

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_epochs = self.args.epsilon_over
		self.decay_counter = self.decay_epochs*len(self.dataset)

		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

	def create_networks(self):
		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		if self.args.discrete_z:
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).cuda()
		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.latent_z_dimensionality, self.number_layers).cuda()
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).cuda()			

		# Create encoder.
		if self.args.discrete_z: 
			# The latent space is just one of 4 z's. So make output of encoder a one hot vector.		
			self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).cuda()
		else:
			# self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality).cuda()
			self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).cuda()

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		self.subpolicy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
		self.encoder_optimizer = torch.optim.Adam(self.encoder_network.parameters(), lr=self.learning_rate)

	def setup(self):
		self.create_networks()
		self.create_training_ops()
		# self.create_util_ops()
		# self.initialize_gt_subpolicies()

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		self.encoder_network.load_state_dict(load_object['Encoder_Network'])

	def initialize_plots(self):
		if self.args.name is not None:
			logdir = os.path.join(self.args.logdir, self.args.name)
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			logdir = os.path.join(logdir, "logs")
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			self.writer = tensorboardX.SummaryWriter(logdir)
		else:
			self.writer = tensorboardX.SummaryWriter()

	def set_epoch(self, counter):
		if counter<self.decay_counter:
			self.epsilon = self.initial_epsilon-self.decay_rate*counter
		else:
			self.epsilon = self.final_epsilon		

	def visualize_trajectory(self, traj, no_axes=False):

		fig = plt.figure()		
		ax = fig.gca()
		ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')
		plt.xlim(-10,10)
		plt.ylim(-10,10)

		if no_axes:
			plt.axis('off')
		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	# def update_plots(self, counter, sample_map, loglikelihood):
	def update_plots(self, counter, loglikelihood, sample_traj):		
	
		self.writer.add_scalar('Baseline', self.baseline.sum(), counter)

		self.writer.add_scalar('Subpolicy Likelihood', loglikelihood.sum(), counter)
		self.writer.add_scalar('Encoder Loss', self.encoder_loss.sum(), counter)
		self.writer.add_scalar('Encoder KL', self.encoder_KL.sum(), counter)

		self.writer.add_scalar('Reinforce Encoder Loss', self.reinforce_encoder_loss.sum(), counter)
		self.writer.add_scalar('Total Encoder Loss', self.total_encoder_loss.sum() ,counter)
		self.writer.add_scalar('Total Loss', self.total_loss.sum(), counter)

		if self.args.regularize_pretraining:
			self.writer.add_scalar('Regularization Loss', torch.mean(self.regularization_loss), counter)

		if self.args.entropy:
			self.writer.add_scalar('SubPolicy Entropy', torch.mean(subpolicy_entropy), counter)

		if counter%self.args.display_freq==0:
			self.writer.add_image("GT Trajectory",self.visualize_trajectory(sample_traj), counter)
	
	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):

		if self.args.discrete_z:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).cuda()
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).cuda()
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sqeuence for policy network. 
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).cuda()
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()			

			assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).cuda()
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sequence for policy network's forward / logprobabilities function. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def get_trajectory_segment(self, i):

		if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero':
			# Sample trajectory segment from dataset. 
			sample_traj, sample_action_seq = self.dataset[i]

			# Subsample trajectory segment. 		
			start_timepoint = np.random.randint(0,self.args.traj_length-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[start_timepoint:end_timepoint]	
			sample_action_seq = sample_action_seq[start_timepoint:end_timepoint-1]

			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

			return concatenated_traj, sample_action_seq, sample_traj
		
		elif self.args.data=='MIME':

			data_element = self.dataset[i]
				
			# Sample a trajectory length that's valid. 			
			trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)

			# Sample random start point.
			start_timepoint = np.random.randint(0,trajectory.shape[0]-self.traj_length)
			end_timepoint = start_timepoint + self.traj_length

			# Get trajectory segment and actions. 
			trajectory = trajectory[start_timepoint:end_timepoint]
			action_sequence = np.diff(trajectory,axis=0)

			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)

			return concatenated_traj, action_sequence, trajectory

	def get_test_trajectory_segment(self, i):
		sample_traj = np.zeros((5,2))

		sample_traj[:,i//2] = np.arange(0,((-1)**i)*5,((-1)**i))
		sample_action_seq = np.diff(sample_traj,axis=0)

		trajectory_segment = self.concat_state_action(sample_traj, sample_action_seq)
		# sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
		
		return trajectory_segment, sample_action_seq, sample_traj

	def construct_dummy_latents(self, latent_z):

		if self.args.discrete_z:

			latent_z_indices = latent_z.float()*torch.ones((self.traj_length)).cuda().float()
		else:
			latent_z_indices = latent_z.squeeze(0)*torch.ones((self.traj_length, self.latent_z_dimensionality)).cuda()

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).cuda().float()
		latent_b = torch.zeros((self.traj_length)).cuda().float()
		# latent_b[-1] = 1.

		return latent_z_indices, latent_b	
		# return latent_z_indices

	def update_policies(self, loglikelihood, latent_z, encoder_logprobabilities, encoder_entropy, encoder_KL, regularization_kl=None, z_distance=None):

		# Update subpolicy. 
		self.subpolicy_optimizer.zero_grad()
		self.subpolicy_loss = -loglikelihood.sum()
		# self.subpolicy_loss.sum().backward()
		# self.subpolicy_optimizer.step()

		# Update encoder via Reinforce. 
		self.encoder_optimizer.zero_grad()

		# Get baseline. 
		if self.baseline is None:		
			self.baseline = torch.zeros_like(loglikelihood.sum()).cuda().float()

		baseline_target = loglikelihood.sum().clone().detach()
		self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*baseline_target

		# Assume uniform (Categorical) prior over the various Z's. 
		prior_probabilities = (1./self.number_policies)*torch.ones((self.number_policies)).cuda()

		if self.args.discrete_z:
			self.encoder_KL = self.KLDivergence_loss_function(encoder_logprobabilities, prior_probabilities).sum()
			self.encoder_loss = self.negative_log_likelihood_loss_function(encoder_logprobabilities.reshape((1,self.number_policies)), latent_z.reshape((1,))) 
		else:
			# Setting "KL" loss / term to negative of entropy. Since we want to maximize entropy of the distribution, minimize the negative of entropy. 
			self.encoder_KL = encoder_KL
			self.encoder_loss = -encoder_logprobabilities.sum()

		self.reinforce_encoder_loss = self.encoder_loss*(baseline_target-self.baseline)
		self.total_encoder_loss = (self.reinforce_encoder_loss + self.args.kl_weight*self.encoder_KL).sum()

		embed()

		if self.args.regularize_pretraining:
			self.regularization_loss = (self.args.reg_loss_wt*(regularization_kl*((1-z_distance**2)/z_distance))).sum()
		else:
			self.regularization_loss = 0.

		self.total_loss = (self.total_encoder_loss + self.subpolicy_loss + self.regularization_loss).sum()

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()
			
		self.total_loss.backward()

		self.subpolicy_optimizer.step()
		self.encoder_optimizer.step()

	def rollout_visuals(self, i, latent_z=None, return_traj=False):

		# Initialize states and latent_z, etc. 
		# For t in range(number timesteps):
		# 	# Retrieve action by feeding input to policy. 
		# 	# Step in environment with action.
		# 	# Update inputs with new state and previously executed action. 

		self.state_dim = 2
		self.rollout_timesteps = 5
		start_state = torch.zeros((self.state_dim))

		if self.args.discrete_z:
			# Assuming 4 discrete subpolicies, just set subpolicy input to 1 at the latent_z index == i. 
			subpolicy_inputs = torch.zeros((1,self.input_size+self.number_policies)).cuda().float()
			subpolicy_inputs[0,self.input_size+i] = 1. 
		else:
			subpolicy_inputs = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).cuda()
			subpolicy_inputs[0,self.input_size:] = latent_z

		subpolicy_inputs[0,:self.state_dim] = start_state
		# subpolicy_inputs[0,-1] = 1.		
		
		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs,greedy=True)
			
			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)
			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row: 
			if self.args.discrete_z:
				input_row = torch.zeros((1,self.input_size+self.number_policies)).cuda().float()
				input_row[0,self.input_size+i] = 1. 
			else:
				input_row = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).cuda().float()
				input_row[0,self.input_size:] = latent_z
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute	
			# input_row[0,-1] = 1.

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)
		print("latent_z:",latent_z)
		trajectory_rollout = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		print("Trajectory:",trajectory_rollout)

		if return_traj:
			return trajectory_rollout		

	def fake_rollout(self, i, j, latent_z, start_state):
		self.state_dim = 2
		self.rollout_timesteps = 5

		# start_state = torch.zeros((self.state_dim))
		start_state = torch.tensor(start_state).cuda().float()

		self.action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]], dtype=np.float)

		# Init subpolicy input.
		subpolicy_inputs = torch.zeros((self.rollout_timesteps,self.input_size+self.latent_z_dimensionality)).cuda()

		# Set latent z
		subpolicy_inputs[:,self.input_size:] = latent_z

		subpolicy_inputs[:,self.state_dim:self.input_size] = torch.tensor(self.action_map[j]).cuda().float()
		subpolicy_inputs[0,:self.state_dim] = start_state
		subpolicy_inputs[range(1,self.rollout_timesteps),:self.state_dim] = start_state+torch.cumsum(subpolicy_inputs[range(self.rollout_timesteps-1),self.state_dim:self.input_size],dim=0)
	
		logprobabilities, _ = self.policy_network.forward(subpolicy_inputs, subpolicy_inputs[:,self.state_dim:self.input_size].detach().cpu().numpy())

		# embed()
		return logprobabilities[:-1].sum().detach().cpu().numpy()

	def run_iteration(self, counter, i, return_z=False):

		# Basic Training Algorithm: 
		# For E epochs:
		# 	# For all trajectories:
		#		# Sample trajectory segment from dataset. 
		# 		# Encode trajectory segment into latent z. 
		# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		# 		# Update parameters. 

		self.set_epoch(counter)

		############# (0) #############
		# Sample trajectory segment from dataset. 			
		if self.args.train or not(self.args.discrete_z):			
			trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)
		else:
			trajectory_segment, sample_action_seq, sample_traj  = self.get_test_trajectory_segment(i)

		############# (1) #############
		# Encode trajectory segment into latent z. 		
		latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(trajectory_segment)

		########## (2) & (3) ##########
		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

		_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

		loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq)
		loglikelihood = loglikelihoods[:-1].sum()
		 
		if self.args.debug:
			print("Embedding in Train.")
			embed()

		############# (3) #############
		# Update parameters. 
		if self.args.train:

			# If we are regularizing: 
			# 	(1) Sample another z. 
			# 	(2) Construct inputs and such.
			# 	(3) Compute distances, and feed to update_policies.
			if self.args.regularize_pretraining:
				alternate_latent_z, _, _, _ = self.encoder_network.forward(trajectory_segment)

				alt_latent_z_seq, _ = self.construct_dummy_latents(alternate_latent_z)
				_, alt_subpolicy_inputs, _ = self.assemble_inputs(trajectory_segment, alt_latent_z_seq, latent_b, sample_action_seq)

				regularization_kl = self.policy_network.get_regularization_kl(subpolicy_inputs, alt_subpolicy_inputs)
				z_distance = torch.norm(latent_z-alternate_latent_z,p=2)
			else:
				regularization_kl = None
				z_distance = None

			self.update_policies(loglikelihood, latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence, regularization_kl, z_distance)
			# Update Plots. 
			self.update_plots(counter, loglikelihood, trajectory_segment)
		else:


			if return_z: 
				return latent_z
			else:
				np.set_printoptions(suppress=True,precision=2)
				print("###################", i)
				# print("Trajectory: \n",trajectory_segment)
				# print("Encoder Likelihood: \n", encoder_loglikelihood.detach().cpu().numpy())
				# print("Policy Mean: \n", self.policy_network.get_actions(subpolicy_inputs, greedy=True).detach().cpu().numpy())
				print("Policy loglikelihood:", loglikelihood)
		
		print("#########################################")	

	def train(self, model=None):

		if model:		
			print("Loading model in training.")
			self.load_all_models(model)

		self.initialize_plots()
		counter = 0

		# Fixing seeds.
		np.random.seed(seed=0)
		torch.manual_seed(0)

		# For number of training epochs. 
		for e in range(self.number_epochs): 
		# for e in range(1):
			
			print("Starting Epoch: ",e)

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			index_list = np.arange(0,len(self.dataset))
			np.random.shuffle(index_list)

			# For every item in the epoch:
			for i in range(len(self.dataset)):

				print("Epoch: ",e," Image:",i)
				self.run_iteration(counter, i)
				counter = counter+1

		self.write_and_close()

	def evaluate(self, model):
		if model:
			self.load_all_models(model)


		np.set_printoptions(suppress=True,precision=2)

		print("Running Eval on Dummy Trajectories!")
		for i in range(4):			
			self.run_iteration(0, i)
		
		print("#########################################")	
		print("#########################################")	
	
		if self.args.discrete_z:
			print("Running Rollouts with each latent variable.")
			for i in range(4):	
				print(i)	
				self.rollout_visuals(i)
		else:
			# self.visualize_embedding_space()
			self.visualize_embedded_likelihoods()

	def visualize_embedding_space(self):

		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z. 
		# 	# Feed latent_z into subpolicy. 
		#	# Rollout subpolicy for t timesteps. 
		#	# Plot rollout.
		# Embed plots. 

		# Set N:
		self.N = 200
		self.rollout_timesteps = 5
		self.state_dim = 2

		latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
		trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))

		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		for i in range(self.N):

			# (1) Encoder trajectory. 
			latent_z = self.run_iteration(0, i, return_z=True)

			# Copy z. 
			latent_z_set[i] = copy.deepcopy(latent_z.detach().cpu().numpy())

			if not (self.args.data=='MIME'):
				# (2) Now rollout policy.
				trajectory_set[i] = self.rollout_visuals(i, latent_z=latent_z, return_traj=True)

			# # (3) Plot trajectory.
			# traj_image = self.visualize_trajectory(rollout_traj)

		# TSNE on latentz's.
		tsne = skl_manifold.TSNE(n_components=2,random_state=0)
		embedded_zs = tsne.fit_transform(latent_z_set)

		ratio = 0.3
		if self.args.data=='MIME':
			plt.scatter(embedded_zs[:,0],embedded_zs[:,1])
		else:
			for i in range(self.N):
				plt.scatter(embedded_zs[i,0]+ratio*trajectory_set[i,:,0],embedded_zs[i,1]+ratio*trajectory_set[i,:,1],c=range(self.rollout_timesteps),cmap='jet')

		# Format with name.
		plt.savefig("Images/Embedding_Joint_{0}.png".format(self.args.name))
		plt.close()

	def visualize_embedded_likelihoods(self):

		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z. 
		# 	# Feed latent_z into subpolicy. 
		# 	# Evaluate likelihoods for all sets of actions.
		# 	# Plot

		# Use a constant TSNE fit to project everything.

		# Set N:
		self.N = 200
		self.rollout_timesteps = 5
		self.state_dim = 2

		latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))
		trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))
		likelihoods = np.zeros((self.N, 4))


		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		for i in range(self.N):

			# (1) Encoder trajectory. 
			latent_z = self.run_iteration(0, i, return_z=True)

			# Copy z. 
			latent_z_set[i] = copy.deepcopy(latent_z.detach().cpu().numpy())

			if not (self.args.data=='MIME'):
				# (2) Now rollout policy.
				trajectory_set[i] = self.rollout_visuals(i, latent_z=latent_z, return_traj=True)

			trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)
			# For each action.
			for j in range(4):
				likelihoods[i,j] = self.fake_rollout(i, j, latent_z=latent_z, start_state=sample_traj[0])
	
		# TSNE on latentz's.
		tsne = skl_manifold.TSNE(n_components=2,random_state=0)
		embedded_zs = tsne.fit_transform(latent_z_set)

		ratio = 0.3
		if self.args.data=='MIME':
			plt.scatter(embedded_zs[:,0],embedded_zs[:,1])
		else:
			for i in range(self.N):
				plt.scatter(embedded_zs[i,0]+ratio*trajectory_set[i,:,0],embedded_zs[i,1]+ratio*trajectory_set[i,:,1],c=range(self.rollout_timesteps),cmap='jet')

		# Format with name.
		plt.savefig("Images/Embedding_Joint_{0}.png".format(self.args.name))
		plt.close()
		

		for j in range(4):
			plt.scatter(embedded_zs[:,0],embedded_zs[:,1],c=likelihoods[:,j],cmap='jet',vmin=-100,vmax=10)
			plt.colorbar()
			# Format with name.
			plt.savefig("Images/Likelihood_Embedding{1}_{0}.png".format(self.args.name,j))
			plt.close()
		# For all 4 actions, make fake rollout, feed into trajectory, evaluate likelihood. 
