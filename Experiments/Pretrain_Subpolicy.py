#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, EncoderNetwork, ContinuousEncoderNetwork
import BaxterVisualizer

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
		# if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
		self.state_size = 2
		self.input_size = 2*self.state_size
		self.hidden_size = 20
		# Number of actions
		self.output_size = 2		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = 4
		self.traj_length = 5
		self.number_epochs = 200

		if self.args.data=='MIME':
			self.state_size = 16			
			self.input_size = 2*self.state_size
			self.hidden_size = 64
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = 5
			self.traj_length = self.args.traj_length
			self.number_epochs = 200

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("MIME_Means.npy")
				self.norm_denom_value = np.load("MIME_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("MIME_Min.npy")
				self.norm_denom_value = np.load("MIME_Max.npy") - np.load("MIME_Min.npy")

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

		if self.args.reparam:
			parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
			self.optimizer = torch.optim.Adam(parameter_list,lr=self.learning_rate)
		else:
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
		if self.args.train:
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
			else:
				self.epsilon = self.final_epsilon		
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
		
		self.writer.add_scalar('Subpolicy Likelihood', loglikelihood.mean(), counter)
		self.writer.add_scalar('Total Loss', self.total_loss.mean(), counter)
		self.writer.add_scalar('Encoder KL', self.encoder_KL.mean(), counter)

		if not(self.args.reparam):
			self.writer.add_scalar('Baseline', self.baseline.sum(), counter)
			self.writer.add_scalar('Encoder Loss', self.encoder_loss.sum(), counter)
			self.writer.add_scalar('Reinforce Encoder Loss', self.reinforce_encoder_loss.sum(), counter)
			self.writer.add_scalar('Total Encoder Loss', self.total_encoder_loss.sum() ,counter)

		# if self.args.regularize_pretraining:
		# 	self.writer.add_scalar('Regularization Loss', torch.mean(self.regularization_loss), counter)

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

		if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='Separable':
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

			if data_element['is_valid']:
				
				# Sample a trajectory length that's valid. 			
				trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)

				# If allowing variable skill length, set length for this sample.				
				if self.args.var_skill_length:
					# Choose length of 12-16 with certain probabilities. 
					self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
				else:
					self.current_traj_len = self.traj_length

				# Sample random start point.
				if trajectory.shape[0]>self.current_traj_len:
					start_timepoint = np.random.randint(0,trajectory.shape[0]-self.current_traj_len)
					end_timepoint = start_timepoint + self.current_traj_len

					# Get trajectory segment and actions. 
					trajectory = trajectory[start_timepoint:end_timepoint]				

					# If normalization is set to some value.
					if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
						trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

				else:					
					return None, None, None

				action_sequence = np.diff(trajectory,axis=0)

				# Concatenate
				concatenated_traj = self.concat_state_action(trajectory, action_sequence)

				return concatenated_traj, action_sequence, trajectory

			else:
				return None, None, None

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
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z.squeeze(0) for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).cuda().float()
		latent_b = torch.zeros((self.current_traj_len)).cuda().float()
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

		# if self.args.regularize_pretraining:
		# 	z_epsilon = 0.1
		# 	self.regularization_loss = (self.args.reg_loss_wt*(regularization_kl*((1-z_distance**2)/(z_distance+z_epsilon)))).sum()
		# else:
		self.regularization_loss = 0.

		self.total_loss = (self.total_encoder_loss + self.subpolicy_loss + self.regularization_loss).sum()

		if self.args.debug:			
			print("Embedding in Update subpolicies.")
			embed()
				
		self.total_loss.backward()

		self.subpolicy_optimizer.step()
		self.encoder_optimizer.step()

	def update_policies_reparam(self, loglikelihood, latent_z, encoder_KL):
		self.optimizer.zero_grad()

		# Losses computed as sums.
		# self.likelihood_loss = -loglikelihood.sum()
		# self.encoder_KL = encoder_KL.sum()

		# Instead of summing losses, we should try taking the mean of the  losses, so we can avoid running into issues of variable timesteps and stuff like that. 
		# We should also consider training with randomly sampled number of timesteps.
		self.likelihood_loss = -loglikelihood.mean()
		self.encoder_KL = encoder_KL.mean()

		self.total_loss = (self.likelihood_loss + self.args.kl_weight*self.encoder_KL)

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()

		self.total_loss.backward()
		self.optimizer.step()

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
	
		# Don't feed in epsilon, since it's a rollout, just use the default 0.001.
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

		if trajectory_segment is not None:
			############# (1) #############
			torch_traj_seg = torch.tensor(trajectory_segment).cuda().float()
			# Encode trajectory segment into latent z. 		
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = self.encoder_network.forward(torch_traj_seg, self.epsilon)

			########## (2) & (3) ##########
			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = self.construct_dummy_latents(latent_z)

			_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
			loglikelihoods, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq)
			loglikelihood = loglikelihoods[:-1].mean()
			 
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
				regularization_kl = None
				z_distance = None

				if self.args.reparam:				
					self.update_policies_reparam(loglikelihood, subpolicy_inputs, kl_divergence)
				else:
					self.update_policies(loglikelihood, latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence, regularization_kl, z_distance)

				# Update Plots. 
				self.update_plots(counter, loglikelihood, trajectory_segment)
			else:


				if return_z: 
					return latent_z, sample_traj, sample_action_seq
				else:
					np.set_printoptions(suppress=True,precision=2)
					print("###################", i)
					# print("Trajectory: \n",trajectory_segment)
					# print("Encoder Likelihood: \n", encoder_loglikelihood.detach().cpu().numpy())
					# print("Policy Mean: \n", self.policy_network.get_actions(subpolicy_inputs, greedy=True).detach().cpu().numpy())
					print("Policy loglikelihood:", loglikelihood)
			
			print("#########################################")	
		else: 
			return None, None, None

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

			if e%self.args.eval_freq==0:
				self.automatic_evaluation(e)

		self.write_and_close()
	
	def automatic_evaluation(self, e):

		# This should be a good template command. 
		base_command = 'python Master.py --train=0 --setting=pretrain_sub --name={0} --data=MIME --kl_weight={1} --var_skill_length={2} --z_dimensions=64 --normalization={3}'.format(self.args.name, self.args.kl_weight, self.args.var_skill_length, self.args.normalization, "Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))
		cluster_command = 'python cluster_run.py --partition=learnfair --name={0} --cmd="'"{1}"'"'.format(self.args.name, base_command)		
		
		subprocess.call([cluster_command],shell=True)
		
	def evaluate(self, model):
		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data=="MIME":
			print("Running Visualization on MIME Data.")	
			self.visualize_MIME_data()			

		else:
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
			latent_z, _, _ = self.run_iteration(0, i, return_z=True)

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
			latent_z, _, _ = self.run_iteration(0, i, return_z=True)

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
			plt.savefig("Images/Likelihood_{0}_Embedding{1}.png".format(self.args.name,j))
			plt.close()
		# For all 4 actions, make fake rollout, feed into trajectory, evaluate likelihood. 

	def visualize_MIME_data(self):

		self.N = 5
		self.rollout_timesteps = self.args.traj_length
		self.state_dim = 16

		self.visualizer = BaxterVisualizer.MujocoVisualizer()

		self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
		# self.trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))
		# self.trajectory_rollout_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))
		self.indices = []

		self.trajectory_set = []
		self.trajectory_rollout_set = []		

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))

		self.rollout_gif_list = []
		self.gt_gif_list = []

		# Create save directory:
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		self.max_len = 0

		for i in range(self.N):

			print("#########################################")	
			print("Getting visuals for trajectory: ",i)
			latent_z, sample_traj, sample_action_seq = self.run_iteration(0, i, return_z=True)

			if latent_z is not None:
				self.indices.append(i)

				if len(sample_traj)>self.max_len:
					self.max_len = len(sample_traj)

				self.latent_z_set[i] = copy.deepcopy(latent_z.detach().cpu().numpy())		
				
				trajectory_rollout = self.get_MIME_visuals(i, latent_z, sample_traj, sample_action_seq)
				
				# self.trajectory_set[i] = copy.deepcopy(sample_traj)
				# self.trajectory_rollout_set[i] = copy.deepcopy(trajectory_rollout)	

				self.trajectory_set.append(copy.deepcopy(sample_traj))
				self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))

		# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
		embedded_z = self.get_MIME_embedding()
		gt_animation_object = self.visualize_MIME_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_MIME_embedding(embedded_z, gt=False)

		self.write_embedding_HTML(gt_animation_object,prefix="GT")
		self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")

		# Save webpage. 
		self.write_results_HTML()


	def rollout_MIME(self, trajectory_start, latent_z):

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).cuda().float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).cuda().float()
		subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).cuda().float()	

		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row. 
			input_row = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).cuda().float()
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute
			input_row[0,2*self.state_dim:] = latent_z

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		return trajectory

	def get_MIME_visuals(self, i, latent_z, trajectory, sample_action_seq):		

		# 1) Feed Z into policy, rollout trajectory. 
		trajectory_rollout = self.rollout_MIME(trajectory[0], latent_z)

		# 2) Unnormalize data. 
		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			unnorm_gt_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
		else:
			unnorm_gt_trajectory = trajectory
			unnorm_pred_trajectory = trajectory_rollout

		# 3) Run unnormalized ground truth trajectory in visualizer. 
		ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GT.gif".format(i), return_and_save=True)
		
		# 4) Run unnormalized rollout trajectory in visualizer. 
		rollout_gif = self.visualizer.visualize_joint_trajectory(unnorm_pred_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_Rollout.gif".format(i), return_and_save=True)
		
		self.gt_gif_list.append(copy.deepcopy(ground_truth_gif))
		self.rollout_gif_list.append(copy.deepcopy(rollout_gif))

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			return unnorm_pred_trajectory
		else:
			return trajectory_rollout

	def write_results_HTML(self):
		# Retrieve, append, and print images from datapoints across different models. 

		print("Writing HTML File.")
		# Open Results HTML file. 	    
		with open(os.path.join(self.dir_name,'Results_{}.html'.format(self.args.name)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))

			for i in range(self.N):
				
				if i%100==0:
					print("Datapoint:",i)                        
				html_file.write('<p> <b> Trajectory {}  </b></p>'.format(i))

				file_prefix = self.dir_name

				# Create gif_list by prefixing base_gif_list with file prefix.
				html_file.write('<div style="display: flex; justify-content: row;">  <img src="Traj_{0}_GT.gif"/>  <img src="Traj_{0}_Rollout.gif"/> </div>'.format(i))
					
				# Add gap space.
				html_file.write('<p> </p>')

			html_file.write('</body>')
			html_file.write('</html>')

	def write_embedding_HTML(self, animation_object, prefix=""):
		print("Writing Embedding File.")
		# Open Results HTML file. 	    
		with open(os.path.join(self.dir_name,'Embedding_{0}_{1}.html'.format(prefix,self.args.name)),'w') as html_file:
			
			# Start HTML doc. 
			html_file.write('<html>')
			html_file.write('<body>')
			html_file.write('<p> Model: {0}</p>'.format(self.args.name))

			html_file.write(animation_object.to_html5_video())
			# print(animation_object.to_html5_video(), file=html_file)

			html_file.write('</body>')
			html_file.write('</html>')

		animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)))		

	def get_MIME_embedding(self):

		# Mean and variance normalize z.
		mean = self.latent_z_set.mean(axis=0)
		std = self.latent_z_set.std(axis=0)
		normed_z = (self.latent_z_set-mean)/std
		
		tsne = skl_manifold.TSNE(n_components=2,random_state=0)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		return scaled_embedded_zs

	def visualize_MIME_embedding(self, scaled_embedded_zs, gt=False):

		# Create figure and axis objects.
		matplotlib.rcParams['figure.figsize'] = [50, 50]
		fig, ax = plt.subplots()

		# number_samples = 400
		number_samples = self.N		

		# Create a scatter plot of the embedding itself. The plot does not seem to work without this. 
		ax.scatter(scaled_embedded_zs[:number_samples,0],scaled_embedded_zs[:number_samples,1])
		ax.axis('off')
		ax.set_title("Embedding of Latent Representation of Pre-trained Subpolicy",fontdict={'fontsize':40})
		artists = []
		
		# For number of samples in TSNE / Embedding, create a Image object for each of them. 
		for i in range(len(self.indices)):
			if i%10==0:
				print(i)
			# Create offset image (so that we can place it where we choose), with specific zoom. 

			if gt:
				imagebox = OffsetImage(self.gt_gif_list[i][0],zoom=0.4)
			else:
				imagebox = OffsetImage(self.rollout_gif_list[i][0],zoom=0.4)			

			# Create an annotation box to put the offset image into. specify offset image, position, and disable bounding frame. 
			ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
			# Add the annotation box artist to the list artists. 
			artists.append(ax.add_artist(ab))
			
		def update(t):
			# for i in range(number_samples):
			for i in range(len(self.indices)):
				
				if gt:
					imagebox = OffsetImage(self.gt_gif_list[i][min(t, len(self.gt_gif_list[i])-1)],zoom=0.4)
				else:
					imagebox = OffsetImage(self.rollout_gif_list[i][min(t, len(self.rollout_gif_list[i])-1)],zoom=0.4)			

				ab = AnnotationBbox(imagebox, (scaled_embedded_zs[self.indices[i],0], scaled_embedded_zs[self.indices[i],1]), frameon=False)
				artists.append(ax.add_artist(ab))
			
		# update_len = 20
		anim = FuncAnimation(fig, update, frames=np.arange(0, self.max_len), interval=200)

		return anim
