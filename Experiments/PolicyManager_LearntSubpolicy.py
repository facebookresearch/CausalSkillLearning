#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, LatentPolicyNetwork, ContinuousLatentPolicyNetwork, VariationalPolicyNetwork
from PolicyNetworks import ContinuousVariationalPolicyNetwork, ContinuousEncoderNetwork, ContinuousVariationalPolicyNetwork_BPrior
from Transformer import TransformerVariationalNet
from Visualizers import BaxterVisualizer, SawyerVisualizer
import TFLogger 

class PolicyManager():

	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample latent variables from conditional. 
	# 			# (Concatenate Latent Variables into Input.)
	# 		# Evaluate log likelihoods of actions and options. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		self.args = args
		self.data = self.args.data
		self.number_policies = number_policies
		self.latent_z_dimensionality = self.args.z_dimensions
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.
		# Model size parameters
		self.state_size = 2
		self.state_dim = 2
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2					
		self.number_layers = self.args.number_layers
		self.traj_length = 5

		if self.args.data=='MIME':
			self.state_size = 16	
			self.state_dim = 16		
			self.output_size = self.state_size			
			self.traj_length = self.args.traj_length

			# Create Baxter visualizer for MIME data
			# self.visualizer = BaxterVisualizer.MujocoVisualizer()
			self.visualizer = BaxterVisualizer()

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("MIME_Means.npy")
				self.norm_denom_value = np.load("MIME_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("MIME_Min.npy")
				self.norm_denom_value = np.load("MIME_Max.npy") - np.load("MIME_Min.npy")

		elif self.args.data=='Roboturk':
			self.state_size = 8	
			self.state_dim = 8		
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length

			self.visualizer = SawyerVisualizer()

		self.training_phase_size = self.args.training_phase_size
		self.number_epochs = 200
		self.baseline_value = 0.
		self.beta_decay = 0.9

		self.learning_rate = 1e-4

		self.latent_b_loss_weight = self.args.lat_b_wt
		self.latent_z_loss_weight = self.args.lat_z_wt

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
		if self.args.discrete_z:
			
			# Create K Policy Networks. 
			# This policy network automatically manages input size. 
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.number_policies, self.number_layers).cuda()	
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).cuda()		

			# Create latent policy, whose action space = self.number_policies. 
			# This policy network automatically manages input size. 
			self.latent_policy = LatentPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.number_layers, self.args.b_exploration_bias).cuda()

			# Create variational network. 
			# self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, number_layers=self.number_layers, z_exploration_bias=self.args.z_exploration_bias, b_exploration_bias=self.args.b_exploration_bias).cuda()
			self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.args, number_layers=self.number_layers).cuda()

		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.latent_z_dimensionality, self.number_layers).cuda()
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).cuda()			

			# self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.number_layers, self.args.b_exploration_bias).cuda()
			self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size, self.hidden_size, self.args, self.number_layers).cuda()

			if self.args.b_prior:
				self.variational_policy = ContinuousVariationalPolicyNetwork_BPrior(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).cuda()
			else:
				self.variational_policy = ContinuousVariationalPolicyNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).cuda()

	def create_training_ops(self):
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		
		# If we are using reparameterization, use a global optimizer, and a global loss function. 
		# This means gradients are being handled properly. 
		if self.args.reparam:
			parameter_list = list(self.latent_policy.parameters()) + list(self.variational_policy.parameters())
			if not(self.args.fix_subpolicy):
				parameter_list = parameter_list + list(self.policy_network.parameters())
			self.optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)

		else:
			self.subpolicy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
			self.latent_policy_optimizer = torch.optim.Adam(self.latent_policy.parameters(), lr=self.learning_rate)
			self.variational_policy_optimizer = torch.optim.Adam(self.variational_policy.parameters(), lr=self.learning_rate)
			# self.latent_policy_optimizer = torch.optim.SGD(self.latent_policy.parameters(), lr=self.learning_rate)
			# self.latent_policy_optimizer = torch.optim.SGD(self.latent_policy.parameters(), lr=self.learning_rate)
			# self.variational_policy_optimizer = torch.optim.SGD(self.variational_policy.parameters(), lr=self.learning_rate)

	def setup(self):
		self.create_networks()
		self.create_training_ops()

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Latent_Policy'] = self.latent_policy.state_dict()
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Variational_Policy'] = self.variational_policy.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, just_subpolicy=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if not just_subpolicy:
			self.latent_policy.load_state_dict(load_object['Latent_Policy'])		
			self.variational_policy.load_state_dict(load_object['Variational_Policy'])

	def initialize_plots(self):
		if self.args.name is not None:
			logdir = os.path.join(self.args.logdir, self.args.name)
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			logdir = os.path.join(logdir, "logs")
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			# self.writer = tensorboardX.SummaryWriter(logdir)
			# Create TF Logger. 
			self.tf_logger = TFLogger.Logger(logdir)
		else:
			# self.writer = tensorboardX.SummaryWriter()
			self.tf_logger = TFLogger.Logger()

	def set_epoch(self, counter):
		if self.args.train:
			if counter<self.decay_counter:
				self.epsilon = self.initial_epsilon-self.decay_rate*counter
			else:
				self.epsilon = self.final_epsilon		

			if counter<self.training_phase_size:
				self.training_phase=1
			elif self.training_phase_size<=counter and counter<2*self.training_phase_size:
				self.training_phase=2
			else:
				self.training_phase=3

				# For training phase = 3, set latent_b_loss weight to 1 and latent_z_loss weight to something like 0.1 or 0.01. 
				# After another double training_phase... (i.e. counter>3*self.training_phase_size), 
				# This should be run when counter > 2*self.training_phase_size, and less than 3*self.training_phase_size.
				if counter>3*self.training_phase_size:
					self.latent_z_loss_weight = self.args.lat_z_wt
				if counter>4*self.training_phase_size:
					# Set equal after 4. 
					self.latent_z_loss_weight = self.args.lat_b_wt
		else:
			self.epsilon = 0.
			self.training_phase=1

	def visualize_trajectory(self, trajectory, segmentations=None):

		if self.args.data=='MIME' or self.args.data=='Roboturk': 

			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				unnorm_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			else:
				unnorm_trajectory = trajectory

			return self.visualizer.visualize_joint_trajectory(unnorm_trajectory, return_gif=True, segmentations=segmentations)
		else:
			return self.visualize_2D_trajectory(trajectory)

	def visualize_2D_trajectory(self, traj):

		fig = plt.figure()		
		ax = fig.gca()
		ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')

		scale = 30
		plt.xlim(-scale,scale)
		plt.ylim(-scale,scale)

		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)

		# Already got image data. Now close plot so it doesn't cry.
		# fig.gcf()
		plt.close()

		image = np.transpose(image, axes=[2,0,1])

		return image

	def update_plots(self, counter, i, subpolicy_loglikelihood, latent_loglikelihood, subpolicy_entropy, sample_traj, latent_z_logprobability, latent_b_logprobability, kl_divergence, prior_loglikelihood):

		self.tf_logger.scalar_summary('Latent Policy Loss', torch.mean(self.total_latent_loss), counter)
		self.tf_logger.scalar_summary('SubPolicy Log Likelihood', subpolicy_loglikelihood.mean(), counter)
		self.tf_logger.scalar_summary('Latent Log Likelihood', latent_loglikelihood.mean(), counter)	
		self.tf_logger.scalar_summary('Variational Policy Loss', torch.mean(self.variational_loss), counter)
		self.tf_logger.scalar_summary('Variational Reinforce Loss', torch.mean(self.reinforce_variational_loss), counter)
		self.tf_logger.scalar_summary('Total Variational Policy Loss', torch.mean(self.total_variational_loss), counter)
		self.tf_logger.scalar_summary('Baseline', self.baseline.mean(), counter)
		self.tf_logger.scalar_summary('Total Likelihood', subpolicy_loglikelihood+latent_loglikelihood, counter)
		self.tf_logger.scalar_summary('Epsilon', self.epsilon, counter)
		self.tf_logger.scalar_summary('Latent Z LogProbability', latent_z_logprobability, counter)
		self.tf_logger.scalar_summary('Latent B LogProbability', latent_b_logprobability, counter)
		self.tf_logger.scalar_summary('KL Divergence', torch.mean(kl_divergence), counter)
		self.tf_logger.scalar_summary('Prior LogLikelihood', torch.mean(prior_loglikelihood), counter)

		if counter%self.args.display_freq==0:
			# Now adding visuals for MIME, so it doesn't depend what data we use.
			variational_rollout_image, latent_rollout_image = self.rollout_visuals(counter, i)
			gt_trajectory_image = np.array(self.visualize_trajectory(sample_traj))
			variational_rollout_image = np.array(variational_rollout_image)
			latent_rollout_image = np.array(latent_rollout_image)

			if self.args.data=='MIME' or if self.args.data=='Roboturk':
				# Feeding as list of image because gif_summary.
				self.tf_logger.gif_summary("GT Trajectory",[gt_trajectory_image],counter)
				self.tf_logger.gif_summary("Variational Rollout",[variational_rollout_image],counter)
				self.tf_logger.gif_summary("Latent Rollout",[latent_rollout_image],counter)
			else:
				# Feeding as list of image because gif_summary.
				self.tf_logger.image_summary("GT Trajectory",[gt_trajectory_image],counter)
				self.tf_logger.image_summary("Variational Rollout",[variational_rollout_image],counter)
				self.tf_logger.image_summary("Latent Rollout",[latent_rollout_image],counter)				

	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq, conditional_information=None):

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

			# # This method of concatenation is wrong, because it evaluates likelihood of action [0,0] as well. 
			# # Concatenated action sqeuence for policy network. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			# This is the right method of concatenation, because it evaluates likelihood 			
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:

			if self.training_phase>1:
				latent_z_copy = torch.tensor(latent_z_indices).cuda()
			else:
				latent_z_copy = latent_z_indices

			if conditional_information is None:
				conditional_information = torch.zeros((self.args.condition_size)).cuda().float()			

			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 			
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1+self.args.condition_size)).cuda()
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()			
			assembled_inputs[range(1,len(input_trajectory)),self.input_size:self.input_size+self.latent_z_dimensionality] = latent_z_copy[:-1]
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+self.latent_z_dimensionality+1] = latent_b[:-1].float()	
			# assembled_inputs[range(1,len(input_trajectory)),-self.args.condition_size:] = torch.tensor(conditional_information).cuda().float()

			# Instead of feeding conditional infromation only from 1'st timestep onwards, we are going to st it from the first timestep. 
			assembled_inputs[:,-self.args.condition_size:] = torch.tensor(conditional_information).cuda().float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).cuda()
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size:] = latent_z_indices

			# # This method of concatenation is wrong, because it evaluates likelihood of action [0,0] as well. 
			# # Concatenated action sqeuence for policy network. 
			# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)
			# This is the right method of concatenation, because it evaluates likelihood 			
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

	def concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to start of action sequence and then concatenate. 
		sample_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		# Currently returns: 
		# s0, s1, s2, s3, ..., sn-1, sn
		#  _, a0, a1, a2, ..., an_1, an
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def old_concat_state_action(self, sample_traj, sample_action_seq):
		# Add blank to the END of action sequence and then concatenate.
		sample_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)
		return np.concatenate([sample_traj, sample_action_seq],axis=-1)

	def collect_inputs(self, i, get_latents=False):

		if self.args.data=='Continuous' or self.args.data=='ContinuousDir' or self.args.data=='ContinuousNonZero' or self.args.data=='ContinuousDirNZ' or self.args.data=='GoalDirected' or self.args.data=='DeterGoal' or self.args.data=='Separable':

			sample_traj, sample_action_seq = self.dataset[i]
			latent_b_seq, latent_z_seq = self.dataset.get_latent_variables(i)

			start = 0

			if self.args.traj_length>0:
				sample_action_seq = sample_action_seq[start:self.args.traj_length-1]
				latent_b_seq = latent_b_seq[start:self.args.traj_length-1]
				latent_z_seq = latent_z_seq[start:self.args.traj_length-1]
				sample_traj = sample_traj[start:self.args.traj_length]	
			else:
				# Traj length is going to be -1 here. 
				# Don't need to modify action sequence because it does have to be one step less than traj_length anyway.
				sample_action_seq = sample_action_seq[start:]
				sample_traj = sample_traj[start:]
				latent_b_seq = latent_b_seq[start:]
				latent_z_seq = latent_z_seq[start:]

			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 		
			# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)
			old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)
		
			if self.args.data=='GoalDirected' or self.args.data=='DeterGoal' or self.args.data=='Separable':

				self.conditional_information = np.zeros((self.args.condition_size))
				self.conditional_information[self.dataset.get_goal(i)] = 1
			else:
				self.conditional_information = np.zeros((self.args.condition_size))

			if get_latents:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj, latent_b_seq, latent_z_seq
			else:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj
	
		elif self.args.data=='MIME' or self.args.data=='Roboturk':

			data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None, None
				
			self.conditional_information = np.zeros((self.args.condition_size))

			if self.args.data=='MIME':
				# Sample a trajectory length that's valid. 						
				trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)
			else:
				trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj

	def setup_eval_against_encoder(self):
		# Creates a network, loads the network from pretraining model file. 
		self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).cuda()				
		load_object = torch.load(self.args.subpolicy_model)
		self.encoder_network.load_state_dict(load_object['Encoder_Network'])

		# Force encoder to use original variance for eval.
		self.encoder_network.variance_factor = 1.

	def evaluate_loglikelihoods(self, sample_traj, sample_action_seq, concatenated_traj, latent_z_indices, latent_b):

		# Initialize both loglikelihoods to 0. 
		subpolicy_loglikelihood = 0.
		latent_loglikelihood = 0.

		# Need to assemble inputs first - returns a Torch CUDA Tensor.
		# This doesn't need to take in actions, because we can evaluate for all actions then select. 
		assembled_inputs, subpolicy_inputs, padded_action_seq = self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b, sample_action_seq, self.conditional_information)

		###########################
		# Compute learnt subpolicy loglikelihood.
		###########################
		learnt_subpolicy_loglikelihoods, entropy = self.policy_network.forward(subpolicy_inputs, padded_action_seq)

		# Clip values. # Comment this out to remove clipping.
		learnt_subpolicy_loglikelihoods = torch.clamp(learnt_subpolicy_loglikelihoods,min=self.args.subpolicy_clamp_value)

		# Multiplying the likelihoods with the subpolicy ratio before summing.
		learnt_subpolicy_loglikelihoods = self.args.subpolicy_ratio*learnt_subpolicy_loglikelihoods

		# Summing until penultimate timestep.
		# learnt_subpolicy_loglikelihood = learnt_subpolicy_loglikelihoods[:-1].sum()
		# TAKING AVERAGE HERE AS WELL.		
		learnt_subpolicy_loglikelihood = learnt_subpolicy_loglikelihoods[:-1].mean()

		###########################
		# Compute Latent policy loglikelihood values. 
		###########################		

		# Whether to clone assembled_inputs based on the phase of training. 
		# In phase one it doesn't matter if we use the clone or not, because we never use latent policy loss. 
		# So just clone anyway. 
		# For now, ignore phase 3. This prevents gradients from going into the variational policy from the latent policy.		
		assembled_inputs_copy = assembled_inputs.clone().detach()
		latent_z_copy = latent_z_indices.clone().detach()
		# Consideration for later:
		# if self.training_phase==3:
		# Don't clone.

		if self.args.discrete_z:
			# Return discrete probabilities from latent policy network. 
			latent_z_logprobabilities, latent_b_logprobabilities, latent_b_probabilities, latent_z_probabilities = self.latent_policy.forward(assembled_inputs_copy)
			# # Selects first option for variable = 1, second option for variable = 0. 
			
			# Use this to check if latent_z elements are equal: 
			diff_val = (1-(latent_z_indices==latent_z_indices.roll(1,0))[1:]).cuda().float()
			# We rolled latent_z, we didn't roll diff. This works because latent_b is always guaranteed to be 1 in the first timestep, so it doesn't matter what's in diff_val[0].
			diff_val = diff_val.roll(1,0)

			# Selects first option for variable = 1, second option for variable = 0. 
			latent_z_temporal_logprobabilities = torch.where(latent_b[:-1].byte(), latent_z_logprobabilities[range(len(sample_traj)-1),latent_z_indices[:-1].long()], -self.lambda_likelihood_penalty*diff_val)
			latent_z_logprobability = latent_z_temporal_logprobabilities.mean()

		else:
			# If not, we need to evaluate the latent probabilties of latent_z_indices under latent_policy. 
			latent_b_logprobabilities, latent_b_probabilities, latent_distributions = self.latent_policy.forward(assembled_inputs_copy, self.epsilon)
			# Evalute loglikelihood of latent z vectors under the latent policy's distributions. 
			latent_z_logprobabilities = latent_distributions.log_prob(latent_z_copy.unsqueeze(1))

			# Multiply logprobabilities by the latent policy ratio.
			latent_z_temporal_logprobabilities = latent_z_logprobabilities[:-1]*self.args.latentpolicy_ratio
			latent_z_logprobability = latent_z_temporal_logprobabilities.mean()
			latent_z_probabilities = None			

		# LATENT LOGLIKELIHOOD is defined as: 
		# =	\sum_{t=1}^T \log p(\zeta_t | \tau_{1:t}, \zeta_{1:t-1})
		# = \sum_{t=1}^T \log { \phi_t(b_t)} + \log { 1[b_t==1] \eta_t(h_t|s_{1:t}) + 1[b_t==0] 1[z_t==z_{t-1}] } 

		# Adding log probabilities of termination (of whether it terminated or not), till penultimate step. 

		latent_b_temporal_logprobabilities = latent_b_logprobabilities[range(len(sample_traj)-1),latent_b[:-1].long()]
		latent_b_logprobability = latent_b_temporal_logprobabilities.mean()
		latent_loglikelihood += latent_b_logprobability
		latent_loglikelihood += latent_z_logprobability

		# DON'T CLAMP, JUST MULTIPLY BY SUITABLE RATIO! Probably use the same lat_z_wt and lat_b_wt ratios from the losses. 
		latent_temporal_loglikelihoods = self.args.lat_b_wt*latent_b_temporal_logprobabilities + self.args.lat_z_wt*latent_z_temporal_logprobabilities.squeeze(1)

		##################################################
		#### Manage merging likelihoods for REINFORCE ####
		##################################################

		if self.training_phase==1: 
			temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1)
		elif self.training_phase==2 or self.training_phase==3:
			# temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1) + self.args.temporal_latentpolicy_ratio*latent_temporal_loglikelihoods
			temporal_loglikelihoods = learnt_subpolicy_loglikelihoods[:-1].squeeze(1)

		if self.args.debug:
			if self.iter%self.args.debug==0:
				print("Embedding in the Evaluate Likelihoods Function.")
				embed()

		return None, None, None, latent_loglikelihood, \
		 latent_b_logprobabilities, latent_z_logprobabilities, latent_b_probabilities, latent_z_probabilities, \
		 latent_z_logprobability, latent_b_logprobability, learnt_subpolicy_loglikelihood, learnt_subpolicy_loglikelihoods, temporal_loglikelihoods

	def new_update_policies(self, i, sample_action_seq, subpolicy_loglikelihoods, subpolicy_entropy, latent_b, latent_z_indices,\
		variational_z_logprobabilities, variational_b_logprobabilities, variational_z_probabilities, variational_b_probabilities, kl_divergence, \
		latent_z_logprobabilities, latent_b_logprobabilities, latent_z_probabilities, latent_b_probabilities, \
		learnt_subpolicy_loglikelihood, learnt_subpolicy_loglikelihoods, loglikelihood, prior_loglikelihood, latent_loglikelihood, temporal_loglikelihoods):

		# Set optimizer gradients to zero.
		self.optimizer.zero_grad()

		# Assemble prior and KL divergence losses. 
		# Since these are output by the variational network, and we don't really need the last z predicted by it. 
		prior_loglikelihood = prior_loglikelihood[:-1]		
		kl_divergence = kl_divergence[:-1]

		######################################################
		############## Update latent policy. #################
		######################################################
		
		# Remember, an NLL loss function takes <Probabilities, Sampled Value> as arguments. 
		self.latent_b_loss = self.negative_log_likelihood_loss_function(latent_b_logprobabilities, latent_b.long())

		if self.args.discrete_z:
			self.latent_z_loss = self.negative_log_likelihood_loss_function(latent_z_logprobabilities, latent_z_indices.long())
		# If continuous latent_z, just calculate loss as negative log likelihood of the latent_z's selected by variational network.
		else:
			self.latent_z_loss = -latent_z_logprobabilities.squeeze(1)

		# Compute total latent loss as weighted sum of latent_b_loss and latent_z_loss.
		self.total_latent_loss = (self.latent_b_loss_weight*self.latent_b_loss+self.latent_z_loss_weight*self.latent_z_loss)[:-1]

		#######################################################
		############# Compute Variational Losses ##############
		#######################################################

		# MUST ALWAYS COMPUTE: # Compute cross entropies. 
		self.variational_b_loss = self.negative_log_likelihood_loss_function(variational_b_logprobabilities[:-1], latent_b[:-1].long())

		# In case of reparameterization, the variational loss that goes to REINFORCE should just be variational_b_loss.
		self.variational_loss = self.args.var_loss_weight*self.variational_b_loss

		#######################################################
		########## Compute Variational Reinforce Loss #########
		#######################################################

		# Compute reinforce target based on how we express the objective:
		# The original implementation, i.e. the entropic implementation, uses:
		# (1) \mathbb{E}_{x, z \sim q(z|x)} \Big[ \nabla_{\omega} \log q(z|x,\omega) \{ \log p(x||z) + \log p(z||x) - \log q(z|x) - 1 \} \Big] 

		# The KL divergence implementation uses:
		# (2) \mathbb{E}_{x, z \sim q(z|x)} \Big[ \nabla_{\omega} \log q(z|x,\omega) \{ \log p(x||z) + \log p(z||x) - \log p(z) \} \Big] - \nabla_{\omega} D_{KL} \Big[ q(z|x) || p(z) \Big]

		# Compute baseline target according to NEW GRADIENT, and Equation (2) above. 
		baseline_target = (temporal_loglikelihoods - self.args.prior_weight*prior_loglikelihood).clone().detach()

		if self.baseline is None:
			self.baseline = torch.zeros_like(baseline_target.mean()).cuda().float()
		else:
			self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*baseline_target.mean()
			
		self.reinforce_variational_loss = self.variational_loss*(baseline_target-self.baseline)

		# If reparam, the variational loss is a combination of three things. 
		# Losses from latent policy and subpolicy into variational network for the latent_z's, the reinforce loss on the latent_b's, and the KL divergence. 
		# But since we don't need to additionall compute the gradients from latent and subpolicy into variational network, just set the variational loss to reinforce + KL.
		# self.total_variational_loss = (self.reinforce_variational_loss.sum() + self.args.kl_weight*kl_divergence.squeeze(1).sum()).sum()
		self.total_variational_loss = (self.reinforce_variational_loss + self.args.kl_weight*kl_divergence.squeeze(1)).mean()

		######################################################
		# Set other losses, subpolicy, latent, and prior.
		######################################################

		# Get subpolicy losses.
		self.subpolicy_loss = (-learnt_subpolicy_loglikelihood).mean()

		# Get prior losses. 
		self.prior_loss = (-self.args.prior_weight*prior_loglikelihood).mean()

		# Reweight latent loss.
		self.total_weighted_latent_loss = (self.args.latent_loss_weight*self.total_latent_loss).mean()

		################################################
		# Setting total loss based on phase of training.
		################################################

		# IF PHASE ONE: 
		if self.training_phase==1:
			self.total_loss = self.subpolicy_loss + self.total_variational_loss + self.prior_loss
		# IF DONE WITH PHASE ONE:
		elif self.training_phase==2 or self.training_phase==3:
			self.total_loss = self.subpolicy_loss + self.total_weighted_latent_loss + self.total_variational_loss + self.prior_loss

		################################################
		if self.args.debug:
			if self.iter%self.args.debug==0:
				print("Embedding in Update Policies")
				embed()
		################################################

		self.total_loss.sum().backward()
		self.optimizer.step()

	def take_rollout_step(self, subpolicy_input, t):

		actions = self.policy_network.get_actions(subpolicy_input,greedy=True)
		
		# Select last action to execute. 
		action_to_execute = actions[-1].squeeze(1)

		# Compute next state. 
		new_state = subpolicy_input[t,:self.state_dim]+action_to_execute

		# # Concatenate with current subpolicy input. 
		# new_subpolicy_input = torch.cat([subpolicy_input, input_row],dim=0)

		# return new_subpolicy_input
		return action_to_execute, new_state

	def rollout_visuals(self, counter, i):

		# Rollout policy with 
		# 	a) Latent variable samples from variational policy operating on dataset trajectories - Tests variational network and subpolicies. 
		# 	b) Latent variable samples from latent policy in a rolling fashion, initialized with states from the trajectory - Tests latent and subpolicies. 
		# 	c) Latent variables from the ground truth set (only valid for the toy dataset) - Just tests subpolicies. 

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now.
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		if self.args.traj_length>0:
			self.rollout_timesteps = self.args.traj_length
		else:
			self.rollout_timesteps = len(sample_traj)		

		############# (1) #############
		# Sample latent variables from p(\zeta | \tau).
		latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
		variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood = self.variational_policy.forward(torch.tensor(old_concatenated_traj).cuda().float(), self.epsilon)

		############# (1.5) ###########
		# Get assembled inputs and subpolicy inputs for variational rollout.
		orig_assembled_inputs, orig_subpolicy_inputs, padded_action_seq = self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b, sample_action_seq, self.conditional_information)		

		#####################################
		## (A) VARIATIONAL POLICY ROLLOUT. ##		
		#####################################
	
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()

		# For number of rollout timesteps: 
		for t in range(self.rollout_timesteps-1):
			# Take a rollout step. Feed into policy, get action, step, return new input. 
			action_to_execute, new_state = self.take_rollout_step(subpolicy_inputs[:(t+1)].view((t+1,-1)), t)
			state_action_tuple = torch.cat([new_state, action_to_execute],dim=1)
			# Overwrite the subpolicy inputs with the new state action tuple.
			subpolicy_inputs[t+1,:self.input_size] = state_action_tuple
		
		# Get trajectory from this. 
		variational_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())
		variational_rollout_image = self.visualize_trajectory(variational_trajectory_rollout, segmentations=latent_b.detach().cpu().numpy())

		#####################################
		##### (B) LATENT POLICY ROLLOUT. ####
		#####################################

		assembled_inputs = orig_assembled_inputs.clone().detach()
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()

		# For number of rollout timesteps:
		for t in range(self.rollout_timesteps-1):

			##########################################
			#### CODE FOR NEW Z SELECTION ROLLOUT ####
			##########################################

			# Pick latent_z and latent_b. 

			selected_b, new_selected_z = self.latent_policy.get_actions(assembled_inputs[:(t+1)].view((t+1,-1)),greedy=True)
			# selected_b, new_selected_z = self.latent_policy.get_actions(assembled_inputs[:(t+1)].view((t+1,-1)),greedy=False)

			if t==0:
				selected_b = torch.ones_like(selected_b).cuda().float()

			# print("Embedding in latent policy rollout")
			# embed()

			if selected_b[-1]==1:
				# Copy over ALL z's. This is okay to do because we're greedily selecting, and hte latent policy is hence deterministic.
				selected_z = torch.tensor(new_selected_z).cuda().float()

			# Set z's to 0. 
			assembled_inputs[t+1, self.input_size:self.input_size+self.number_policies] = 0.
			# Set z and b in assembled input for the future latent policy passes. 
			if self.args.discrete_z:
				assembled_inputs[t+1, self.input_size+selected_z[-1]] = 1.
			else:

				assembled_inputs[t+1, self.input_size:self.input_size+self.latent_z_dimensionality] = selected_z[-1]
			
			assembled_inputs[t+1, self.input_size+self.latent_z_dimensionality+1] = selected_b[-1]
			assembled_inputs[t+1, -self.args.condition_size:] = torch.tensor(self.conditional_information).cuda().float()

			# Set z's to 0.
			subpolicy_inputs[t, self.input_size:self.input_size+self.number_policies] = 0.

			# Set z and b in subpolicy input for the future subpolicy passes.			
			if self.args.discrete_z:
				subpolicy_inputs[t, self.input_size+selected_z[-1]] = 1.
			else:
				subpolicy_inputs[t, self.input_size:] = selected_z[-1]

			# Now pass subpolicy net forward and get action and next state. 
			action_to_execute, new_state = self.take_rollout_step(subpolicy_inputs[:(t+1)].view((t+1,-1)), t)
			state_action_tuple = torch.cat([new_state, action_to_execute],dim=1)

			# Now update assembled input. 
			assembled_inputs[t+1, :self.input_size] = state_action_tuple
			subpolicy_inputs[t+1, :self.input_size] = state_action_tuple

		latent_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())

		concatenated_selected_b = np.concatenate([selected_b.detach().cpu().numpy(),np.zeros((1))],axis=-1)

		latent_rollout_image = self.visualize_trajectory(latent_trajectory_rollout, concatenated_selected_b)

		# Clear these variables from memory.
		del subpolicy_inputs, assembled_inputs

		return variational_rollout_image, latent_rollout_image

	def run_iteration(self, counter, i):

		# With learnt discrete subpolicy: 

		# For all epochs:
		#	# For all trajectories:
		# 		# Sample z from variational network.
		# 		# Evalute likelihood of latent policy, and subpolicy.
		# 		# Update policies using likelihoods.		

		self.set_epoch(counter)	
		self.iter = counter

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now. 		
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		if sample_traj is not None:
			############# (1) #############
			# Sample latent variables from p(\zeta | \tau).
			latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
			variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood = self.variational_policy.forward(torch.tensor(old_concatenated_traj).cuda().float(), self.epsilon)
			
			########## (2) & (3) ##########
			# Evaluate Log Likelihoods of actions and options as "Return" for Variational policy.
			subpolicy_loglikelihoods, subpolicy_loglikelihood, subpolicy_entropy,\
			latent_loglikelihood, latent_b_logprobabilities, latent_z_logprobabilities,\
			 latent_b_probabilities, latent_z_probabilities, latent_z_logprobability, latent_b_logprobability, \
			 learnt_subpolicy_loglikelihood, learnt_subpolicy_loglikelihoods, temporal_loglikelihoods = self.evaluate_loglikelihoods(sample_traj, sample_action_seq, concatenated_traj, latent_z_indices, latent_b)

			if self.args.train:
				if self.args.debug:
					if self.iter%self.args.debug==0:
						print("Embedding in Train Function.")
						embed()

				############# (3) #############
				# Update latent policy Pi_z with Reinforce like update using LL as return. 			
				self.new_update_policies(i, sample_action_seq, subpolicy_loglikelihoods, subpolicy_entropy, latent_b, latent_z_indices,\
					variational_z_logprobabilities, variational_b_logprobabilities, variational_z_probabilities, variational_b_probabilities, kl_divergence, \
					latent_z_logprobabilities, latent_b_logprobabilities, latent_z_probabilities, latent_b_probabilities, \
					learnt_subpolicy_loglikelihood, learnt_subpolicy_loglikelihoods, learnt_subpolicy_loglikelihood+latent_loglikelihood, \
					prior_loglikelihood, latent_loglikelihood, temporal_loglikelihoods)

				# Update Plots. 
				# self.update_plots(counter, sample_map, loglikelihood)
				self.update_plots(counter, i, learnt_subpolicy_loglikelihood, latent_loglikelihood, subpolicy_entropy, 
					sample_traj, latent_z_logprobability, latent_b_logprobability, kl_divergence, prior_loglikelihood)
					
				# print("Latent LogLikelihood: ", latent_loglikelihood)
				# print("Subpolicy LogLikelihood: ", learnt_subpolicy_loglikelihood)
				print("#########################################")

			else:
				print("#############################################")			
				print("Trajectory",i)
				print("Predicted Z: \n", latent_z_indices.detach().cpu().numpy())
				print("True Z     : \n", np.array(self.dataset.Y_array[i][:self.args.traj_length]))
				print("Latent B   : \n", latent_b.detach().cpu().numpy())
				# print("Variational Probs: \n", variational_z_probabilities.detach().cpu().numpy())
				# print("Latent Probs     : \n", latent_z_probabilities.detach().cpu().numpy())
				print("Latent B Probs   : \n", latent_b_probabilities.detach().cpu().numpy())

				if self.args.subpolicy_model:

					eval_encoded_logprobs = torch.zeros((latent_z_indices.shape[0]))
					eval_orig_encoder_logprobs = torch.zeros((latent_z_indices.shape[0]))

					torch_concat_traj = torch.tensor(concatenated_traj).cuda().float()

					# For each timestep z in latent_z_indices, evaluate likelihood under pretrained encoder model. 
					for t in range(latent_z_indices.shape[0]):
						eval_encoded_logprobs[t] = self.encoder_network.forward(torch_concat_traj, z_sample_to_evaluate=latent_z_indices[t])					
						_, eval_orig_encoder_logprobs[t], _, _ = self.encoder_network.forward(torch_concat_traj)

					print("Encoder Loglikelihood:", eval_encoded_logprobs.detach().cpu().numpy())
					print("Orig Encoder Loglikelihood:", eval_orig_encoder_logprobs.detach().cpu().numpy())
				embed()			

	def train(self, model=None):

		if model:
			self.load_all_models(model)

			if not(self.args.reparam):
				if self.args.fix_subpolicy:
					for param in self.policy_network.parameters():
						param.requires_grad = False

		self.initialize_plots()
		counter = 0
		np.set_printoptions(suppress=True,precision=2)

		# Fixing seeds.
		np.random.seed(seed=0)
		torch.manual_seed(0)

		# For number of training epochs. 
		for e in range(self.number_epochs): 
			
			print("Starting Epoch: ",e)
			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			index_list = np.arange(0,len(self.dataset))
			np.random.shuffle(index_list)

			# For every item in the epoch:
			for i in range(len(self.dataset)):

				print("Epoch: ",e," Image:",i)
				self.run_iteration(counter, index_list[i])				

				counter = counter+1

		self.write_and_close()

	def evaluate(self, model):

		self.set_epoch(0)

		if model:
			self.load_all_models(model)

		if self.args.subpolicy_model:
			print("Loading encoder.")
			self.setup_eval_against_encoder()

		# Evaluate NLL and (potentially Expected Value Difference) on Validation / Test Datasets. 		
		self.epsilon = 0.

		np.set_printoptions(suppress=True,precision=2)
		for i in range(60):
			self.run_iteration(0, i)

		embed()

