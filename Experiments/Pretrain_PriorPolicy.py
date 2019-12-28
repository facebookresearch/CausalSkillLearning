#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, EncoderNetwork, ContinuousEncoderNetwork
from Transformer import TransformerEncoder
from Visualizers import BaxterVisualizer

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

		# Set z dimensions to 0. 
		self.args.z_dimensions = 0
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

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		parameter_list = list(self.policy_network.parameters())
		self.optimizer = torch.optim.Adam(parameter_list,lr=self.learning_rate)

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
		save_object['PriorPolicy_Network'] = self.policy_network.state_dict()
		# save_object['Policy_Network'] = self.policy_network.state_dict()
		# save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['PriorPolicy_Network'])
		# self.policy_network.load_state_dict(load_object['Policy_Network'])
		# self.encoder_network.load_state_dict(load_object['Encoder_Network'])

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
		
		self.writer.add_scalar('PriorPolicy Likelihood', loglikelihood.mean(), counter)
		self.writer.add_scalar('Total Loss', self.total_loss.mean(), counter)
		# self.writer.add_scalar('Encoder KL', self.encoder_KL.mean(), counter)

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

	# def construct_dummy_latents(self, latent_z):

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

	def update_policies_reparam(self, loglikelihood):
		self.optimizer.zero_grad()
		self.likelihood_loss = -loglikelihood.mean()
		self.total_loss = self.likelihood_loss

		if self.args.debug:
			print("Embedding in Update subpolicies.")
			embed()

		self.total_loss.backward()
		self.optimizer.step()

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

			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
			loglikelihoods, _ = self.policy_network.forward(torch_traj_seg, padded_action_seq)
			loglikelihood = loglikelihoods[:-1].mean()
			 
			# if self.args.debug:
			# 	print("Embedding in Train.")
			# 	embed()

			############# (3) #############
			# Update parameters. 
			if self.args.train:

				self.update_policies_reparam(loglikelihood)

				# Update Plots. 
				self.update_plots(counter, loglikelihood, trajectory_segment)
			else:

				print("Length:",trajectory_segment.shape[0]," Mean Likelihood:", loglikelihood, " Likelihoods: \n", loglikelihoods)
				embed()
				# if return_z: 
				# 	return latent_z, sample_traj, sample_action_seq
				# else:
				# 	np.set_printoptions(suppress=True,precision=2)
				# 	print("###################", i)
				# 	# print("Trajectory: \n",trajectory_segment)
				# 	# print("Encoder Likelihood: \n", encoder_loglikelihood.detach().cpu().numpy())
				# 	# print("Policy Mean: \n", self.policy_network.get_actions(subpolicy_inputs, greedy=True).detach().cpu().numpy())
				# 	print("Policy loglikelihood:", loglikelihood)
			
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

			# self.automatic_evaluation(e)

			index_list = np.arange(0,len(self.dataset))
			np.random.shuffle(index_list)

			# For every item in the epoch:
			for i in range(len(self.dataset)):

				print("Epoch: ",e," Image:",i)
				self.run_iteration(counter, i)
				counter = counter+1

			# if e%self.args.eval_freq==0:
			# 	self.automatic_evaluation(e)

		self.write_and_close()
