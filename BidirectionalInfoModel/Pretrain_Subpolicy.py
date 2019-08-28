#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, EncoderNetwork

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
		self.number_policies = number_policies
		self.dataset = dataset

		# Global input size: trajectory at every step - x,y,action
		# Inputs is now states and actions.
		self.input_size = 2*2
		self.hidden_size = 20
		# Number of actions
		self.output_size = 2
		self.state_size = 20

		self.number_epochs = 50
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
		self.number_layers = 4
		self.decay_counter = self.decay_epochs*len(self.dataset)

		# Log-likelihood penalty.
		self.lambda_likelihood_penalty = self.args.likelihood_penalty
		self.baseline = None

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_counter)

	def initialize_gt_subpolicies(self):

		self.action_map = np.array([[0,-1],[-1,0],[0,1],[1,0]], dtype=np.float)
		# self.dists = [stats.norm(loc=self.action_map[i],scale=[0.05,0.05]) for i in range(4)]
		self.dists = [torch.distributions.MultivariateNormal(loc=torch.tensor(self.action_map[i]).cuda().float(),covariance_matrix=0.05*torch.eye((2)).cuda().float()) for i in range(4)]

	def create_networks(self):
		# Create K Policy Networks. 
		# This policy network automatically manages input size. 
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).cuda()

		# Create encoder.
		# The latent space is just one of 4 z's. So make output of encoder a one hot vector.
		self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).cuda()

	def create_util_ops(self):
		self.filter = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0, groups=1, bias=False)	
		kernel = np.array([-1.0, 1.0])
		kernel = torch.from_numpy(kernel).view(1,1,2).cuda().float()
		self.filter.weight.data = kernel
		self.filter.weight.requires_grad = False

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		self.subpolicy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
		self.encoder_optimizer = torch.optim.Adam(self.encoder_network.parameters(), lr=self.learning_rate)

	def setup(self):
		self.create_networks()
		self.create_training_ops()
		self.create_util_ops()
		self.initialize_gt_subpolicies()

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

	def visualize_trajectory(self, traj):

		fig = plt.figure()		
		ax = fig.gca()
		ax.scatter(traj[:,0],traj[:,1],c=range(len(traj)),cmap='jet')
		plt.xlim(-10,10)
		plt.ylim(-10,10)

		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	# def update_plots(self, counter, sample_map, loglikelihood):
	def update_plots(self, counter, loglikelihood, sample_traj):		
	
		self.writer.add_scalar('Baseline', self.baseline.sum(), counter)
		self.writer.add_scalar('Total Likelihood', loglikelihood.sum(), counter)

		if self.args.entropy:
			self.writer.add_scalar('SubPolicy Entropy', torch.mean(subpolicy_entropy), counter)

		if counter%self.args.display_freq==0:
			self.writer.add_image("GT Trajectory",self.visualize_trajectory(sample_traj), counter)
	
	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b, sample_action_seq):
		# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
		# print("Embedding at Assemble.")	
		assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).cuda()
		assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
		assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
		assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

		# Now assemble inputs for subpolicy.
		subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).cuda()
		subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
		subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
		subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

		# # Concatenated action sqeuence for policy network. 
		# padded_action_seq = np.concatenate([np.zeros((1,self.output_size)),sample_action_seq],axis=0)

		return assembled_inputs, subpolicy_inputs, sample_action_seq

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
		# Sample trajectory segment from dataset. 
		sample_traj, sample_action_seq = self.dataset[i]

		start = 0

		if self.args.traj_length>0:
			sample_action_seq = sample_action_seq[start:self.args.traj_length-1]
		else:
			sample_action_seq = sample_action_seq[start:self.args.traj_length]
		# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
		sample_traj = sample_traj[start:self.args.traj_length]	

		# Now manage concatenated trajectory differently - {{s0,_},{s1,a0},{s2,a1},...,{sn,an-1}}.
		concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)

		# Subsample trajectory segment. 
		min_length = 5
		start_timepoint = np.random.randint(0,self.args.traj_length-min_length)
		end_timepoint = start_timepoint + min_length

		trajectory_segment = concatenated_traj[start_timepoint:end_timepoint]
		sample_action_seq = sample_action_seq[start_timepoint:end_timepoint]
		sample_traj = sample_traj[start_timepoint:end_timepoint]

		return trajectory_segment, sample_action_seq, sample_traj

	def construct_dummy_latents(self, latent_z):

		latent_z_indices = latent_z*torch.ones((5)).cuda().float()

		# Setting latent_b's to 00001. 
		latent_b = torch.ones((5)).cuda().float()
		# latent_b = torch.zeros((5)).cuda().float()
		# latent_b[-1] = 1.

		return latent_z_indices, latent_b

	def update_policies(self, loglikelihood, latent_z, encoder_logprobabilities):

		# Update subpolicy. 
		self.subpolicy_optimizer.zero_grad()
		self.subpolicy_loss = -loglikelihood
		self.subpolicy_loss.sum().backward()
		self.subpolicy_optimizer.step()

		# Update encoder via Reinforce. 
		self.encoder_optimizer.zero_grad()
		self.encoder_loss = self.negative_log_likelihood_loss_function(encoder_logprobabilities.reshape((1,self.number_policies)), latent_z.reshape((1,)))

		# Get baseline. 
		if self.baseline is None:		
			self.baseline = torch.zeros_like(loglikelihood.sum()).cuda().float()
		else:
			self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*loglikelihood

		self.encoder_loss.sum().backward(torch.ones_like(self.encoder_loss).cuda()*(loglikelihood.sum()-self.baseline))
		self.encoder_optimizer.step()

	def run_iteration(self, counter, i):

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
		trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)

		############# (1) #############
		# Encode trajectory segment into latent z. 
		latent_z, encoder_loglikelihood = self.encoder_network.forward(trajectory_segment)

		########## (2) & (3) ##########
		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
		latent_z_indices, latent_b = self.construct_dummy_latents(latent_z)
		_, subpolicy_inputs, sample_action_seq = self.assemble_inputs(trajectory_segment, latent_z_indices, latent_b, sample_action_seq)
		loglikelihood, _ = self.policy_network.forward(subpolicy_inputs, sample_action_seq)

		############# (3) #############
		# Update parameters. 
		if self.args.train:
			self.update_policies(loglikelihood, latent_z, encoder_loglikelihood)
			# Update Plots. 
			# self.update_plots(counter, sample_map, loglikelihood)
			self.update_plots(counter, loglikelihood, trajectory_segment)
		else:
			embed()
		
		print("#########################################")	

	def train(self, model=None):

		if model:
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

		for i in range(len(self.dataset)):
			self.run_iteration(0, i)