#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, LatentPolicyNetwork, VariationalPolicyNetwork

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
		self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.number_policies, self.number_layers).cuda()

		# Create latent policy, whose action space = self.number_policies. 
		# This policy network automatically manages input size. 
		self.latent_policy = LatentPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.number_layers, self.args.b_exploration_bias).cuda()

		# Create variational network. 
		self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, number_layers=self.number_layers, z_exploration_bias=self.args.z_exploration_bias, b_exploration_bias=self.args.b_exploration_bias).cuda()
		# self.variational_policy = VariationalPolicyNetwork.PolicyNetwork(self.input_size, self.hidden_size, self.number_policies).cuda()

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
		# self.latent_policy_optimizer = torch.optim.Adam(self.latent_policy.parameters(), lr=self.learning_rate)
		# self.variational_policy_optimizer = torch.optim.Adam(self.variational_policy.parameters(), lr=self.learning_rate)
		self.latent_policy_optimizer = torch.optim.SGD(self.latent_policy.parameters(), lr=self.learning_rate)
		self.variational_policy_optimizer = torch.optim.SGD(self.variational_policy.parameters(), lr=self.learning_rate)

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
		save_object['Latent_Policy'] = self.latent_policy.state_dict()
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Variational_Policy'] = self.variational_policy.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.latent_policy.load_state_dict(load_object['Latent_Policy'])
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		self.variational_policy.load_state_dict(load_object['Variational_Policy'])

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
	def update_plots(self, counter, subpolicy_loglikelihood, latent_loglikelihood, subpolicy_entropy, sample_traj, latent_z_logprobability, latent_b_logprobability):
		# if len(sample_map.shape)==3:
		# 	self.writer.add_image('Map', sample_map[:,:,0], counter)
		# else:
		# 	self.writer.add_image('Map', sample_map, counter)

		self.writer.add_scalar('Latent Policy Loss', torch.mean(self.total_latent_loss), counter)
		self.writer.add_scalar('SubPolicy Log Likelihood', subpolicy_loglikelihood, counter)
		self.writer.add_scalar('Latent Log Likelihood', latent_loglikelihood, counter)
		self.writer.add_scalar('Variational Policy Loss', torch.mean(self.total_variational_loss), counter)
		self.writer.add_scalar('Baseline', self.baseline, counter)
		self.writer.add_scalar('Total Likelihood', subpolicy_loglikelihood+latent_loglikelihood, counter)
		self.writer.add_scalar('Epsilon', self.epsilon, counter)
		self.writer.add_scalar('Latent Z LogProbability', latent_z_logprobability, counter)
		self.writer.add_scalar('Latent B LogProbability', latent_b_logprobability, counter)

		if self.args.entropy:
			self.writer.add_scalar('SubPolicy Entropy', torch.mean(subpolicy_entropy), counter)
			self.writer.add_scalar('Latent Policy Entropy', -torch.mean(self.latent_b_entropy+self.latent_z_entropy), counter)
		# self.writer.add_scalar('Variational Entropy', -torch.mean(self.total_variational_entropy_loss), counter)
		
		# self.writer.add_scalar('Latent Policy Chosen', sampled_, counter)

		# for i in range(self.number_policies):
		# 	self.writer.add_scalar('Sub Policy {0} Loss '.format(i), self.policy_loss[i], counter)
			# self.writer.add_histogram('Sub Policy {0} Histogram'.format(i), , counter)


		if counter%self.args.display_freq==0:
			self.writer.add_image("GT Trajectory",self.visualize_trajectory(sample_traj), counter)
	
	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def assemble_inputs(self, input_trajectory, latent_z_indices, latent_b):
		# # # Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
		# # # print("Embedding at Assemble.")	
		# assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).cuda()
		# assembled_inputs[range(len(input_trajectory)),:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()
		# assembled_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices] = 1.
		# assembled_inputs[:,-1] = latent_b

		# return assembled_inputs	
		
		# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
		# print("Embedding at Assemble.")	
		assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).cuda()
		assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()

		# THIS IS A WRONG WAY OF ASSIGNING, because 1: means all indices in this range, whilst range(1,len(x)) means index sequentially over all dims / axes.
		# assembled_inputs[1:,self.input_size+latent_z_indices[:-1]] = 1.
		# assembled_inputs[1:,-1] = latent_b[:-1]
		# THIS IS CORRECT: 
		assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1]] = 1.
		assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()
		return assembled_inputs	

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

	def evaluate_loglikelihoods(self, sample_traj, sample_action_seq, concatenated_traj, latent_z_indices, latent_b):
		# Initialize both loglikelihoods to 0. 
		subpolicy_loglikelihood = 0.
		latent_loglikelihood = 0.

		# Need to assemble inputs first - returns a Torch CUDA Tensor.
		# This doesn't need to take in actions, because we can evaluate for all actions then select. 
		assembled_inputs = self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b)

		###########################
		# Compute SubPolicy LogLikelihood. 
		###########################

		# # THIS NOW RETURNS LOG PROBS AND PROBS, FOR ENTROPY CALCULATION. '
		# subpolicy_loglikelihoods, subpolicy_entropy = self.policy_network.forward(assembled_inputs[:-1], sample_action_seq)
		# # subpolicy_loglikelihood = subpolicy_loglikelihoods[range(len(sample_traj)-1), torch.from_numpy(sample_action_seq).cuda().long()].sum()
		# subpolicy_loglikelihood = subpolicy_loglikelihoods.sum()
		# embed()
		subpolicy_loglikelihoods = torch.zeros((len(sample_action_seq)))
		for t in range(len(sample_action_seq)):
			subpolicy_loglikelihoods[t] = self.dists[int(latent_z_indices[t])].log_prob(torch.tensor(sample_action_seq[t]).cuda().float())
		subpolicy_loglikelihood = subpolicy_loglikelihoods.sum()*self.args.subpolicy_ratio
		subpolicy_entropy = torch.zeros(1)

		###########################
		# Compute Latent policy loglikelihood values. 
		###########################		
		latent_z_logprobabilities, latent_b_logprobabilities, latent_b_probabilities, latent_z_probabilities = self.latent_policy.forward(assembled_inputs)
		# embed()
		# LATENT LOGLIKELIHOOD is defined as: 
		# =	\sum_{t=1}^T \log p(\zeta_t | \tau_{1:t}, \zeta_{1:t-1})
		# = \sum_{t=1}^T \log { \phi_t(b_t)} + 1[b_t==1] \eta_t(h_t|s_{1:t}) + 1[b_t==0] 1[z_t==z_{t-1}]

		# Adding log probabilities of termination (of whether it terminated or not), till penultimate step. 
		latent_b_logprobability = latent_b_logprobabilities[range(len(sample_traj)-1),latent_b[:-1].long()].sum()
		latent_loglikelihood += latent_b_logprobability
		
		# # Selects first option for variable = 1, second option for variable = 0. 
		# diff_val = self.filter(latent_z_indices.view(1,1,len(latent_z_indices)).float()).abs()
		# norm_diff_val = torch.where(diff_val.byte(),torch.ones_like(diff_val),torch.zeros_like(diff_val))

		# Instead of this filter crap, use this to check if latent_z elements are equal: 
		diff_val = (1-(latent_z_indices==latent_z_indices.roll(1,0))[1:]).cuda().float()

		latent_z_logprobability = torch.where(latent_b[:-1].byte(), latent_z_logprobabilities[range(len(sample_traj)-1),latent_z_indices[:-1].long()], -self.lambda_likelihood_penalty*diff_val).sum()
		latent_loglikelihood += latent_z_logprobability

		return subpolicy_loglikelihoods, subpolicy_loglikelihood, subpolicy_entropy, latent_loglikelihood, latent_b_logprobabilities, latent_z_logprobabilities, latent_b_probabilities, latent_z_probabilities, latent_z_logprobability, latent_b_logprobability

	def update_policies(self, sample_action_seq, subpolicy_loglikelihoods, subpolicy_entropy, latent_b, latent_z_indices,\
		variational_z_logprobabilities, variational_b_logprobabilities, variational_z_probabilities, variational_b_probabilities,\
		latent_z_logprobabilities, latent_b_logprobabilities, latent_z_probabilities, latent_b_probabilities, loglikelihood):

		###########################
		# Update latent policy. 
		###########################
		# Remember, an NLL loss function takes <Probabilities, Sampled Value> as arguments. 
		self.latent_b_cross_entropy = self.negative_log_likelihood_loss_function(latent_b_logprobabilities, latent_b.long())
		self.latent_b_entropy = torch.bmm(latent_b_probabilities.view(-1,1,2),latent_b_logprobabilities.view(-1,2,1)).squeeze(1).squeeze(1)

		if self.args.entropy:		
			self.latent_b_loss = self.latent_b_cross_entropy+self.entropy_regularization_weight*self.latent_b_entropy
		else:
			self.latent_b_loss = self.latent_b_cross_entropy

		self.latent_z_cross_entropy = self.negative_log_likelihood_loss_function(latent_z_logprobabilities, latent_z_indices.long())
		self.latent_z_entropy = torch.bmm(latent_z_probabilities.view(-1,1,self.number_policies),latent_z_logprobabilities.view(-1,self.number_policies,1)).squeeze(1).squeeze(1)
		
		if self.args.entropy:
			self.latent_z_loss = self.latent_z_cross_entropy+self.entropy_regularization_weight*self.latent_z_entropy
		else:
			self.latent_z_loss = self.latent_z_cross_entropy

		# self.total_latent_loss = torch.where(latent_b.byte(),self.latent_b_loss,self.latent_b_loss+self.latent_z_loss)
		# THE FIRST ARGUMENT SHOULD BE FOR CONDITION==TRUE! SWITCHING THIS! 
		self.total_latent_loss = torch.where(latent_b.byte(),self.latent_b_loss+self.latent_z_loss,self.latent_b_loss)
		self.latent_policy_optimizer.zero_grad()
		self.total_latent_loss.sum().backward()
		self.latent_policy_optimizer.step()

		###########################
		# Update Variational Policy
		###########################

		self.variational_policy_optimizer.zero_grad()
		# self.variational_b_cross_entropy = self.binary_cross_entropy_loss_function(variational_b_preprobabilities, latent_b)
		self.variational_b_cross_entropy = self.negative_log_likelihood_loss_function(variational_b_logprobabilities, latent_b.long())
		# self.variational_b_entropy = torch.bmm(variational_b_probabilities.view(-1,1,2),variational_b_logprobabilities.view(-1,2,1)).squeeze(1).squeeze(1)
		# self.variational_b_loss = self.variational_b_cross_entropy+self.entropy_regularization_weight*self.variational_b_entropy

		self.variational_z_cross_entropy = self.negative_log_likelihood_loss_function(variational_z_logprobabilities, latent_z_indices)
		# self.variational_z_entropy = torch.bmm(variational_z_probabilities.view(-1,1,self.number_policies),variational_z_logprobabilities.view(-1,self.number_policies,1)).squeeze(1).squeeze(1)

		# Always add both for variational net. Because the variational net always chooses both. 
		self.total_variational_loss = self.variational_b_cross_entropy+self.variational_z_cross_entropy
		# self.total_variational_loss = self.variational_z_cross_entropy
 
		if self.args.var_entropy:
			baseline_target = loglikelihood+self.total_variational_loss.sum()-1.
		else:
			baseline_target = loglikelihood

		# COMPUTE BASELINE.
		if self.baseline is None:
			self.baseline = torch.zeros_like(baseline_target).cuda().float()
		else:
			self.baseline = (self.beta_decay*self.baseline)+(1.-self.beta_decay)*baseline_target

		# Gradient expression:		
		# self.total_variational_loss.sum().backward(torch.ones_like(self.total_variational_loss).cuda()*(baseline_target-self.baseline), retain_graph=True)		
		
		# self.total_variational_entropy_loss = self.variational_b_ent_reg_weight*self.variational_b_entropy+self.variational_z_ent_reg_weight*self.variational_z_entropy
		# self.total_variational_entropy_loss.sum().backward()

		self.total_variational_loss.sum().backward(torch.ones_like(self.total_variational_loss).cuda()*(baseline_target-self.baseline))
		self.variational_policy_optimizer.step()

	def rollout_visuals(self, counter, i):

		# Rollout policy with 
		# 	a) Latent variable samples from variational policy operating on dataset trajectories. 
		# 	b) Latent variable samples from latent policy in a rolling fashion, initialized with states from the trajectory. 
		# 	c) Latent variables from the ground truth set (only valid for the toy dataset).

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now. 
		sample_traj, sample_action_seq = self.dataset[i]

		start = 0
		max_length = self.args.traj_length
		sample_action_seq = sample_action_seq[start:max_length]
		sample_traj = sample_traj[start:len(sample_action_seq)]
		concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)


		# (a): Get variationally predicted latent_z's and latent_b's. 
		latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
		variational_b_probabilities, variational_z_probabilities = self.variational_policy.forward(torch.tensor(concatenated_traj).cuda().float(), self.epsilon)

		# For number of timesteps: 
		#	# Feed in states[:t], actions[:t-1], and latent variables[:t] into policy.
		#	# Get action actions[t] from policy.
		# 	# Retrieve state from actions[t] by adding to previous state (this is essentially forward propagating dynamics with random noise)

		# for t in range(len(sample_action_seq)):

	def run_iteration(self, counter, i):
		self.set_epoch(counter)

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now. 
		# sample_traj, sample_action_seq = self.dataset[index_list[i]]
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
		old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)

		############# (1) #############
		# Sample latent variables from p(\zeta | \tau).
		# latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
		# variational_b_probabilities, variational_z_probabilities = self.variational_policy.forward(torch.tensor(concatenated_traj).cuda().float(), self.epsilon)

		latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
		variational_b_probabilities, variational_z_probabilities = self.variational_policy.forward(torch.tensor(old_concatenated_traj).cuda().float(), self.epsilon)

		# # USE "EXPERT" SAMPLING: 
		# if self.args.expert:		
		# 	expert_sample = torch.randint(0,2,(len(latent_z_indices),))
		# 	latent_z_indices = torch.where(expert_sample.cuda().byte(), latent_z_indices, torch.tensor(self.dataset.Y_array[i][start:max_length]).cuda().long()).cuda()

		########## (2) & (3) ##########
		# Evaluate Log Likelihoods of actions and options as "Return" for Variational policy.
		subpolicy_loglikelihoods, subpolicy_loglikelihood, subpolicy_entropy,\
		latent_loglikelihood, latent_b_logprobabilities, latent_z_logprobabilities,\
		 latent_b_probabilities, latent_z_probabilities, latent_z_logprobability, latent_b_logprobability = self.evaluate_loglikelihoods(sample_traj, sample_action_seq, concatenated_traj, latent_z_indices, latent_b)
		
		############# (3) #############
		# Update latent policy Pi_z with Reinforce like update using LL as return. 
		self.update_policies(sample_action_seq, subpolicy_loglikelihoods, subpolicy_entropy, latent_b, latent_z_indices,\
			variational_z_logprobabilities, variational_b_logprobabilities, variational_z_probabilities, variational_b_probabilities,\
			latent_z_logprobabilities, latent_b_logprobabilities, latent_z_probabilities, latent_b_probabilities, subpolicy_loglikelihood+latent_loglikelihood)

		# Update Plots. 
		# self.update_plots(counter, sample_map, loglikelihood)
		self.update_plots(counter, subpolicy_loglikelihood, latent_loglikelihood, subpolicy_entropy, sample_traj, latent_z_logprobability, latent_b_logprobability)
		
		print("Latent LogLikelihood: ", latent_loglikelihood)
		print("Subpolicy LogLikelihood: ",subpolicy_loglikelihood)
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
		
		# Evaluate NLL and (potentially Expected Value Difference) on Validation / Test Datasets. 		
		# self.rollout()
		self.epsilon = 0.
		np.set_printoptions(suppress=True,precision=2)
		for i in range(60):
			sample_traj, sample_action_seq = self.dataset[i]   
			start = 0

			if self.args.traj_length>0:
				sample_action_seq = sample_action_seq[start:self.args.traj_length-1]
			else:
				sample_action_seq = sample_action_seq[start:self.args.traj_length]
			# The trajectory is going to be one step longer than the action sequence, because action sequences are constructed from state differences. Instead, truncate trajectory to length of action sequence. 
			sample_traj = sample_traj[start:self.args.traj_length]	

			concatenated_traj = self.concat_state_action(sample_traj, sample_action_seq)
			old_concatenated_traj = self.old_concat_state_action(sample_traj, sample_action_seq)

			############# (1) #############
			# Sample latent variables from p(\zeta | \tau).
			latent_z_indices, latent_b, variational_b_logprobabilities, variational_z_logprobabilities,\
			variational_b_probabilities, variational_z_probabilities = self.variational_policy.forward(torch.tensor(old_concatenated_traj).cuda().float(), self.epsilon)


			subpolicy_loglikelihoods, subpolicy_loglikelihood, subpolicy_entropy,\
			latent_loglikelihood, latent_b_logprobabilities, latent_z_logprobabilities,\
		 	latent_b_probabilities, latent_z_probabilities, latent_z_logprobability, latent_b_logprobability = self.evaluate_loglikelihoods(sample_traj, sample_action_seq, concatenated_traj, latent_z_indices, latent_b)

			print("#############################################")			
			print("Trajectory",i)
			print("Predicted Z: \n", latent_z_indices.detach().cpu().numpy())
			print("True Z     : \n", np.array(self.dataset.Y_array[i][:self.args.traj_length]))
			print("Latent B   : \n", latent_b.detach().cpu().numpy())
			print("Variational Probs: \n", variational_z_probabilities.detach().cpu().numpy())
			print("Latent Probs     : \n", latent_z_probabilities.detach().cpu().numpy())
			print("Latent B Probs   : \n", latent_b_probabilities.detach().cpu().numpy())

		embed()