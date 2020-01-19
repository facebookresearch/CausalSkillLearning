class PolicyManager_MemoryDownstreamRL(PolicyManager_BaseClass):

	def __init__(self, args):

		# Create environment, setup things, etc. 
		self.args = args		

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_episodes = self.args.epsilon_over
		self.baseline = None
		self.learning_rate = 1e-4
		self.max_timesteps = 250
		self.gamma = 0.99
		self.batch_size = 10

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_episodes)
		self.number_episodes = 5000000

		self.policy_loss_statistics = 0.
		self.critic_loss_statistics = 0.
		self.batch_reward_statistics = 0.

	def create_networks(self):

		# Create policy and critic. 		
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers).cuda()			
		self.critic_network = CriticNetwork(self.input_size, self.args.hidden_size, 1, self.args, self.args.number_layers).cuda()

	def create_training_ops(self):

		self.NLL_Loss = torch.nn.NLLLoss(reduction='none')
		self.MSE_Loss = torch.nn.MSELoss(reduction='none')
		
		# parameter_list = list(self.policy_network.parameters()) + list(self.critic_network.parameters())
		self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
		self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

	def save_all_models(self, suffix):
		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")	
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Critic_Network'] = self.critic_network.state_dict()
		
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		self.critic_network.load_state_dict(load_object['Critic_Network'])

	def setup(self):
		# Create Mujoco environment. 
		self.environment = robosuite.make(self.args.environment, has_renderer=False)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		self.input_size = self.state_size + self.output_size		
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()
		
		self.initialize_plots()

	def set_parameters(self, episode_counter):
		if self.args.train:
			if episode_counter<self.decay_episodes:
				self.epsilon = self.initial_epsilon-self.decay_rate*episode_counter
			else:
				self.epsilon = self.final_epsilon		
		else:
			self.epsilon = 0.

	def reset_lists(self):
		self.reward_trajectory = []
		self.state_trajectory = []
		self.action_trajectory = []
		self.image_trajectory = []
		self.cummulative_rewards = None
		self.episode = None

	def rollout(self, random=False, test=False, visualize=False):
	
		counter = 0		
		eps_reward = 0.	
		state = self.environment.reset()
		terminal = False

		self.reset_lists()

		if visualize:			
			image = self.environment.sim.render(600,600, camera_name='frontview')
			self.image_trajectory.append(np.flipud(image))

		self.state_trajectory.append(state)
		self.reward_trajectory.append(0.)		

		while not(terminal) and counter<self.max_timesteps:

			if random:
				action = self.environment.action_space.sample()
			else:
				# Assemble states. 
				assembled_inputs = self.assemble_inputs()

				if test:
					predicted_action = self.policy_network.reparameterized_get_actions(torch.tensor(assembled_inputs).cuda().float(), greedy=True)
				else:
					predicted_action = self.policy_network.reparameterized_get_actions(torch.tensor(assembled_inputs).cuda().float(), action_epsilon=0.2*self.epsilon)

				action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		

			# Take a step in the environment. 
			next_state, onestep_reward, terminal, success = self.environment.step(action)

			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			self.reward_trajectory.append(onestep_reward)

			# Copy next state into state. 
			state = copy.deepcopy(next_state)

			# Counter
			counter += 1 

			# Append image. 
			if visualize:
				image = self.environment.sim.render(600,600, camera_name='frontview')
				self.image_trajectory.append(np.flipud(image))

		print("Rolled out an episode for ",counter," timesteps.")
		# Now that the episode is done, compute cummulative rewards... 
		self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])

		# NOW construct an episode out of this..	
		self.episode = RLUtils.Episode(self.state_trajectory, self.action_trajectory, self.reward_trajectory)
		# Since we're doing TD updates, we DON'T want to use the cummulative reward, but rather the reward trajectory itself.

	def assemble_inputs(self):

		# Assemble states.
		state_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1) for t in range(len(self.state_trajectory))],axis=0)
		if len(self.action_trajectory)>0:
			action_sequence = np.concatenate([self.action_trajectory[t].reshape((1,-1)) for t in range(len(self.action_trajectory))],axis=0)
			# Appending 0 action to start of sequence.
			action_sequence = np.concatenate([np.zeros((1,self.output_size)),action_sequence],axis=0)
		else:
			action_sequence = np.zeros((1,self.output_size))

		inputs = np.concatenate([state_sequence, action_sequence],axis=1)
		
		return inputs

	def process_episode(self, episode):
		# Assemble states, actions, targets.

		# First reset all the lists from the rollout now that they've been written to memory. 
		self.reset_lists()

		# Now set the lists. 
		self.state_trajectory = episode.state_list
		self.action_trajectory = episode.action_list
		self.reward_trajectory = episode.action_list		

		assembled_inputs = self.assemble_inputs()

		# Input to the policy should be states and actions. 
		self.policy_inputs = torch.tensor(assembled_inputs).cuda().float()	

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

	def set_differentiable_critic_inputs(self):

		# Get policy's predicted actions. 
		self.predicted_actions = self.policy_network.reparameterized_get_actions(self.policy_inputs, action_epsilon=0.2*self.epsilon).squeeze(1)
		# Concatenate the states from policy inputs and the predicted actions. 
		self.critic_inputs = torch.cat([self.policy_inputs[:,:self.state_size], self.predicted_actions],axis=1)

	def set_TD_targets(self):
		# Construct TD Targets. 
		self.TD_targets = self.critic_predictions.clone().detach().cpu().numpy()
		self.TD_targets = np.roll(self.TD_targets,-1,axis=0)
		# Set last element in this to 0.
		self.TD_targets[-1] = 0.
		self.TD_targets *= self.gamma
		self.TD_targets += np.array(self.reward_trajectory)
		self.TD_targets = torch.tensor(self.TD_targets).cuda().float()

	def update_policies_TD(self, counter):
		######################################
		# Compute losses for actor.
		self.policy_optimizer.zero_grad()
		self.set_differentiable_critic_inputs()
		self.policy_loss = - self.critic_network.forward(self.critic_inputs).mean()
		self.policy_loss_statistics += self.policy_loss.clone().detach().cpu().numpy().mean()
		self.policy_loss.backward()
		# self.policy_optimizer.step()

		# Zero gradients, then backprop into critic.
		self.critic_optimizer.zero_grad()		
		self.critic_predictions = self.critic_network.forward(self.policy_inputs).squeeze(1).squeeze(1)

		# Before we actually compute loss, compute targets.
		self.set_TD_targets()
		self.critic_loss = self.MSE_Loss(self.critic_predictions, self.TD_targets).mean()
		self.critic_loss_statistics += self.critic_loss.clone().detach().cpu().numpy().mean()	
		self.critic_loss.backward()
		# self.critic_optimizer.step()
		######################################

	def step_networks(self):
		self.policy_optimizer.step()
		self.critic_optimizer.step()
		self.policy_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		# Can also reset the policy and critic loss statistcs here. 
		self.policy_loss_statistics = 0.
		self.critic_loss_statistics = 0.
		self.batch_reward_statistics = 0.

	def update_batch(self, counter):

		# Get set of indices of episodes in the memory. 
		batch_indices = self.memory.sample_batch(self.batch_size)

		for ind in batch_indices:

			# Retrieve appropriate episode from memory. 
			episode = self.memory[ind]

			# Set quantities from episode.
			self.process_episode(episode)

			# Now compute gradients to both networks from batch.
			self.update_policies_TD(counter)

		# Now actually make a step. 
		self.step_networks()

	def update_plots(self, counter):
		self.tf_logger.scalar_summary('Total Episode Reward', self.cummulative_rewards[0], counter)
		self.tf_logger.scalar_summary('Batch Rewards', self.batch_reward_statistics/self.batch_size, counter)
		self.tf_logger.scalar_summary('Policy Loss', self.policy_loss_statistics/self.batch_size, counter)
		self.tf_logger.scalar_summary('Critic Loss', self.critic_loss_statistics/self.batch_size, counter)

		if counter%self.args.display_freq==0:

			# print("Embedding in Update Plots.")
			
			# Rollout policy.
			self.rollout(random=False, test=True, visualize=True)
			self.tf_logger.gif_summary("Rollout Trajectory", [np.array(self.image_trajectory)], counter)

	def run_iteration(self, counter):

		# This is really a run episode function. Ignore the index, just use the counter. 
		# 1) 	Rollout trajectory. 
		# 2) 	Collect stats / append to memory and stuff.
		# 3) 	Update policies. 
		self.set_parameters(counter)

		# Maintain counter to keep track of updating the policy regularly. 			
		self.rollout(random=False)

		self.memory.append_to_memory(self.episode)

		if self.args.train:

			# Update on batch. 
			self.update_batch(counter)

			# Now upate the policy and critic.
			self.update_policies_TD(counter)
			
			# Update plots. 
			self.update_plots(counter)

	def initialize_memory(self):

		# Create memory object. 
		self.memory = Memory.ReplayMemory(memory_size=10000)

		# Number of initial episodes needs to be less than memory size. 
		self.initial_episodes = 200

		# While number of transitions is less than initial_transitions.
		episode_counter = 0 
		while episode_counter<self.initial_episodes:

			# Rollout an episode.
			self.rollout()

			# Add episode to memory.
			self.memory.append_to_memory(self.episode)

			episode_counter += 1			

	def train(self, model=None):

		# 1) Initialize memory maybe.
		# 2) For number of iterations, RUN ITERATION:
		# 3) 	Rollout trajectory. 
		# 4) 	Collect stats. 
		# 5) 	Update policies. 

		if model:
			print("Loading model in training.")
			self.load_all_models(model)

		print("Starting Main Training Procedure.")
		self.set_parameters(0)

		np.set_printoptions(suppress=True,precision=2)

		# Fixing seeds.
		np.random.seed(seed=0)
		torch.manual_seed(0)		

		print("Initializing Memory.")
		self.initialize_memory()

		for e in range(self.number_episodes):

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			self.run_iteration(e)

			print("Episode: ",e)

			# if e%self.args.eval_freq==0:
			# 	self.automatic_evaluation(e)
