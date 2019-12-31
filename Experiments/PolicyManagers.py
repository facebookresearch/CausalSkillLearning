#!/usr/bin/env python
from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, LatentPolicyNetwork, ContinuousLatentPolicyNetwork, VariationalPolicyNetwork, ContinuousEncoderNetwork, EncoderNetwork
from PolicyNetworks import ContinuousVariationalPolicyNetwork, ContinuousEncoderNetwork, ContinuousVariationalPolicyNetwork_BPrior, CriticNetwork
from Transformer import TransformerVariationalNet, TransformerEncoder
from Visualizers import BaxterVisualizer, SawyerVisualizer
import TFLogger

class PolicyManager_BaseClass():

	def __init__(self):
		super(PolicyManager_BaseClass, self).__init__()

	def setup(self):
		self.create_networks()
		self.create_training_ops()
		# self.create_util_ops()
		# self.initialize_gt_subpolicies()

	def initialize_plots(self):
		if self.args.name is not None:
			logdir = os.path.join(self.args.logdir, self.args.name)
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			logdir = os.path.join(logdir, "logs")
			if not(os.path.isdir(logdir)):
				os.mkdir(logdir)
			# Create TF Logger. 
			self.tf_logger = TFLogger.Logger(logdir)
		else:
			self.tf_logger = TFLogger.Logger()

	def write_and_close(self):
		self.writer.export_scalars_to_json("./all_scalars.json")
		self.writer.close()

	def train(self, model=None):

		if model:
			print("Loading model in training.")
			self.load_all_models(model)

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

			# self.automatic_evaluation(e)

			index_list = np.arange(0,len(self.dataset))
			np.random.shuffle(index_list)

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			# For every item in the epoch:
			for i in range(len(self.dataset)):

				print("Epoch: ",e," Image:",i)
				self.run_iteration(counter, index_list[i])				

				counter = counter+1

			if e%self.args.eval_freq==0:
				self.automatic_evaluation(e)

		self.write_and_close()

	def automatic_evaluation(self, e):

		# Writing new automatic evaluation that parses arguments and creates an identical command loading the appropriate model. 
		# Note: If the initial command loads a model, ignore that. 

		command_args = self.args._get_kwargs()
		base_command = 'python Master.py --train=0 --model={0}'.format("Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))

		# For every argument in the command arguments, add it to the base command with the value used, unless it's train or model. 
		for ar in command_args:
			# Skip model and train, because we need to set these manually.
			if ar[0]=='model' or ar[0]=='train':
				pass
			# Add the rest
			else:				
				base_command = base_command + ' --{0}={1}'.format(ar[0],ar[1])
		
		cluster_command = 'python cluster_run.py --partition=learnfair --name={0}_Eval --cmd=\'{1}\''.format(self.args.name, base_command)				
		subprocess.call([cluster_command],shell=True)

	def visualize_robot_data(self):

		self.N = 100
		self.rollout_timesteps = self.args.traj_length
	
		if self.args.data=='MIME':
			self.visualizer = BaxterVisualizer()
			# self.state_dim = 16
		elif self.args.data=='Roboturk':
			self.visualizer = SawyerVisualizer()
			# self.state_dim = 8

		self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
		# These are lists because they're variable length individually.
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
				
				trajectory_rollout = self.get_robot_visuals(i, latent_z, sample_traj, sample_action_seq)
				
				# self.trajectory_set[i] = copy.deepcopy(sample_traj)
				# self.trajectory_rollout_set[i] = copy.deepcopy(trajectory_rollout)	

				self.trajectory_set.append(copy.deepcopy(sample_traj))
				self.trajectory_rollout_set.append(copy.deepcopy(trajectory_rollout))

		# Get MIME embedding for rollout and GT trajectories, with same Z embedding. 
		embedded_z = self.get_robot_embedding()
		gt_animation_object = self.visualize_robot_embedding(embedded_z, gt=True)
		rollout_animation_object = self.visualize_robot_embedding(embedded_z, gt=False)

		self.write_embedding_HTML(gt_animation_object,prefix="GT")
		self.write_embedding_HTML(rollout_animation_object,prefix="Rollout")

		# Save webpage. 
		self.write_results_HTML()

	def rollout_robot_trajectory(self, trajectory_start, latent_z):

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).cuda().float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).cuda().float()
		subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).cuda().float()	

		for t in range(self.rollout_timesteps-1):

			actions = self.policy_network.get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

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

	def get_robot_visuals(self, i, latent_z, trajectory, sample_action_seq):		

		# 1) Feed Z into policy, rollout trajectory. 
		trajectory_rollout = self.rollout_robot_trajectory(trajectory[0], latent_z)

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

	def get_robot_embedding(self):

		# Mean and variance normalize z.
		mean = self.latent_z_set.mean(axis=0)
		std = self.latent_z_set.std(axis=0)
		normed_z = (self.latent_z_set-mean)/std
		
		tsne = skl_manifold.TSNE(n_components=2,random_state=0)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		return scaled_embedded_zs

	def visualize_robot_embedding(self, scaled_embedded_zs, gt=False):

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

class PolicyManager_Prior(PolicyManager_BaseClass):

	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample trajectory segment from dataset. 
	# 		# Encode trajectory segment into latent z. 
	# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Prior, self).__init__()

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
		self.number_epochs = 500

		if self.args.data=='MIME':
			self.state_size = 16			
			self.input_size = 2*self.state_size
			self.hidden_size = 64
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = 5
			self.traj_length = self.args.traj_length
			self.number_epochs = 500

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
		
		elif self.args.data=='MIME' or self.args.data=='Roboturk':

			data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None
				
			# # Sample a trajectory length that's valid. 			
			# trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)
			trajectory = data_element['demo']

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

class PolicyManager_Pretrain(PolicyManager_BaseClass):
	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample trajectory segment from dataset. 
	# 		# Encode trajectory segment into latent z. 
	# 		# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Pretrain, self).__init__()

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
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = self.args.number_layers
		self.traj_length = 5
		self.number_epochs = 500

		if self.args.data=='MIME':
			self.state_size = 16			
			self.state_dim = 16
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.latent_z_dimensionality = self.args.z_dimensions
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length
			self.number_epochs = 500

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("MIME_Means.npy")
				self.norm_denom_value = np.load("MIME_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("MIME_Min.npy")
				self.norm_denom_value = np.load("MIME_Max.npy") - np.load("MIME_Min.npy")

		elif self.args.data=='Roboturk':
			self.state_size = 8	
			self.state_dim = 8		
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length

		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self.learning_rate = 1e-4
		
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

			if self.args.transformer:
				self.encoder_network = TransformerEncoder(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).cuda()
			else:
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

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False):
		load_object = torch.load(path)		
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if not(only_policy):
			self.encoder_network.load_state_dict(load_object['Encoder_Network'])

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
		
		elif self.args.data=='MIME' or self.args.data=='Roboturk':

			data_element = self.dataset[i]

			# If Invalid.
			if not(data_element['is_valid']):
				return None, None, None
				
			# if self.args.data=='MIME':
			# 	# Sample a trajectory length that's valid. 			
			# 	trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)
			# elif self.args.data=='Roboturk':
			# 	trajectory = data_element['demo']
			trajectory = data_element['demo']

			# If allowing variable skill length, set length for this sample.				
			if self.args.var_skill_length:
				# Choose length of 12-16 with certain probabilities. 
				self.current_traj_len = np.random.choice([12,13,14,15,16],p=[0.1,0.2,0.4,0.2,0.1])
			else:
				self.current_traj_len = self.traj_length

			# Sample random start point.
			if trajectory.shape[0]>self.current_traj_len:

				bias_length = int(self.args.pretrain_bias_sampling*trajectory.shape[0])

				# Probability with which to sample biased segment: 
				sample_biased_segment = np.random.binomial(1,p=self.args.pretrain_bias_sampling_prob)

				# If we want to bias sampling of trajectory segments towards the middle of the trajectory, to increase proportion of trajectory segments
				# that are performing motions apart from reaching and returning. 

				# Sample a biased segment if trajectory length is sufficient, and based on probability of sampling.
				if ((trajectory.shape[0]-2*bias_length)>self.current_traj_len) and sample_biased_segment:		
					start_timepoint = np.random.randint(bias_length, trajectory.shape[0] - self.current_traj_len - bias_length)
				else:
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

			# NOW SCALE THIS ACTION SEQUENCE BY SOME FACTOR: 
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			return concatenated_traj, scaled_action_sequence, trajectory

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

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

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
	
	# def automatic_evaluation(self, e):

	# 	# This should be a good template command. 
	# 	base_command = 'python Master.py --train=0 --setting=pretrain_sub --name={0} --data={5} --kl_weight={1} --var_skill_length={2} --z_dimensions=64 --normalization={3} --model={4}'.format(self.args.name, self.args.kl_weight, self.args.var_skill_length, self.args.normalization, "Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e), self.args.data)
	# 	# base_command = 'python Master.py --train=0 --setting=pretrain_sub --name={0} --data=MIME --kl_weight={1} --var_skill_length={2} --transformer=1 --z_dimensions=64 --normalization={3} --model={4}'.format(self.args.name, self.args.kl_weight, self.args.var_skill_length, self.args.normalization, "Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))
	# 	# cluster_command = 'python cluster_run.py --partition=learnfair --name={0} --cmd="'"{1}"'"'.format(self.args.name, base_command)		
	# 	cluster_command = 'python cluster_run.py --partition=learnfair --name={0} --cmd=\'{1}\''.format(self.args.name, base_command)				
	# 	subprocess.call([cluster_command],shell=True)
				
	def evaluate(self, model):
		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data=="MIME" or self.args.data=='Roboturk':
			print("Running Visualization on Robot Data.")	
			self.visualize_robot_data()			

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

class PolicyManager_Joint(PolicyManager_BaseClass):

	# Basic Training Algorithm: 
	# For E epochs:
	# 	# For all trajectories:
	#		# Sample latent variables from conditional. 
	# 			# (Concatenate Latent Variables into Input.)
	# 		# Evaluate log likelihoods of actions and options. 
	# 		# Update parameters. 

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Joint, self).__init__()

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
			self.input_size = 2*self.state_size
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
			self.input_size = 2*self.state_size	
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length

			self.visualizer = SawyerVisualizer()

		self.training_phase_size = self.args.training_phase_size
		self.number_epochs = 500
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
		parameter_list = list(self.latent_policy.parameters()) + list(self.variational_policy.parameters())
		if not(self.args.fix_subpolicy):
			parameter_list = parameter_list + list(self.policy_network.parameters())
		self.optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)

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

	def compute_evaluation_metrics(self, sample_traj, counter, i):

		# # Generate trajectory rollouts so we can calculate distance metric. 
		# self.rollout_visuals(counter, i, get_image=False)

		# Compute trajectory distance between:
		var_rollout_distance = ((self.variational_trajectory_rollout-sample_traj)**2).mean()
		latent_rollout_distance = ((self.latent_trajectory_rollout-sample_traj)**2).mean()

		return var_rollout_distance, latent_rollout_distance

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

			# Compute distance metrics. 
			var_dist, latent_dist = self.compute_evaluation_metrics(sample_traj, counter, i)
			self.tf_logger.scalar_summary('Variational Trajectory Distance', var_dist, counter)
			self.tf_logger.scalar_summary('Latent Trajectory Distance', latent_dist, counter)

			gt_trajectory_image = np.array(self.visualize_trajectory(sample_traj))
			variational_rollout_image = np.array(variational_rollout_image)
			latent_rollout_image = np.array(latent_rollout_image)

			if self.args.data=='MIME' or self.args.data=='Roboturk':
				# Feeding as list of image because gif_summary.
				self.tf_logger.gif_summary("GT Trajectory",[gt_trajectory_image],counter)
				self.tf_logger.gif_summary("Variational Rollout",[variational_rollout_image],counter)
				self.tf_logger.gif_summary("Latent Rollout",[latent_rollout_image],counter)
			else:
				# Feeding as list of image because gif_summary.
				self.tf_logger.image_summary("GT Trajectory",[gt_trajectory_image],counter)
				self.tf_logger.image_summary("Variational Rollout",[variational_rollout_image],counter)
				self.tf_logger.image_summary("Latent Rollout",[latent_rollout_image],counter)				

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

			# if self.args.data=='MIME':
			# 	# Sample a trajectory length that's valid. 						
			# 	trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)
			# else:
			# 	trajectory = data_element['demo']
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

	def rollout_visuals(self, counter, i, get_image=True):

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
		self.variational_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())
		

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

		self.latent_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())

		concatenated_selected_b = np.concatenate([selected_b.detach().cpu().numpy(),np.zeros((1))],axis=-1)

		# Clear these variables from memory.
		del subpolicy_inputs, assembled_inputs

		if get_image==True:
			latent_rollout_image = self.visualize_trajectory(self.latent_trajectory_rollout, concatenated_selected_b)
			variational_rollout_image = self.visualize_trajectory(self.variational_trajectory_rollout, segmentations=latent_b.detach().cpu().numpy())	

			return variational_rollout_image, latent_rollout_image
		else:
			return None, None

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

				if self.args.data=='MIME' or self.args.data=='Roboturk':
					pass
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
				
				if self.args.debug:
					embed()			

	def evaluate(self, model):

		self.set_epoch(0)

		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		# Visualize space if the subpolicy has been trained...
		if (self.args.data=="MIME" or self.args.data=='Roboturk') and (self.args.fix_subpolicy==0):
			print("Running Visualization on Robot Data.")	
			self.pretrain_policy_manager = PolicyManager_Pretrain(self.args.number_policies, self.dataset, self.args)
			self.pretrain_policy_manager.setup()
			self.pretrain_policy_manager.load_all_models(model, only_policy=True)			
			self.pretrain_policy_manager.visualize_robot_data()			

		if self.args.subpolicy_model:
			print("Loading encoder.")
			self.setup_eval_against_encoder()

		# Evaluate NLL and (potentially Expected Value Difference) on Validation / Test Datasets. 		
		self.epsilon = 0.

		# np.set_printoptions(suppress=True,precision=2)
		# for i in range(60):
		# 	self.run_iteration(0, i)

		if self.args.debug:
			embed()

class PolicyManager_DownstreamRL(PolicyManager_BaseClass):

	def __init__(self, args):

		# Create environment, setup things, etc. 
		self.args = args		

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_episodes = self.args.epsilon_over
		self.baseline = None
		self.learning_rate = 1e-4
		self.max_timesteps = 100

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_episodes)

		self.number_episodes = 5000

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
		self.input_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0] + self.output_size
		
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

	def rollout(self, random=False):
	
		counter = 0		
		eps_reward = 0.	
		state = self.environment.reset()
		terminal = False

		reward_trajectory = []
		self.state_trajectory = []
		self.state_trajectory.append(state)
		self.action_trajectory = []		

		while not(terminal) and counter<self.max_timesteps:

			if random:
				action = self.environment.action_space.sample()
			else:

				# Assemble states. 
				assembled_inputs = self.assemble_inputs()
				action = self.policy_network.reparameterized_get_actions(torch.tensor(assembled_inputs).cuda().float())

			# Take a step in the environment. 
			next_state, onestep_reward, terminal, success = self.environment.step(action)

			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			reward_trajectory.append(onestep_reward)

			# Copy next state into state. 
			state = copy.deepcopy(next_state)

			# Counter
			counter +=1 

		# Now that the episode is done, compute cummulative rewards... 
		self.cummulative_rewards = np.cumsum(np.array(reward_trajectory)[::-1])[::-1]

	def assemble_inputs(self):

		# Assemble states.
		state_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1) for t in range(len(self.state_trajectory))],axis=0)
		if len(self.action_trajectory)>0:
			action_sequence = np.concatenate([self.action_trajectory[t].reshape((1,-1)) for t in range(len(self.action_trajectory))],axis=0)
			# Appending 0 action to start of sequence.
			action_sequence = np.concatenate([np.zeros((1,8)),action_sequence],axis=0)
		else:
			action_sequence = np.zeros((1,8))

		inputs = np.concatenate([state_sequence, action_sequence],axis=1)
		embed()
		return inputs

	def process_episode(self):
		# Assemble states, actions, targets.

		# Targets are basically just cummulative rewards. Make torch tensors out of them.
		self.targets = torch.tensor(self.cummulative_rewards).cuda().float()

		assembled_inputs = self.assemble_inputs()

		# Input to the policy should be states and actions. 
		self.policy_inputs = torch.tensor(assembled_inputs).cuda().float()	

	def update_policies(self, counter):
	
		######################################
		# Compute losses for actor.
		self.policy_optimizer.zero_grad()
		self.policy_loss = - self.critic_network.forward(self.policy_inputs).mean()
		self.policy_loss.backward()
		self.policy_optimizer.step()

		# Zero gradients, then backprop into critic.
		self.critic_optimizer.zero_grad()
		self.critic_loss = self.MSE_Loss(self.critic_predictions, self.targets).mean()
		self.critic_loss.backward()
		self.critic_optimizer.step()
		######################################

	def update_plots(self, counter):
		self.tf_logger.scalar_summary('Average Reward', torch.mean(self.targets), counter)
		self.tf_logger.scalar_summary('Policy Loss', self.policy_loss, counter)
		self.tf_logger.scalar_summary('Critic Loss', self.critic_loss, counter)

	def run_iteration(self, counter):

		# This is really a run episode function. Ignore the index, just use the counter. 
		# 1) 	Rollout trajectory. 
		# 2) 	Collect stats / append to memory and stuff.
		# 3) 	Update policies. 
		self.set_parameters(counter)

		# Maintain coujnter to keep track of updating the policy regularly. 			
		self.rollout(random=False)

		if self.args.train:

			# Instead of using memory, just assemble states and everything then update the policy. 
			self.process_episode()

			# Now upate the policy and critic.
			self.update_policies(counter)

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

		for e in range(self.number_episodes):

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			self.run_iteration(e)

			print("Episode: ",e)

			if e%self.args.eval_freq==0:
				self.automatic_evaluation(e)