from headers import *
from PolicyNetworks import ContinuousPolicyNetwork, LatentPolicyNetwork, ContinuousLatentPolicyNetwork, VariationalPolicyNetwork, ContinuousEncoderNetwork, EncoderNetwork
from PolicyNetworks import ContinuousVariationalPolicyNetwork, ContinuousEncoderNetwork, ContinuousVariationalPolicyNetwork_BPrior, CriticNetwork
from PolicyNetworks import ContinuousMLP, CriticMLP
from Visualizers import BaxterVisualizer, SawyerVisualizer, MocapVisualizer
import TFLogger, DMP, RLUtils

class PolicyManager_BaseClass():

	def __init__(self):
		super(PolicyManager_BaseClass, self).__init__()

	def setup(self):

		# Fixing seeds.
		np.random.seed(seed=0)
		torch.manual_seed(0)
		np.set_printoptions(suppress=True,precision=2)

		self.create_networks()
		self.create_training_ops()
		# self.create_util_ops()
		# self.initialize_gt_subpolicies()

		if self.args.setting=='imitation':
			extent = self.dataset.get_number_task_demos(self.demo_task_index)
		else:
			extent = len(self.dataset)-self.test_set_size

		self.index_list = np.arange(0,extent)	
		self.initialize_plots()

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

	def collect_inputs(self, i, get_latents=False):

		if self.args.data=='DeterGoal':

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
		
			if self.args.data=='DeterGoal':
				self.conditional_information = np.zeros((self.args.condition_size))
				self.conditional_information[self.dataset.get_goal(i)] = 1
				self.conditional_information[4:] = self.dataset.get_goal_position[i]
			else:
				self.conditional_information = np.zeros((self.args.condition_size))

			if get_latents:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj, latent_b_seq, latent_z_seq
			else:
				return sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj
	
		elif self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			# If we're imitating... select demonstrations from the particular task.
			if self.args.setting=='imitation' and self.args.data=='Roboturk':
				data_element = self.dataset.get_task_demo(self.demo_task_index, i)
			else:
				data_element = self.dataset[i]

			if not(data_element['is_valid']):
				return None, None, None, None
				
			self.conditional_information = np.zeros((self.conditional_info_size))

			trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			self.current_traj_len = len(trajectory)

			if self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
				robot_states = data_element['robot-state']
				object_states = data_element['object-state']

				# Don't set this if pretraining / baseline.
				if self.args.setting=='learntsub':
					self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
					self.conditional_information[:,:self.cond_robot_state_size] = robot_states
					# Doing this instead of self.cond_robot_state_size: because the object_states size varies across demonstrations.
					self.conditional_information[:,self.cond_robot_state_size:self.cond_robot_state_size+object_states.shape[-1]] = object_states			

			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj

	def train(self, model=None):

		if model:
			print("Loading model in training.")
			self.load_all_models(model)		
		counter = 0

		# For number of training epochs. 
		for e in range(self.number_epochs): 
			
			print("Starting Epoch: ",e)

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			# self.automatic_evaluation(e)
			np.random.shuffle(self.index_list)

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			# For every item in the epoch:
			if self.args.setting=='imitation':
				extent = self.dataset.get_number_task_demos(self.demo_task_index)
			else:
				extent = len(self.dataset)-self.test_set_size

			for i in range(extent):

				print("Epoch: ",e," Trajectory:",i, "Datapoint: ", self.index_list[i])
				self.run_iteration(counter, self.index_list[i])				

				counter = counter+1

			if e%self.args.eval_freq==0:
				self.automatic_evaluation(e)

		self.write_and_close()

	def automatic_evaluation(self, e):

		# Writing new automatic evaluation that parses arguments and creates an identical command loading the appropriate model. 
		# Note: If the initial command loads a model, ignore that. 

		command_args = self.args._get_kwargs()			
		base_command = 'python Master.py --train=0 --model={0}'.format("Experiment_Logs/{0}/saved_models/Model_epoch{1}".format(self.args.name, e))

		if self.args.data=='Mocap':
			base_command = 'xvfb-run-safe ' + base_command

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
		elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			self.visualizer = SawyerVisualizer()
			# self.state_dim = 8
		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)

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

	def rollout_robot_trajectory(self, trajectory_start, latent_z, rollout_length=None):

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).cuda().float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).cuda().float()
		subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).cuda().float()	

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		for t in range(length):

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
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[0,self.state_dim:2*self.state_dim] = actions[-1].squeeze(1)
			input_row[0,2*self.state_dim:] = latent_z

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		return trajectory

	def get_robot_visuals(self, i, latent_z, trajectory, sample_action_seq):		

		# 1) Feed Z into policy, rollout trajectory. 
		trajectory_rollout = self.rollout_robot_trajectory(trajectory[0], latent_z, rollout_length=trajectory.shape[0])

		# 2) Unnormalize data. 
		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
			unnorm_gt_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			unnorm_pred_trajectory = (trajectory_rollout*self.norm_denom_value) + self.norm_sub_value
		else:
			unnorm_gt_trajectory = trajectory
			unnorm_pred_trajectory = trajectory_rollout

		if self.args.data=='Mocap':
			# Get animation object from dataset. 
			animation_object = self.dataset[i]['animation']

		# 3) Run unnormalized ground truth trajectory in visualizer. 
		ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GT.gif".format(i), return_and_save=True, additional_info=animation_object)
		
		# 4) Run unnormalized rollout trajectory in visualizer. 
		rollout_gif = self.visualizer.visualize_joint_trajectory(unnorm_pred_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_Rollout.gif".format(i), return_and_save=True, additional_info=animation_object)
		
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
			html_file.write('<p> Average Trajectory Distance: {0}</p>'.format(self.mean_distance))

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

		t1 = time.time()
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
		# animation_object.save(os.path.join(self.dir_name,'{0}_Embedding_Video.mp4'.format(self.args.name)), writer='imagemagick')
		t2 = time.time()

		print("Time taken to write this embedding: ",t2-t1)

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

class PolicyManager_Pretrain(PolicyManager_BaseClass):

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
		self.test_set_size = 500

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
				self.norm_sub_value = np.load("Statistics/MIME_Means.npy")
				self.norm_denom_value = np.load("Statistics/MIME_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/MIME_Min.npy")
				self.norm_denom_value = np.load("Statistics/MIME_Max.npy") - self.norm_sub_value

			# Max of robot_state + object_state sizes across all Baxter environments. 			
			self.cond_robot_state_size = 60
			self.cond_object_state_size = 25
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			if self.args.gripper:
				self.state_size = 8
				self.state_dim = 8
			else:
				self.state_size = 7
				self.state_dim = 7		
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.number_layers = self.args.number_layers
			self.traj_length = self.args.traj_length

			if self.args.normalization=='meanvar':
				self.norm_sub_value = np.load("Statistics/Roboturk_Mean.npy")
				self.norm_denom_value = np.load("Statistics/Roboturk_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/Roboturk_Min.npy")
				self.norm_denom_value = np.load("Statistics/Roboturk_Max.npy") - self.norm_sub_value

			# Max of robot_state + object_state sizes across all sawyer environments. 
			# Robot size always 30. Max object state size is... 23. 
			self.cond_robot_state_size = 30
			self.cond_object_state_size = 23
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size

		elif self.args.data=='Mocap':
			self.state_size = 22*3
			self.state_dim = 22*3	
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
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

			# if self.args.transformer:
			# 	self.encoder_network = TransformerEncoder(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).cuda()
			# else:
			self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).cuda()		

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		if self.args.train_only_policy:
			parameter_list = self.policy_network.parameters()
		else:
			parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
		self.optimizer = torch.optim.Adam(parameter_list,lr=self.learning_rate)

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
		if self.args.train_only_policy and self.args.train: 		
			self.encoder_network.load_state_dict(load_object['Encoder_Network'])
		else:
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
		
		self.tf_logger.scalar_summary('Subpolicy Likelihood', loglikelihood.mean(), counter)
		self.tf_logger.scalar_summary('Total Loss', self.total_loss.mean(), counter)
		self.tf_logger.scalar_summary('Encoder KL', self.encoder_KL.mean(), counter)

		if not(self.args.reparam):
			self.tf_logger.scalar_summary('Baseline', self.baseline.sum(), counter)
			self.tf_logger.scalar_summary('Encoder Loss', self.encoder_loss.sum(), counter)
			self.tf_logger.scalar_summary('Reinforce Encoder Loss', self.reinforce_encoder_loss.sum(), counter)
			self.tf_logger.scalar_summary('Total Encoder Loss', self.total_encoder_loss.sum() ,counter)

		if self.args.entropy:
			self.tf_logger.scalar_summary('SubPolicy Entropy', torch.mean(subpolicy_entropy), counter)

		if counter%self.args.display_freq==0:
			self.tf_logger.image_summary("GT Trajectory",self.visualize_trajectory(sample_traj), counter)
	
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
		
		elif self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			data_element = self.dataset[i]

			# If Invalid.
			if not(data_element['is_valid']):
				return None, None, None
				
			# if self.args.data=='MIME':
			# 	# Sample a trajectory length that's valid. 			
			# 	trajectory = np.concatenate([data_element['la_trajectory'],data_element['ra_trajectory'],data_element['left_gripper'].reshape((-1,1)),data_element['right_gripper'].reshape((-1,1))],axis=-1)
			# elif self.args.data=='Roboturk':
			# 	trajectory = data_element['demo']

			if self.args.gripper:
				trajectory = data_element['demo']
			else:
				trajectory = data_element['demo'][:,:-1]

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

				# CONDITIONAL INFORMATION for the encoder... 

				if self.args.data=='MIME' or self.args.data=='Mocap':
					pass
				elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
					# robot_states = data_element['robot-state'][start_timepoint:end_timepoint]
					# object_states = data_element['object-state'][start_timepoint:end_timepoint]
					pass

					# self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
					# self.conditional_information[:,:self.cond_robot_state_size] = robot_states
					# self.conditional_information[:,self.cond_robot_state_size:object_states.shape[-1]] = object_states								
					# conditional_info = np.concatenate([robot_states,object_states],axis=1)	
			else:					
				return None, None, None

			action_sequence = np.diff(trajectory,axis=0)
			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)

			# NOW SCALE THIS ACTION SEQUENCE BY SOME FACTOR: 
			scaled_action_sequence = self.args.action_scale_factor*action_sequence

			return concatenated_traj, scaled_action_sequence, trajectory

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
		if self.args.traj_segments:			
			trajectory_segment, sample_action_seq, sample_traj  = self.get_trajectory_segment(i)
		else:
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)				
			# Calling it trajectory segment, but it's not actually a trajectory segment here.
			trajectory_segment = concatenated_traj

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

				self.update_policies_reparam(loglikelihood, subpolicy_inputs, kl_divergence)

				# Update Plots. 
				self.update_plots(counter, loglikelihood, trajectory_segment)
			else:

				if return_z: 
					return latent_z, sample_traj, sample_action_seq
				else:
					np.set_printoptions(suppress=True,precision=2)
					print("###################", i)
					print("Policy loglikelihood:", loglikelihood)
			
			print("#########################################")	
		else: 
			return None, None, None
		
	def evaluate_metrics(self):

		self.distances = -np.ones((self.test_set_size))

		# Get test set elements as last (self.test_set_size) number of elements of dataset.
		for i in range(self.test_set_size):

			index = i + len(self.dataset)-self.test_set_size
			print("Evaluating ", i, " in test set, or ", index, " in dataset.")
			# Get latent z. 					
			latent_z, sample_traj, sample_action_seq = self.run_iteration(0, index, return_z=True)

			if sample_traj is not None:
				# Feed latent z to the rollout.
				# rollout_trajectory = self.rollout_visuals(index, latent_z=latent_z, return_traj=True)
				rollout_trajectory = self.rollout_robot_trajectory(sample_traj[0], latent_z, rollout_length=len(sample_traj))

				self.distances[i] = ((sample_traj-rollout_trajectory)**2).mean()	

		self.mean_distance = self.distances[self.distances>0].mean()

	def evaluate(self, model):
		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data=="MIME" or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			print("Running Evaluation of State Distances on small test set.")
			self.evaluate_metrics()

			# Only running viz if we're actually pretraining.
			if self.args.traj_segments:
				print("Running Visualization on Robot Data.")	
				self.visualize_robot_data()
			else:
				# Create save directory:
				upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

				if not(os.path.isdir(upper_dir_name)):
					os.mkdir(upper_dir_name)

				model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
				self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
				if not(os.path.isdir(self.dir_name)):
					os.mkdir(self.dir_name)

			np.save(os.path.join(self.dir_name,"Trajectory_Distances_{0}.npy".format(self.args.name)),self.distances)
			np.save(os.path.join(self.dir_name,"Mean_Trajectory_Distance_{0}.npy".format(self.args.name)),self.mean_distance)

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
		self.conditional_info_size = 6

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
				self.norm_sub_value = np.load("Statistics/MIME_Means.npy")
				self.norm_denom_value = np.load("Statistics/MIME_Var.npy")
			elif self.args.normalization=='minmax':
				self.norm_sub_value = np.load("Statistics/MIME_Min.npy")
				self.norm_denom_value = np.load("Statistics/MIME_Max.npy") - np.load("Statistics/MIME_Min.npy")

			# Max of robot_state + object_state sizes across all Baxter environments. 			
			self.cond_robot_state_size = 60
			self.cond_object_state_size = 25
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size
			self.conditional_viz_env = False

		elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			self.state_size = 8	
			self.state_dim = 8
			self.input_size = 2*self.state_size	
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length

			self.visualizer = SawyerVisualizer()

			# Max of robot_state + object_state sizes across all sawyer environments. 
			# Robot size always 30. Max object state size is... 23. 
			self.cond_robot_state_size = 30
			self.cond_object_state_size = 23
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size
			self.conditional_viz_env = True

		self.training_phase_size = self.args.training_phase_size
		self.number_epochs = 500
		self.test_set_size = 500
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

			# Also add conditional_info_size to this. 
			self.latent_policy = LatentPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.number_layers, self.args.b_exploration_bias).cuda()

			# Create variational network. 
			# self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, number_layers=self.number_layers, z_exploration_bias=self.args.z_exploration_bias, b_exploration_bias=self.args.b_exploration_bias).cuda()
			self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.args, number_layers=self.number_layers).cuda()

		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.latent_z_dimensionality, self.number_layers).cuda()
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).cuda()			

			# self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.number_layers, self.args.b_exploration_bias).cuda()
			self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).cuda()

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

		if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':

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

			if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
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
				# Prevents gradients being propagated through this..
				latent_z_copy = torch.tensor(latent_z_indices).cuda()
			else:
				latent_z_copy = latent_z_indices

			if conditional_information is None:
				conditional_information = torch.zeros((self.conditional_info_size)).cuda().float()

			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 			
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1+self.conditional_info_size)).cuda()
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).cuda().float()			
			assembled_inputs[range(1,len(input_trajectory)),self.input_size:self.input_size+self.latent_z_dimensionality] = latent_z_copy[:-1]
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+self.latent_z_dimensionality+1] = latent_b[:-1].float()	
			# assembled_inputs[range(1,len(input_trajectory)),-self.conditional_info_size:] = torch.tensor(conditional_information).cuda().float()

			# Instead of feeding conditional infromation only from 1'st timestep onwards, we are going to st it from the first timestep. 
			assembled_inputs[:,-self.conditional_info_size:] = torch.tensor(conditional_information).cuda().float()

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

	def set_env_conditional_info(self):
		obs = self.environment._get_observation()
		self.conditional_information = np.zeros((self.conditional_info_size))
		cond_state = np.concatenate([obs['robot-state'],obs['object-state']])
		self.conditional_information[:cond_state.shape[-1]] = cond_state

	def take_rollout_step(self, subpolicy_input, t, use_env=False):

		# Feed subpolicy input into the policy. 
		actions = self.policy_network.get_actions(subpolicy_input,greedy=True)
		
		# Select last action to execute. 
		action_to_execute = actions[-1].squeeze(1)

		if use_env==True:
			# Take a step in the environment. 
			step_res = self.environment.step(action_to_execute.squeeze(0).detach().cpu().numpy())
			# Get state. 
			observation = step_res[0]
			# Now update conditional information... 
			# self.conditional_information = np.concatenate([new_state['robot-state'],new_state['object-state']])

			gripper_open = np.array([0.0115, -0.0115])
			gripper_closed = np.array([-0.020833, 0.020833])

			# The state that we want is ... joint state?
			gripper_finger_values = step_res[0]['gripper_qpos']
			gripper_values = (gripper_finger_values - gripper_open)/(gripper_closed - gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = 2*finger_diff-1

			# Concatenate joint and gripper state. 			
			new_state_numpy = np.concatenate([observation['joint_pos'], np.array(gripper_value).reshape((1,))])
			new_state = torch.tensor(new_state_numpy).cuda().float().view((1,-1))

			# This should be true by default...
			# if self.conditional_viz_env:
			# 	self.set_env_conditional_info()
			self.set_env_conditional_info()
			
		else:
			# Compute next state by adding action to state. 
			new_state = subpolicy_input[t,:self.state_dim]+action_to_execute	

		# return new_subpolicy_input
		return action_to_execute, new_state

	def create_RL_environment_for_rollout(self, environment_name, state=None):

		self.environment = robosuite.make(environment_name)

		if state is not None:
			self.environment.sim.set_state_from_flattened(state)

	def rollout_variational_network(self, counter, i):

		###########################################################
		###########################################################

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
		# Doesn't really matter what the conditional information is here... because latent policy isn't being rolled out. 
		# We still call it becasue these assembled inputs are passed to the latnet policy rollout later.

		if self.conditional_viz_env:
			self.set_env_conditional_info()
		# Get assembled inputs and subpolicy inputs for variational rollout.
		orig_assembled_inputs, orig_subpolicy_inputs, padded_action_seq = self.assemble_inputs(concatenated_traj, latent_z_indices, latent_b, sample_action_seq, self.conditional_information)		

		###########################################################
		############# (A) VARIATIONAL POLICY ROLLOUT. #############
		###########################################################
	
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

		return orig_assembled_inputs, orig_subpolicy_inputs, latent_b

	def rollout_latent_policy(self, orig_assembled_inputs, orig_subpolicy_inputs):
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

			# Before copying over, set conditional_info from the environment at the current timestep.

			if self.conditional_viz_env:
				self.set_env_conditional_info()

			assembled_inputs[t+1, -self.conditional_info_size:] = torch.tensor(self.conditional_information).cuda().float()

			# Set z's to 0.
			subpolicy_inputs[t, self.input_size:self.input_size+self.number_policies] = 0.

			# Set z and b in subpolicy input for the future subpolicy passes.			
			if self.args.discrete_z:
				subpolicy_inputs[t, self.input_size+selected_z[-1]] = 1.
			else:
				subpolicy_inputs[t, self.input_size:] = selected_z[-1]

			# Now pass subpolicy net forward and get action and next state. 
			action_to_execute, new_state = self.take_rollout_step(subpolicy_inputs[:(t+1)].view((t+1,-1)), t, use_env=self.conditional_viz_env)
			state_action_tuple = torch.cat([new_state, action_to_execute],dim=1)

			# Now update assembled input. 
			assembled_inputs[t+1, :self.input_size] = state_action_tuple
			subpolicy_inputs[t+1, :self.input_size] = state_action_tuple

		self.latent_trajectory_rollout = copy.deepcopy(subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy())

		concatenated_selected_b = np.concatenate([selected_b.detach().cpu().numpy(),np.zeros((1))],axis=-1)

		# Clear these variables from memory.
		del subpolicy_inputs, assembled_inputs

		return concatenated_selected_b

	def rollout_visuals(self, counter, i, get_image=True):

		# if self.args.data=='Roboturk':
		if self.conditional_viz_env:
			self.create_RL_environment_for_rollout(self.dataset[i]['environment-name'], self.dataset[i]['flat-state'][0])

		# Rollout policy with 
		# 	a) Latent variable samples from variational policy operating on dataset trajectories - Tests variational network and subpolicies. 
		# 	b) Latent variable samples from latent policy in a rolling fashion, initialized with states from the trajectory - Tests latent and subpolicies. 
		# 	c) Latent variables from the ground truth set (only valid for the toy dataset) - Just tests subpolicies. 

		###########################################################
		############# (A) VARIATIONAL POLICY ROLLOUT. #############
		###########################################################

		orig_assembled_inputs, orig_subpolicy_inputs, variational_segmentation = self.rollout_variational_network(counter, i)		

		###########################################################
		################ (B) LATENT POLICY ROLLOUT. ###############
		###########################################################

		latent_segmentation = self.rollout_latent_policy(orig_assembled_inputs, orig_subpolicy_inputs)

		if get_image==True:
			latent_rollout_image = self.visualize_trajectory(self.latent_trajectory_rollout, latent_segmentation)
			variational_rollout_image = self.visualize_trajectory(self.variational_trajectory_rollout, segmentations=variational_segmentation.detach().cpu().numpy())	

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

				if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
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
		if (self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk') and (self.args.fix_subpolicy==0):
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

class PolicyManager_BaselineRL(PolicyManager_BaseClass):

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
		self.number_test_episodes = 100

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_episodes)
		self.number_episodes = 5000000

		# Orhnstein Ullenhbeck noise process parameters. 
		self.theta = 0.15
		self.sigma = 0.2		

		self.reset_statistics()

	def create_networks(self):

		if self.args.MLP_policy:
			self.policy_network = ContinuousMLP(self.input_size, self.args.hidden_size, self.output_size, self.args).cuda()
			self.critic_network = CriticMLP(self.input_size, self.args.hidden_size, 1, self.args).cuda()
		else:
			# Create policy and critic. 		
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers, small_init=True).cuda()			
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
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		self.input_size = self.state_size + self.output_size		
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()
		
		self.initialize_plots()

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size)

	def set_parameters(self, episode_counter, evaluate=False):
		if self.args.train and not(evaluate):
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
		self.terminal_trajectory = []
		self.cummulative_rewards = None
		self.episode = None

	def get_action(self, hidden=None, random=True):

		# Change this to epsilon greedy...
		if random==False:
			whether_greedy = np.random.binomial(n=1,p=0.8)
		else:
			action = 2*np.random.random((self.output_size))-1

		if whether_greedy:
			# Assemble states of current input row.
			current_input_row = self.get_current_input_row()

			# Using the incremental get actions. Still get action greedily, then add noise. 		
			predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).cuda().float(), greedy=True, hidden=hidden)

			if test:
				noise = torch.zeros_like(predicted_action).cuda().float()
			else:
				# Get noise from noise process. 					
				noise = torch.randn_like(predicted_action).cuda().float()*self.epsilon

			# Perturb action with noise. 			
			perturbed_action = predicted_action + noise

			if self.args.MLP_policy:
				action = perturbed_action[-1].detach().cpu().numpy()
			else:
				action = perturbed_action[-1].squeeze(0).detach().cpu().numpy()		
		else:
			action = 2*np.random.random((self.output_size))-1

		return action, hidden

	def get_OU_action(self, hidden=None, random=False, counter=0, evaluate=False):

		if random==True:
			action = 2*np.random.random((self.output_size))-1
			return action, hidden
		
		# Assemble states of current input row.
		current_input_row = self.get_current_input_row()

		# Using the incremental get actions. Still get action greedily, then add noise. 		
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).cuda().float(), greedy=True, hidden=hidden)

		# Numpy action
		if self.args.MLP_policy:
			action = predicted_action[-1].detach().cpu().numpy()
		else:
			action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		

		if evaluate:
			perturbed_action = action
		else:
			# Perturb action with noise. 			
			perturbed_action = self.NoiseProcess.get_action(action, counter)

		return perturbed_action, hidden

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
		# self.terminal_trajectory.append(terminal)
		# self.reward_trajectory.append(0.)		

		hidden = None

		while not(terminal) and counter<self.max_timesteps:

			# action, hidden = self.get_action(hidden=hidden,random=random)
			action, hidden = self.get_OU_action(hidden=hidden,random=random,counter=counter, evaluate=test)
				
			# Take a step in the environment. 	
			next_state, onestep_reward, terminal, success = self.environment.step(action)
		
			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			self.reward_trajectory.append(onestep_reward)
			self.terminal_trajectory.append(terminal)
				
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

		self.episode_reward_statistics = copy.deepcopy(self.cummulative_rewards[0])
		print("Achieved reward: ", self.episode_reward_statistics)
		# print("########################################################")

		# NOW construct an episode out of this..	
		self.episode = RLUtils.Episode(self.state_trajectory, self.action_trajectory, self.reward_trajectory, self.terminal_trajectory)
		# Since we're doing TD updates, we DON'T want to use the cummulative reward, but rather the reward trajectory itself.

	def get_current_input_row(self):
		if len(self.action_trajectory)>0:
			return np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1)),self.action_trajectory[-1].reshape((1,-1))],axis=1)
		else:
			return np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1)),np.zeros((1,self.output_size))],axis=1)

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
		self.reward_trajectory = episode.reward_list
		self.terminal_trajectory = episode.terminal_list

		assembled_inputs = self.assemble_inputs()

		# Input to the policy should be states and actions. 
		self.state_action_inputs = torch.tensor(assembled_inputs).cuda().float()	

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

	def set_differentiable_critic_inputs(self):
		# Get policy's predicted actions by getting action greedily, then add noise. 				
		predicted_action = self.policy_network.reparameterized_get_actions(self.state_action_inputs, greedy=True).squeeze(1)
		noise = torch.zeros_like(predicted_action).cuda().float()
		
		# Get noise from noise process. 					
		noise = torch.randn_like(predicted_action).cuda().float()*self.epsilon

		# Concatenate the states from policy inputs and the predicted actions. 
		self.critic_inputs = torch.cat([self.state_action_inputs[:,:self.state_size], predicted_action],axis=1).cuda().float()

	def update_policies(self):
		######################################
		# Compute losses for actor.
		self.set_differentiable_critic_inputs()		

		self.policy_optimizer.zero_grad()
		self.policy_loss = - self.critic_network.forward(self.critic_inputs[:-1]).mean()
		self.policy_loss_statistics += self.policy_loss.clone().detach().cpu().numpy().mean()
		self.policy_loss.backward()
		self.policy_optimizer.step()

	def set_targets(self):
		if self.args.TD:
			# Construct TD Targets. 
			self.TD_targets = self.critic_predictions.clone().detach().cpu().numpy()
			# Select till last time step, because we don't care what critic says after last timestep.
			self.TD_targets = np.roll(self.TD_targets,-1,axis=0)[:-1]
			# Mask with terminal. 
			self.TD_targets = self.gamma*np.array(self.terminal_trajectory)*self.TD_targets		
			self.TD_targets += np.array(self.reward_trajectory)
			self.critic_targets = torch.tensor(self.TD_targets).cuda().float()
		else:
			self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])
			self.critic_targets = torch.tensor(self.cummulative_rewards).cuda().float()

	def update_critic(self):
		######################################
		# Zero gradients, then backprop into critic.		
		self.critic_optimizer.zero_grad()	
		# Get critic predictions first. 
		if self.args.MLP_policy:
			self.critic_predictions = self.critic_network.forward(self.state_action_inputs).squeeze(1)
		else:
			self.critic_predictions = self.critic_network.forward(self.state_action_inputs).squeeze(1).squeeze(1)

		# Before we actually compute loss, compute targets.
		self.set_targets()

		# We predicted critic values from states S_1 to S_{T+1} because we needed all for bootstrapping. 
		# For loss, we don't actually need S_{T+1}, so throw it out.
		self.critic_loss = self.MSE_Loss(self.critic_predictions[:-1], self.critic_targets).mean()
		self.critic_loss_statistics += self.critic_loss.clone().detach().cpu().numpy().mean()	
		self.critic_loss.backward()
		self.critic_optimizer.step()
		######################################

	def update_networks(self):
		# Update policy network. 
		self.update_policies()
		# Now update critic network.
		self.update_critic()

	def reset_statistics(self):
		# Can also reset the policy and critic loss statistcs here. 
		self.policy_loss_statistics = 0.
		self.critic_loss_statistics = 0.
		self.batch_reward_statistics = 0.
		self.episode_reward_statistics = 0.

	def update_batch(self):

		# Get set of indices of episodes in the memory. 
		batch_indices = self.memory.sample_batch(self.batch_size)

		for ind in batch_indices:

			# Retrieve appropriate episode from memory. 
			episode = self.memory.memory[ind]

			# Set quantities from episode.
			self.process_episode(episode)

			# Now compute gradients to both networks from batch.
			self.update_networks()

	def update_plots(self, counter):
		self.tf_logger.scalar_summary('Total Episode Reward', copy.deepcopy(self.episode_reward_statistics), counter)
		self.tf_logger.scalar_summary('Batch Rewards', self.batch_reward_statistics/self.batch_size, counter)
		self.tf_logger.scalar_summary('Policy Loss', self.policy_loss_statistics/self.batch_size, counter)
		self.tf_logger.scalar_summary('Critic Loss', self.critic_loss_statistics/self.batch_size, counter)

		if counter%self.args.display_freq==0:

			# print("Embedding in Update Plots.")
			
			# Rollout policy.
			self.rollout(random=False, test=True, visualize=True)
			self.tf_logger.gif_summary("Rollout Trajectory", [np.array(self.image_trajectory)], counter)

		# Now that we've updated these into TB, reset stats. 
		self.reset_statistics()

	def run_iteration(self, counter, evaluate=False):

		# This is really a run episode function. Ignore the index, just use the counter. 
		# 1) 	Rollout trajectory. 
		# 2) 	Collect stats / append to memory and stuff.
		# 3) 	Update policies. 

		self.set_parameters(counter, evaluate=evaluate)
		# Maintain counter to keep track of updating the policy regularly. 			

		# cProfile.runctx('self.rollout()',globals(), locals(),sort='cumtime')
		self.rollout(random=False, test=evaluate)

		if self.args.train and not(evaluate):

			# If training, append to memory. 
			self.memory.append_to_memory(self.episode)
			# Update on batch. 
			self.update_batch()
			# Update plots. 
			self.update_plots(counter)

	def initialize_memory(self):

		# Create memory object. 
		self.memory = RLUtils.ReplayMemory(memory_size=self.args.memory_size)

		# Number of initial episodes needs to be less than memory size. 
		self.initial_episodes = self.args.burn_in_eps

		# While number of transitions is less than initial_transitions.
		episode_counter = 0 
		while episode_counter<self.initial_episodes:

			print("Initializing Memory Episode: ", episode_counter)
			# Rollout an episode.
			self.rollout(random=True)

			# Add episode to memory.
			self.memory.append_to_memory(self.episode)

			episode_counter += 1			

	def evaluate(self, model=None):

		if model is not None:
			print("Loading model in training.")
			self.load_all_models(model)

		self.total_rewards = np.zeros((self.number_test_episodes))

		# For number of test episodes. 
		for eps in range(self.number_test_episodes):
			# Run an iteration (and rollout)...
			self.run_iteration(eps, evaluate=True)
			self.total_rewards[eps] = np.array(self.reward_trajectory).sum()

		# Create save directory to save these results. 
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.mean_rewards)

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
			print("#############################")
			print("Running Episode: ",e)

			if e%self.args.eval_freq==0:
				self.evaluate(e)

class PolicyManager_DownstreamRL(PolicyManager_BaselineRL):

	def __init__(self, args):

		super(PolicyManager_DownstreamRL, self).__init__(args)

	def setup(self):
		# Create Mujoco environment. 
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.environment.action_spec[0].shape[0]
		self.conditional_info_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		self.input_size = 2*self.state_size
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()
		
		self.initialize_plots()

		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size)

	def create_networks(self):
		# Copying over the create networks from Joint Policy training. 

		# Not sure if there's a better way to inherit - unless we inherit from both classes.
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers).cuda()				
		self.critic_network = CriticNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, 1, self.args, self.args.number_layers).cuda()
		self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, self.args, self.args.number_layers).cuda()

	def create_training_ops(self):
		
		self.NLL_Loss = torch.nn.NLLLoss(reduction='none')
		self.MSE_Loss = torch.nn.MSELoss(reduction='none')
		
		# If we are using reparameterization, use a global optimizer for both policies, and a global loss function.
		parameter_list = list(self.policy_network.parameters()) + list(self.latent_policy.parameters())
		# The policy optimizer handles both the low and high level policies, as long as the z's being passed from the latent to sub policy are differentiable.
		self.policy_optimizer = torch.optim.Adam(parameter_list, lr=self.learning_rate)
		self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters(), lr=self.learning_rate)

	def save_all_models(self, suffix):
		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")	
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Latent_Policy'] = self.latent_policy.state_dict()
		save_object['Critic_Network'] = self.critic_network.state_dict()
		
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		self.latent_policy.load_state_dict(load_object['Latent_Policy'])
		self.critic_network.load_state_dict(load_object['Critic_Network'])

	def reset_lists(self):
		self.reward_trajectory = []
		self.state_trajectory = []
		self.action_trajectory = []
		self.image_trajectory = []
		self.terminal_trajectory = []
		self.latent_z_trajectory = []
		self.latent_b_trajectory = []
		self.cummulative_rewards = None
		self.episode = None

	def get_conditional_information_row(self, t=-1):
		# Get robot and object state.
		return np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1)		

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

		finger_diff = gripper_values[1]-gripper_values[0]
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))
		return gripper_value

	def get_current_input_row(self, t=-1):

		# The state that we want is ... joint state?
		gripper_finger_values = self.state_trajectory[t]['gripper_qpos']

		if len(self.action_trajectory)==0 or t==0:
			return np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), np.zeros((1,1)), np.zeros((1,self.output_size))],axis=1)
		elif t==-1:
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t].reshape((1,-1))],axis=1)
		else: 
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t-1].reshape((1,-1))],axis=1)

	def get_latent_input_row(self, t=-1):
		# If first timestep, z's are 0 and b is 1. 
		if len(self.latent_z_trajectory)==0 or t==0:
			return np.concatenate([np.zeros((1, self.args.z_dimensions)),np.ones((1,1))],axis=1)
		if t==-1:
			return np.concatenate([self.latent_z_trajectory[t].reshape((1,-1)),self.latent_b_trajectory[t].reshape((1,1))],axis=1)
		elif t>0:
			t-=1	
			return np.concatenate([self.latent_z_trajectory[t].reshape((1,-1)),self.latent_b_trajectory[t].reshape((1,1))],axis=1)

	def assemble_latent_input_row(self, t=-1):
		# Function to assemble ONE ROW of latent policy input. 
		# Remember, the latent policy takes.. JOINT_states, actions, z's, b's, and then conditional information of robot-state and object-state. 

		# Assemble these three pieces: 
		return np.concatenate([self.get_current_input_row(t), self.get_latent_input_row(t), self.get_conditional_information_row(t)],axis=1)

	def assemble_latent_inputs(self):
		# Assemble latent policy inputs over time.
		return np.concatenate([self.assemble_latent_input_row(t) for t in range(len(self.state_trajectory))],axis=0)		

	def assemble_subpolicy_input_row(self, latent_z=None, t=-1):
		# Remember, the subpolicy takes.. JOINT_states, actions, z's. 
		# Assemble (remember, without b, and without conditional info).

		if latent_z is not None:
			# return np.concatenate([self.get_current_input_row(t), latent_z.reshape((1,-1))],axis=1)

			# Instead of numpy, use torch. 
			return torch.cat([torch.tensor(self.get_current_input_row(t)).cuda().float(), latent_z.reshape((1,-1))],dim=1)
		else:
			# Remember, get_latent_input_row isn't operating on something that needs to be differentiable, so just use numpy and then wrap with torch tensor. 
			# return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_latent_input_row(t)[:,:-1]],axis=1)).cuda().float()
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.latent_z_trajectory[t].reshape((1,-1))],axis=1)).cuda().float()

	def assemble_subpolicy_inputs(self, latent_z_list=None):
		# Assemble sub policy inputs over time.	
		if latent_z_list is None:
			# return np.concatenate([self.assemble_subpolicy_input_row(t) for t in range(len(self.state_trajectory))],axis=0)

			# Instead of numpy, use torch... 
			return torch.cat([self.assemble_subpolicy_input_row(t=t) for t in range(len(self.state_trajectory))],dim=0)
		else:
			# return np.concatenate([self.assemble_subpolicy_input_row(t, latent_z=latent_z_list[t]) for t in range(len(self.state_trajectory))],axis=0)

			# Instead of numpy, use torch... 
			return torch.cat([self.assemble_subpolicy_input_row(t=t, latent_z=latent_z_list[t]) for t in range(len(self.state_trajectory))],dim=0)

	def assemble_state_action_row(self, action=None, t=-1):
		# Get state action input row for critic.
		if action is not None:

			gripper_finger_values = self.state_trajectory[t]['gripper_qpos']
			gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

			finger_diff = gripper_values[1]-gripper_values[0]
			gripper_value = np.array(2*finger_diff-1).reshape((1,-1))

			# Don't create a torch tensor out of actions. 
			return torch.cat([torch.tensor(self.state_trajectory[t]['joint_pos']).cuda().float().reshape((1,-1)), torch.tensor(gripper_value).cuda().float(), action.reshape((1,-1)), torch.tensor(self.get_conditional_information_row(t)).cuda().float()],dim=1)
		else:		
			# Just use actions that were used in the trajectory. This doesn't need to be differentiable, because it's going to be used for the critic targets, so just make a torch tensor from numpy. 
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_conditional_information_row(t)],axis=1)).cuda().float()

	def assemble_state_action_inputs(self, action_list=None):
		# return np.concatenate([self.assemble_state_action_row(t) for t in range(len(self.state_trajectory))],axis=0)
		
		# Instead of numpy use torch.
		if action_list is not None:
			return torch.cat([self.assemble_state_action_row(t=t, action=action_list[t]) for t in range(len(self.state_trajectory))],dim=0)
		else:
			return torch.cat([self.assemble_state_action_row(t=t) for t in range(len(self.state_trajectory))],dim=0)

	def get_OU_action_latents(self, policy_hidden=None, latent_hidden=None, random=False, counter=0, previous_z=None, test=False):

		# if random==True:
		# 	action = 2*np.random.random((self.output_size))-1
		# 	return action, 

		# Get latent policy inputs.
		latent_policy_inputs = self.assemble_latent_input_row()
		
		# Feed in latent policy inputs and get the latent policy outputs (z, b, and hidden)
		latent_z, latent_b, latent_hidden = self.latent_policy.incremental_reparam_get_actions(torch.tensor(latent_policy_inputs).cuda().float(), greedy=True, hidden=latent_hidden, previous_z=previous_z)

		# Now get subpolicy inputs.
		# subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z.detach().cpu().numpy())
		subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z=latent_z)

		# Feed in subpolicy inputs and get the subpolicy outputs (a, hidden)
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(subpolicy_inputs).cuda().float(), greedy=True, hidden=policy_hidden)

		# Numpy action
		action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		
		
		if test:
			perturbed_action = action
		else:	
			# Perturb action with noise. 			
			perturbed_action = self.NoiseProcess.get_action(action, counter)

		return perturbed_action, latent_z, latent_b, policy_hidden, latent_hidden

	def rollout(self, random=False, test=False, visualize=False):
		
		# Reset some data for the rollout. 
		counter = 0		
		eps_reward = 0.			
		terminal = False
		self.reset_lists()

		# Reset environment and add state to the list.
		state = self.environment.reset()
		self.state_trajectory.append(state)		

		# If we are going to visualize, get an initial image.
		if visualize:			
			image = self.environment.sim.render(600,600, camera_name='frontview')
			self.image_trajectory.append(np.flipud(image))

		# Instead of maintaining just one LSTM hidden state... now have one for each policy level.
		policy_hidden = None
		latent_hidden = None
		latent_z = None

		# For number of steps / while we don't terminate:
		while not(terminal) and counter<self.max_timesteps:

			# Get the action to execute, b, z, and hidden states. 
			action, latent_z, latent_b, policy_hidden, latent_hidden = self.get_OU_action_latents(policy_hidden=policy_hidden, latent_hidden=latent_hidden, random=random, counter=counter, previous_z=latent_z, test=test)
				
			# Take a step in the environment. 	
			next_state, onestep_reward, terminal, success = self.environment.step(action)
			
			# Append everything to lists. 
			self.state_trajectory.append(next_state)
			self.action_trajectory.append(action)
			self.reward_trajectory.append(onestep_reward)
			self.terminal_trajectory.append(terminal)
			self.latent_z_trajectory.append(latent_z.detach().cpu().numpy())
			self.latent_b_trajectory.append(latent_b.detach().cpu().numpy())

			# Copy next state into state. 
			state = copy.deepcopy(next_state)

			# Counter
			counter += 1 

			# Append image to image list if we are visualizing. 
			if visualize:
				image = self.environment.sim.render(600,600, camera_name='frontview')
				self.image_trajectory.append(np.flipud(image))
				
		# Now that the episode is done, compute cummulative rewards... 
		self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])
		self.episode_reward_statistics = copy.deepcopy(self.cummulative_rewards[0])
		
		print("Rolled out an episode for ",counter," timesteps.")
		print("Achieved reward: ", self.episode_reward_statistics)

		# NOW construct an episode out of this..	
		self.episode = RLUtils.HierarchicalEpisode(self.state_trajectory, self.action_trajectory, self.reward_trajectory, self.terminal_trajectory, self.latent_z_trajectory, self.latent_b_trajectory)

	def process_episode(self, episode):
		# Assemble states, actions, targets.

		# First reset all the lists from the rollout now that they've been written to memory. 
		self.reset_lists()

		# Now set the lists. 
		self.state_trajectory = episode.state_list
		self.action_trajectory = episode.action_list
		self.reward_trajectory = episode.reward_list
		self.terminal_trajectory = episode.terminal_list
		self.latent_z_trajectory = episode.latent_z_list
		self.latent_b_trajectory = episode.latent_b_list

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

		# Assembling state_action inputs to feed to the Critic network for TARGETS. (These don't need to, and in fact shouldn't, be differentiable).
		self.state_action_inputs = torch.tensor(self.assemble_state_action_inputs()).cuda().float()

	def update_policies(self):
		# There are a few steps that need to be taken. 
		# 1) Assemble latent policy inputs.
		# 2) Get differentiable latent z's from latent policy. 
		# 3) Assemble subpolicy inputs with these differentiable latent z's. 
		# 4) Get differentiable actions from subpolicy. 
		# 5) Assemble critic inputs with these differentiable actions. 
		# 6) Now compute critic predictions that are differentiable w.r.t. sub and latent policies. 
		# 7) Backprop.

		# 1) Assemble latent policy inputs. # Remember, these are the only things that don't need to be differentiable.
		self.latent_policy_inputs = torch.tensor(self.assemble_latent_inputs()).cuda().float()		

		# 2) Feed this into latent policy. 
		latent_z, latent_b, _ = self.latent_policy.incremental_reparam_get_actions(torch.tensor(self.latent_policy_inputs).cuda().float(), greedy=True)

		# 3) Assemble subpolicy inputs with diff latent z's. Remember, this needs to be differentiable. Modify the assembling to torch, WITHOUT creating new torch tensors of z. 

		self.subpolicy_inputs = self.assemble_subpolicy_inputs(latent_z_list=latent_z)

		# 4) Feed into subpolicy. 
		diff_actions, _ = self.policy_network.incremental_reparam_get_actions(self.subpolicy_inputs, greedy=True)

		# 5) Now assemble critic inputs. 
		self.differentiable_critic_inputs = self.assemble_state_action_inputs(action_list=diff_actions)

		# 6) Compute critic predictions. 
		self.policy_loss = - self.critic_network.forward(self.differentiable_critic_inputs[:-1]).mean()

		# Also log statistics. 
		self.policy_loss_statistics += self.policy_loss.clone().detach().cpu().numpy().mean()

		# 7) Now backprop into policy.
		self.policy_optimizer.zero_grad()		
		self.policy_loss.backward()
		self.policy_optimizer.step()

class PolicyManager_DMPBaselines(PolicyManager_Joint):

	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_DMPBaselines, self).__init__(number_policies, dataset, args)

		self.number_kernels = 30
		self.window = 8
		self.kernel_bandwidth = 1.5

		self.number_kernels = self.args.baseline_kernels
		self.window = self.args.baseline_window
		self.kernel_bandwidth = self.args.baseline_kernel_bandwidth

	def get_MSE(self, sample_traj, trajectory_rollout):
		# Evaluate MSE between reconstruction and sample trajectory. 
		return ((sample_traj-trajectory_rollout)**2).mean()

	def get_FlatDMP_rollout(self, sample_traj, velocities=None):
		# Reinitialize DMP Class. 
		self.dmp = DMP.DMP(time_steps=len(sample_traj), num_ker=self.number_kernels, dimensions=self.state_size, kernel_bandwidth=self.kernel_bandwidth, alphaz=5., time_basis=True)

		# Learn DMP for particular trajectory.
		self.dmp.learn_DMP(sample_traj)

		# Get rollout. 
		if velocities is not None: 
			trajectory_rollout = self.dmp.rollout(sample_traj[0],sample_traj[-1],velocities)
		else:
			trajectory_rollout = self.dmp.rollout(sample_traj[0],sample_traj[-1],np.zeros((self.state_size)))

		return trajectory_rollout

	def evaluate_FlatDMPBaseline_iteration(self, index, sample_traj):
		trajectory_rollout = self.get_FlatDMP_rollout(sample_traj)
		self.FlatDMP_distances[index] = self.get_MSE(sample_traj, trajectory_rollout)

	def get_AccelerationChangepoint_rollout(self, sample_traj):

		# Get magnitudes of acceleration across time.
		acceleration_norm = np.linalg.norm(np.diff(sample_traj,n=2,axis=0),axis=1)

		# Get velocities. 
		velocities = np.diff(sample_traj,n=1,axis=0,prepend=sample_traj[0].reshape((1,-1)))

		# Find peaks with minimum length = 8.
		window = self.window
		segmentation = find_peaks(acceleration_norm, distance=window)[0]
		
		# Add start and end to peaks. 
		if segmentation[0]<window:
			segmentation[0] = 0
		else:
			segmentation = np.insert(segmentation, 0, 0)
		# If end segmentation is within WINDOW of end, change segment to end. 
		if (len(sample_traj) - segmentation[-1])<window:
			segmentation[-1] = len(sample_traj)
		else:
			segmentation = np.insert(segmentation, len(segmentation), sample_traj.shape[0])

		trajectory_rollout = np.zeros_like(sample_traj)		

		# For every segment.
		for i in range(len(segmentation)-1):
			# Get trajectory segment. 
			trajectory_segment = sample_traj[segmentation[i]:segmentation[i+1]]

			# Get rollout. # Feed velocities into rollout. # First velocity is 0. 
			segment_rollout = self.get_FlatDMP_rollout(trajectory_segment, velocities[segmentation[i]])

			# Copy segment rollout into full rollout. 
			trajectory_rollout[segmentation[i]:segmentation[i+1]] = segment_rollout

		return trajectory_rollout

	def evaluate_AccelerationChangepoint_iteration(self, index, sample_traj):
		trajectory_rollout = self.get_AccelerationChangepoint_rollout(sample_traj)
		self.AccChangepointDMP_distances[index] = self.get_MSE(sample_traj, trajectory_rollout)

	def evaluate_MeanRegression_iteration(self, index, sample_traj):
		mean = sample_traj.mean(axis=0)
		self.MeanRegression_distances[index] = ((sample_traj-mean)**2).mean()

	def get_GreedyDMP_rollout(self, sample_traj):
		pass		

	def evaluate_across_testset(self):

		# Create array for distances. 
		self.FlatDMP_distances = -np.ones((self.test_set_size))
		self.AccChangepointDMP_distances = -np.ones((self.test_set_size))
		self.MeanRegression_distances = -np.ones((self.test_set_size))
		self.lengths = -np.ones((self.test_set_size))

		for i in range(self.test_set_size):

			# Set actual index. 
			index = i + len(self.dataset) - self.test_set_size

			if i%100==0:
				print("Evaluating Datapoint ", i)

			# Get trajectory. 
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

			if sample_traj is not None: 

				# Set sample trajectory to ignore gripper. 
				if self.args.data=='MIME':
					sample_traj = sample_traj[:,:-2]
					self.state_size = 14
				elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
					sample_traj = sample_traj[:,:-1]
					self.state_size = 7
					# sample_traj = gaussian_filter1d(sample_traj,3.5,axis=0,mode='nearest')
					
				self.lengths[i] = len(sample_traj)

				# Eval Flat DMP.
				self.evaluate_FlatDMPBaseline_iteration(i, sample_traj)

				# Eval AccChange DMP Baseline.
				self.evaluate_AccelerationChangepoint_iteration(i, sample_traj)

				# Evaluate Mean regression Basleine. 
				self.evaluate_MeanRegression_iteration(i, sample_traj)

		# self.mean_distance = self.distances[self.distances>0].mean()		
		print("Average Distance of Flat DMP Baseline: ", self.FlatDMP_distances[self.FlatDMP_distances>0].mean())
		print("Average Distance of Acceleration Changepoint Baseline: ", self.AccChangepointDMP_distances[self.AccChangepointDMP_distances>0].mean())
		print("Average Distance of Mean Regression Baseline: ", self.MeanRegression_distances[self.MeanRegression_distances>0].mean())

		embed()

class PolicyManager_Imitation(PolicyManager_Pretrain, PolicyManager_BaselineRL):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_Imitation, self).__init__(number_policies, dataset, args)

		# Set train only policy to true.
		self.args.train_only_policy = 1

		# Get task index from task name.
		self.demo_task_index = np.where(np.array(self.dataset.environment_names)==self.args.environment)[0][0]

	def create_networks(self):

		# We don't need a decoder.
		# Policy Network is the only thing we need.
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).cuda()

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])

	def update_policies(self, logprobabilities):

		# Set gradients to 0.
		self.optimizer.zero_grad()

		# Set policy loss. 
		self.policy_loss = -logprobabilities[:-1].mean()

		# Backward. 
		self.policy_loss.backward()

		# Take a step. 
		self.optimizer.step()

	def update_plots(self, counter, logprobabilities):
		self.tf_logger.scalar_summary('Policy LogLikelihood', torch.mean(logprobabilities), counter)

	def run_iteration(self, counter, i):

		self.set_epoch(counter)	
		self.iter = counter

		############# (0) #############
		# Get sample we're going to train on.		
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		if sample_traj is not None:

			# Add zeros to the last action, so that we evaluate likelihood correctly. 
			padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			# Feed concatenated trajectory into the policy. 
			logprobabilities, _ = self.policy_network.forward(torch.tensor(concatenated_traj).cuda().float(), padded_action_seq)

			if self.args.train:

				if self.args.debug:
					if self.iter%self.args.debug==0:
						print("Embedding in Train Function.")
						embed()
				
				# Update policy. 						
				self.update_policies(logprobabilities)

				# Update plots.
				self.update_plots(counter, logprobabilities)

	def evaluate(self, model=None):

		if model is not None:
			self.load_all_models(model)

		self.total_rewards = np.zeros((self.number_test_episodes))

		# Set parameters like epsilon.
		self.set_parameters(0, evaluate=True)

		# For number of test episodes. 
		for eps in range(self.number_test_episodes):
			# Now run a rollout. 
			self.rollout(random=False, test=True)

			self.total_rewards[eps] = np.array(self.reward_trajectory).sum()

		# Create save directory to save these results. 
		upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

		if not(os.path.isdir(upper_dir_name)):
			os.mkdir(upper_dir_name)

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.mean_rewards)
