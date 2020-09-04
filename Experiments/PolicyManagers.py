# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from headers import *
from PolicyNetworks import *
from Visualizers import BaxterVisualizer, SawyerVisualizer, ToyDataVisualizer #, MocapVisualizer
import TFLogger, DMP, RLUtils

# Check if CUDA is available, set device to GPU if it is, otherwise use CPU.
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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
		if (self.args.setting=='transfer' and isinstance(self, PolicyManager_Transfer)) or \
			(self.args.setting=='cycle_transfer' and isinstance(self, PolicyManager_CycleConsistencyTransfer)):
				extent = self.extent
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

		if self.args.data=='MIME':
			self.visualizer = BaxterVisualizer()
			# self.state_dim = 16
		elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
			self.visualizer = SawyerVisualizer()
			# self.state_dim = 8
		elif self.args.data=='Mocap':
			self.visualizer = MocapVisualizer(args=self.args)
		else: 
			self.visualizer = ToyDataVisualizer()

		self.rollout_gif_list = []
		self.gt_gif_list = []

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

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

			trajectory = data_element['demo']

			# If normalization is set to some value.
			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				trajectory = (trajectory-self.norm_sub_value)/self.norm_denom_value

			action_sequence = np.diff(trajectory,axis=0)

			self.current_traj_len = len(trajectory)

			if self.args.data=='MIME':
				self.conditional_information = np.zeros((self.conditional_info_size))				
			elif self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk':
				robot_states = data_element['robot-state']
				object_states = data_element['object-state']

				self.conditional_information = np.zeros((self.conditional_info_size))
				# Don't set this if pretraining / baseline.
				if self.args.setting=='learntsub' or self.args.setting=='imitation':
					self.conditional_information = np.zeros((len(trajectory),self.conditional_info_size))
					self.conditional_information[:,:self.cond_robot_state_size] = robot_states
					# Doing this instead of self.cond_robot_state_size: because the object_states size varies across demonstrations.
					self.conditional_information[:,self.cond_robot_state_size:self.cond_robot_state_size+object_states.shape[-1]] = object_states	
					# Setting task ID too.		
					self.conditional_information[:,-self.number_tasks+data_element['task-id']] = 1.
			# Concatenate
			concatenated_traj = self.concat_state_action(trajectory, action_sequence)
			old_concatenated_traj = self.old_concat_state_action(trajectory, action_sequence)

			if self.args.setting=='imitation':
				action_sequence = RLUtils.resample(data_element['demonstrated_actions'],len(trajectory))
				concatenated_traj = np.concatenate([trajectory, action_sequence],axis=1)

			return trajectory, action_sequence, concatenated_traj, old_concatenated_traj

	def train(self, model=None):

		if model:
			print("Loading model in training.")
			self.load_all_models(model)		
		counter = self.args.initial_counter_value

		# For number of training epochs. 
		for e in range(self.number_epochs): 
			
			self.current_epoch_running = e
			print("Starting Epoch: ",e)

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			np.random.shuffle(self.index_list)

			if self.args.debug:
				print("Embedding in Outer Train Function.")
				embed()

			# For every item in the epoch:
			if self.args.setting=='imitation':
				extent = self.dataset.get_number_task_demos(self.demo_task_index)
			if self.args.setting=='transfer' or self.args.setting=='cycle_transfer':
				extent = self.extent
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
			base_command = './xvfb-run-safe ' + base_command

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
			# Because there are just more invalid DP's in Mocap.
			self.N = 100
		else: 
			self.visualizer = ToyDataVisualizer()

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
				
				trajectory_rollout = self.get_robot_visuals(i, latent_z, sample_traj)
				
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

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()
		subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()	

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
			input_row = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[0,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[0,self.state_dim:2*self.state_dim] = actions[-1].squeeze(1)
			input_row[0,2*self.state_dim:] = latent_z

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		return trajectory

	def get_robot_visuals(self, i, latent_z, trajectory, return_image=False):		

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
		ground_truth_gif = self.visualizer.visualize_joint_trajectory(unnorm_gt_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_GT.gif".format(i), return_and_save=True)
		
		# 4) Run unnormalized rollout trajectory in visualizer. 
		rollout_gif = self.visualizer.visualize_joint_trajectory(unnorm_pred_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_Rollout.gif".format(i), return_and_save=True)
		
		self.gt_gif_list.append(copy.deepcopy(ground_truth_gif))
		self.rollout_gif_list.append(copy.deepcopy(rollout_gif))

		if self.args.normalization=='meanvar' or self.args.normalization=='minmax':

			if return_image:
				return unnorm_pred_trajectory, ground_truth_gif, rollout_gif
			else:
				return unnorm_pred_trajectory
		else:

			if return_image:
				return trajectory_rollout, ground_truth_gif, rollout_gif
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

	def get_robot_embedding(self, return_tsne_object=False):

		# Mean and variance normalize z.
		mean = self.latent_z_set.mean(axis=0)
		std = self.latent_z_set.std(axis=0)
		normed_z = (self.latent_z_set-mean)/std
		
		tsne = skl_manifold.TSNE(n_components=2,random_state=0,perplexity=self.args.perplexity)
		embedded_zs = tsne.fit_transform(normed_z)

		scale_factor = 1
		scaled_embedded_zs = scale_factor*embedded_zs

		if return_tsne_object:
			return scaled_embedded_zs, tsne
		else:
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
		ax.set_title("Embedding of Latent Representation of our Model",fontdict={'fontsize':40})
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

		if args.setting=='imitation':
			super(PolicyManager_Pretrain, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		else:
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
		self.state_dim = 2
		self.input_size = 2*self.state_size
		self.hidden_size = self.args.hidden_size
		# Number of actions
		self.output_size = 2		
		self.latent_z_dimensionality = self.args.z_dimensions
		self.number_layers = self.args.number_layers
		self.traj_length = 5
		self.number_epochs = self.args.epochs
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
			self.number_epochs = self.args.epochs

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
			self.conditional_info_size = 0

		# Training parameters. 		
		self.baseline_value = 0.
		self.beta_decay = 0.9
		self. learning_rate = self.args.learning_rate
		
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
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.number_policies, self.number_layers).to(device)
		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.latent_z_dimensionality, self.number_layers).to(device)
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

		# Create encoder.
		if self.args.discrete_z: 
			# The latent space is just one of 4 z's. So make output of encoder a one hot vector.		
			self.encoder_network = EncoderNetwork(self.input_size, self.hidden_size, self.number_policies).to(device)
		else:
			# self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality).to(device)

			# if self.args.transformer:
			# 	self.encoder_network = TransformerEncoder(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)
			# else:
			self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)		

	def create_training_ops(self):
		# self.negative_log_likelihood_loss_function = torch.nn.NLLLoss()
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		self.KLDivergence_loss_function = torch.nn.KLDivLoss(reduction='none')
		# Only need one object of the NLL loss function for the latent net. 

		# These are loss functions. You still instantiate the loss function every time you evaluate it on some particular data. 
		# When you instantiate it, you call backward on that instantiation. That's why you know what loss to optimize when computing gradients. 		

		if self.args.train_only_policy:
			self.parameter_list = self.policy_network.parameters()
		else:
			self.parameter_list = list(self.policy_network.parameters()) + list(self.encoder_network.parameters())
		
		self.optimizer = torch.optim.Adam(self.parameter_list,lr=self.learning_rate)

	def save_all_models(self, suffix):

		logdir = os.path.join(self.args.logdir, self.args.name)
		savedir = os.path.join(logdir,"saved_models")
		if not(os.path.isdir(savedir)):
			os.mkdir(savedir)
		save_object = {}
		save_object['Policy_Network'] = self.policy_network.state_dict()
		save_object['Encoder_Network'] = self.encoder_network.state_dict()
		torch.save(save_object,os.path.join(savedir,"Model_"+suffix))

	def load_all_models(self, path, only_policy=False, just_subpolicy=False):
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
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			subpolicy_inputs[range(len(input_trajectory)),self.input_size+latent_z_indices.long()] = 1.
			# subpolicy_inputs[range(len(input_trajectory)),-1] = latent_b.float()

			# # Concatenated action sqeuence for policy network. 
			padded_action_seq = np.concatenate([sample_action_seq,np.zeros((1,self.output_size))],axis=0)

			return assembled_inputs, subpolicy_inputs, padded_action_seq

		else:
			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()			

			assembled_inputs[range(1,len(input_trajectory)),self.input_size:-1] = latent_z_indices[:-1]
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
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

			self.current_traj_len = self.traj_length

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
			latent_z_indices = latent_z.float()*torch.ones((self.traj_length)).to(device).float()			
		else:
			# This construction should work irrespective of reparam or not.
			latent_z_indices = torch.cat([latent_z.squeeze(0) for i in range(self.current_traj_len)],dim=0)

		# Setting latent_b's to 00001. 
		# This is just a dummy value.
		# latent_b = torch.ones((5)).to(device).float()
		latent_b = torch.zeros((self.current_traj_len)).to(device).float()
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
			subpolicy_inputs = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
			subpolicy_inputs[0,self.input_size+i] = 1. 
		else:
			subpolicy_inputs = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device)
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
				input_row = torch.zeros((1,self.input_size+self.number_policies)).to(device).float()
				input_row[0,self.input_size+i] = 1. 
			else:
				input_row = torch.zeros((1,self.input_size+self.latent_z_dimensionality)).to(device).float()
				input_row[0,self.input_size:] = latent_z
			input_row[0,:self.state_dim] = new_state
			input_row[0,self.state_dim:2*self.state_dim] = action_to_execute	
			# input_row[0,-1] = 1.

			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)
		# print("latent_z:",latent_z)
		trajectory_rollout = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		# print("Trajectory:",trajectory_rollout)

		if return_traj:
			return trajectory_rollout		

	def run_iteration(self, counter, i, return_z=False, and_train=True):

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
			torch_traj_seg = torch.tensor(trajectory_segment).to(device).float()
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
			if self.args.train and and_train:

				# If we are regularizing: 
				# 	(1) Sample another z. 
				# 	(2) Construct inputs and such.
				# 	(3) Compute distances, and feed to update_policies.
				regularization_kl = None
				z_distance = None

				self.update_policies_reparam(loglikelihood, subpolicy_inputs, kl_divergence)

				# Update Plots. 
				self.update_plots(counter, loglikelihood, trajectory_segment)

				if return_z: 
					return latent_z, sample_traj, sample_action_seq

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

	def evaluate(self, model=None, suffix=None):
		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		if self.args.data=='ContinuousNonZero':
			self.visualize_embedding_space(suffix=suffix)

		if self.args.data=="MIME" or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			print("Running Evaluation of State Distances on small test set.")
			# self.evaluate_metrics()

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

			# np.save(os.path.join(self.dir_name,"Trajectory_Distances_{0}.npy".format(self.args.name)),self.distances)
			# np.save(os.path.join(self.dir_name,"Mean_Trajectory_Distance_{0}.npy".format(self.args.name)),self.mean_distance)

	# @profile
	def get_trajectory_and_latent_sets(self):
		# For N number of random trajectories from MIME: 
		#	# Encode trajectory using encoder into latent_z. 
		# 	# Feed latent_z into subpolicy. 
		#	# Rollout subpolicy for t timesteps. 
		#	# Plot rollout.
		# Embed plots. 

		# Set N:
		self.N = 100
		self.rollout_timesteps = 5
		self.state_dim = 2

		self.latent_z_set = np.zeros((self.N,self.latent_z_dimensionality))		
		self.trajectory_set = np.zeros((self.N, self.rollout_timesteps, self.state_dim))

		# Use the dataset to get reasonable trajectories (because without the information bottleneck / KL between N(0,1), cannot just randomly sample.)
		for i in range(self.N):

			# (1) Encoder trajectory. 
			latent_z, _, _ = self.run_iteration(0, i, return_z=True, and_train=False)

			# Copy z. 
			self.latent_z_set[i] = copy.deepcopy(latent_z.detach().cpu().numpy())

			# (2) Now rollout policy.
			self.trajectory_set[i] = self.rollout_visuals(i, latent_z=latent_z, return_traj=True)

			# # (3) Plot trajectory.
			# traj_image = self.visualize_trajectory(rollout_traj)

	def visualize_embedding_space(self, suffix=None):

		self.get_trajectory_and_latent_sets()

		# TSNE on latentz's.
		tsne = skl_manifold.TSNE(n_components=2,random_state=0)
		embedded_zs = tsne.fit_transform(self.latent_z_set)

		ratio = 0.3
		for i in range(self.N):
			plt.scatter(embedded_zs[i,0]+ratio*self.trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.trajectory_set[i,:,1],c=range(self.rollout_timesteps),cmap='jet')

		model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))		
		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		if suffix is not None:
			self.dir_name = os.path.join(self.dir_name, suffix)

		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		# Format with name.
		plt.savefig("{0}/Embedding_Joint_{1}.png".format(self.dir_name,self.args.name))
		plt.close()

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
			self.number_tasks = 8
			self.conditional_info_size = self.cond_robot_state_size+self.cond_object_state_size+self.number_tasks
			self.conditional_viz_env = True

		elif self.args.data=='Mocap':
			self.state_size = 22*3
			self.state_dim = 22*3	
			self.input_size = 2*self.state_size
			self.hidden_size = self.args.hidden_size
			self.output_size = self.state_size
			self.traj_length = self.args.traj_length	
			self.conditional_info_size = 0
			self.conditional_information = None
			self.conditional_viz_env = False

			# Create visualizer object
			self.visualizer = MocapVisualizer(args=self.args)

		self.training_phase_size = self.args.training_phase_size
		self.number_epochs = self.args.epochs
		self.test_set_size = 500
		self.baseline_value = 0.
		self.beta_decay = 0.9

		self. learning_rate = self.args.learning_rate

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
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.number_policies, self.number_layers).to(device)	
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)		

			# Create latent policy, whose action space = self.number_policies. 
			# This policy network automatically manages input size. 

			# Also add conditional_info_size to this. 
			self.latent_policy = LatentPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.number_layers, self.args.b_exploration_bias).to(device)

			# Create variational network. 
			# self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, number_layers=self.number_layers, z_exploration_bias=self.args.z_exploration_bias, b_exploration_bias=self.args.b_exploration_bias).to(device)
			self.variational_policy = VariationalPolicyNetwork(self.input_size, self.hidden_size, self.number_policies, self.args, number_layers=self.number_layers).to(device)

		else:
			# self.policy_network = ContinuousPolicyNetwork(self.input_size,self.hidden_size,self.output_size,self.latent_z_dimensionality, self.number_layers).to(device)
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)			

			if self.args.constrained_b_prior:
				self.latent_policy = ContinuousLatentPolicyNetwork_ConstrainedBPrior(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).to(device)
				
				self.variational_policy = ContinuousVariationalPolicyNetwork_ConstrainedBPrior(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).to(device)

			else:
				# self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.number_layers, self.args.b_exploration_bias).to(device)
				self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.hidden_size, self.args, self.number_layers).to(device)

				self.variational_policy = ContinuousVariationalPolicyNetwork_BPrior(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args, number_layers=self.number_layers).to(device)

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

		if not(just_subpolicy):
			if self.args.load_latent:
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
				print("In Phase 2.")			

			else:
				
				self.training_phase=3
				self.latent_z_loss_weight = 0.01*self.args.lat_b_wt
				# For training phase = 3, set latent_b_loss weight to 1 and latent_z_loss weight to something like 0.1 or 0.01. 
				# After another double training_phase... (i.e. counter>3*self.training_phase_size), 
				# This should be run when counter > 2*self.training_phase_size, and less than 3*self.training_phase_size.
				if counter>3*self.training_phase_size:
					# Set equal after 3. 
					print("In Phase 4.")
					self.latent_z_loss_weight = 0.1*self.args.lat_b_wt
				else:
					print("In Phase 3.")

		else:
			self.epsilon = 0.
			self.training_phase=1

	def visualize_trajectory(self, trajectory, segmentations=None, i=0, suffix='_Img'):

		if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':

			if self.args.normalization=='meanvar' or self.args.normalization=='minmax':
				unnorm_trajectory = (trajectory*self.norm_denom_value)+self.norm_sub_value
			else:
				unnorm_trajectory = trajectory

			if self.args.data=='Mocap':
				# Create save directory:
				upper_dir_name = os.path.join(self.args.logdir,self.args.name,"MEval")

				if not(os.path.isdir(upper_dir_name)):
					os.mkdir(upper_dir_name)

				if self.args.model is not None:
					model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
				else:
					model_epoch = self.current_epoch_running

				self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
				if not(os.path.isdir(self.dir_name)):
					os.mkdir(self.dir_name)

				animation_object = self.dataset[i]['animation']

				return self.visualizer.visualize_joint_trajectory(unnorm_trajectory, gif_path=self.dir_name, gif_name="Traj_{0}_{1}.gif".format(i,suffix), return_and_save=True, additional_info=animation_object)
			else:
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

			gt_trajectory_image = np.array(self.visualize_trajectory(sample_traj, i=i, suffix='GT'))
			variational_rollout_image = np.array(variational_rollout_image)
			latent_rollout_image = np.array(latent_rollout_image)

			if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
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
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies+1)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+latent_z_indices[:-1].long()] = 1.
			assembled_inputs[range(1,len(input_trajectory)),-1] = latent_b[:-1].float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.number_policies)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
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
				latent_z_copy = torch.tensor(latent_z_indices).to(device)
			else:
				latent_z_copy = latent_z_indices

			if conditional_information is None:
				conditional_information = torch.zeros((self.conditional_info_size)).to(device).float()

			# Append latent z indices to sample_traj data to feed as input to BOTH the latent policy network and the subpolicy network. 			
			assembled_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality+1+self.conditional_info_size)).to(device)
			assembled_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()			
			assembled_inputs[range(1,len(input_trajectory)),self.input_size:self.input_size+self.latent_z_dimensionality] = latent_z_copy[:-1]
			
			# We were writing the wrong dimension... should we be running again? :/ 
			assembled_inputs[range(1,len(input_trajectory)),self.input_size+self.latent_z_dimensionality] = latent_b[:-1].float()	
			# assembled_inputs[range(1,len(input_trajectory)),-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

			# Instead of feeding conditional infromation only from 1'st timestep onwards, we are going to st it from the first timestep. 
			if self.conditional_info_size>0:
				assembled_inputs[:,-self.conditional_info_size:] = torch.tensor(conditional_information).to(device).float()

			# Now assemble inputs for subpolicy.
			subpolicy_inputs = torch.zeros((len(input_trajectory),self.input_size+self.latent_z_dimensionality)).to(device)
			subpolicy_inputs[:,:self.input_size] = torch.tensor(input_trajectory).view(len(input_trajectory),self.input_size).to(device).float()
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
		self.encoder_network = ContinuousEncoderNetwork(self.input_size, self.hidden_size, self.latent_z_dimensionality, self.args).to(device)				
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
			diff_val = (1-(latent_z_indices==latent_z_indices.roll(1,0))[1:]).to(device).float()
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
			self.baseline = torch.zeros_like(baseline_target.mean()).to(device).float()
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
		# Also setting particular index in conditional information to 1 for task ID.
		self.conditional_information[-self.number_tasks+self.task_id_for_cond_info] = 1

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
			new_state = torch.tensor(new_state_numpy).to(device).float().view((1,-1))

			# This should be true by default...
			# if self.conditional_viz_env:
			# 	self.set_env_conditional_info()
			self.set_env_conditional_info()
			
		else:
			# Compute next state by adding action to state. 
			new_state = subpolicy_input[t,:self.state_dim]+action_to_execute	

		# return new_subpolicy_input
		return action_to_execute, new_state

	def create_RL_environment_for_rollout(self, environment_name, state=None, task_id=None):

		self.environment = robosuite.make(environment_name)
		self.task_id_for_cond_info = task_id
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
		variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood = self.variational_policy.forward(torch.tensor(old_concatenated_traj).to(device).float(), self.epsilon)

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

	def alternate_rollout_latent_policy(self, counter, i, orig_assembled_inputs, orig_subpolicy_inputs):
		assembled_inputs = orig_assembled_inputs.clone().detach()
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()

		# This version of rollout uses the incremental reparam get actions function. 		
		hidden = None		

		############# (0) #############
		# Get sample we're going to train on. Single sample as of now.
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		# Set rollout length.
		if self.args.traj_length>0:
			self.rollout_timesteps = self.args.traj_length
		else:
			self.rollout_timesteps = len(sample_traj)		

		# For appropriate number of timesteps. 
		for t in range(self.rollout_timesteps-1):

			# First get input row for latent policy. 

			# Feed into latent policy and get z. 

			# Feed z and b into subpolicy. 

			pass

	def rollout_latent_policy(self, orig_assembled_inputs, orig_subpolicy_inputs):
		assembled_inputs = orig_assembled_inputs.clone().detach()
		subpolicy_inputs = orig_subpolicy_inputs.clone().detach()

		# Set the previous b time to 0.
		delta_t = 0

		# For number of rollout timesteps:
		for t in range(self.rollout_timesteps-1):

			##########################################
			#### CODE FOR NEW Z SELECTION ROLLOUT ####
			##########################################

			# Pick latent_z and latent_b. 
			selected_b, new_selected_z = self.latent_policy.get_actions(assembled_inputs[:(t+1)].view((t+1,-1)), greedy=True, delta_t=delta_t)

			if t==0:
				selected_b = torch.ones_like(selected_b).to(device).float()

			if selected_b[-1]==1:
				# Copy over ALL z's. This is okay to do because we're greedily selecting, and hte latent policy is hence deterministic.
				selected_z = torch.tensor(new_selected_z).to(device).float()

				# If b was == 1, then... reset b to 0.
				delta_t = 0
			else:
				# Increment counter since last time b was 1.
				delta_t += 1

			# Set z's to 0. 
			assembled_inputs[t+1, self.input_size:self.input_size+self.number_policies] = 0.
			# Set z and b in assembled input for the future latent policy passes. 
			if self.args.discrete_z:
				assembled_inputs[t+1, self.input_size+selected_z[-1]] = 1.
			else:
				assembled_inputs[t+1, self.input_size:self.input_size+self.latent_z_dimensionality] = selected_z[-1]
			
			# This was also using wrong dimensions... oops :P 
			assembled_inputs[t+1, self.input_size+self.latent_z_dimensionality]	 = selected_b[-1]

			# Before copying over, set conditional_info from the environment at the current timestep.

			if self.conditional_viz_env:
				self.set_env_conditional_info()

			if self.conditional_info_size>0:
				assembled_inputs[t+1, -self.conditional_info_size:] = torch.tensor(self.conditional_information).to(device).float()

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

		if self.args.debug:
			print("Embedding in Latent Policy Rollout.")
			embed()

		# Clear these variables from memory.
		del subpolicy_inputs, assembled_inputs

		return concatenated_selected_b

	def rollout_visuals(self, counter, i, get_image=True):

		# if self.args.data=='Roboturk':
		if self.conditional_viz_env:
			self.create_RL_environment_for_rollout(self.dataset[i]['environment-name'], self.dataset[i]['flat-state'][0], self.dataset[i]['task-id'],)

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
			latent_rollout_image = self.visualize_trajectory(self.latent_trajectory_rollout, segmentations=latent_segmentation, i=i, suffix='Latent')
			variational_rollout_image = self.visualize_trajectory(self.variational_trajectory_rollout, segmentations=variational_segmentation.detach().cpu().numpy(), i=i, suffix='Variational')	

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
			variational_b_probabilities, variational_z_probabilities, kl_divergence, prior_loglikelihood = self.variational_policy.forward(torch.tensor(old_concatenated_traj).to(device).float(), self.epsilon)
			
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

				if self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap':
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

						torch_concat_traj = torch.tensor(concatenated_traj).to(device).float()

						# For each timestep z in latent_z_indices, evaluate likelihood under pretrained encoder model. 
						for t in range(latent_z_indices.shape[0]):
							eval_encoded_logprobs[t] = self.encoder_network.forward(torch_concat_traj, z_sample_to_evaluate=latent_z_indices[t])					
							_, eval_orig_encoder_logprobs[t], _, _ = self.encoder_network.forward(torch_concat_traj)

						print("Encoder Loglikelihood:", eval_encoded_logprobs.detach().cpu().numpy())
						print("Orig Encoder Loglikelihood:", eval_orig_encoder_logprobs.detach().cpu().numpy())
				
				if self.args.debug:
					embed()			

	def evaluate_metrics(self):
		self.distances = -np.ones((self.test_set_size))

		# Get test set elements as last (self.test_set_size) number of elements of dataset.
		for i in range(self.test_set_size):

			index = i + len(self.dataset)-self.test_set_size
			print("Evaluating ", i, " in test set, or ", index, " in dataset.")

			# Collect inputs. 
			sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

			# If valid
			if sample_traj is not None:

				# Create environment to get conditional info.
				if self.conditional_viz_env:
					self.create_RL_environment_for_rollout(self.dataset[i]['environment-name'], self.dataset[i]['flat-state'][0])

				# Rollout variational. 
				_, _, _ = self.rollout_variational_network(0, i)

				self.distances[i] = ((sample_traj-self.variational_trajectory_rollout)**2).mean()	

		self.mean_distance = self.distances[self.distances>0].mean()

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

	def evaluate(self, model):

		self.set_epoch(0)

		if model:
			self.load_all_models(model)

		np.set_printoptions(suppress=True,precision=2)

		print("Running Evaluation of State Distances on small test set.")
		self.evaluate_metrics()

		# Visualize space if the subpolicy has been trained...
		if (self.args.data=='MIME' or self.args.data=='Roboturk' or self.args.data=='OrigRoboturk' or self.args.data=='FullRoboturk' or self.args.data=='Mocap') and (self.args.fix_subpolicy==0):
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

	def __init__(self, number_policies=4, dataset=None, args=None):
	
		# super(PolicyManager_BaselineRL, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		super(PolicyManager_BaselineRL, self).__init__()

		# Create environment, setup things, etc. 
		self.args = args		

		self.initial_epsilon = self.args.epsilon_from
		self.final_epsilon = self.args.epsilon_to
		self.decay_episodes = self.args.epsilon_over
		self.baseline = None
		self. learning_rate = self.args.learning_rate
		self.max_timesteps = 100
		self.gamma = 0.99
		self.batch_size = 10
		self.number_test_episodes = 100

		# Per step decay. 
		self.decay_rate = (self.initial_epsilon-self.final_epsilon)/(self.decay_episodes)
		self.number_episodes = 5000000

		# Orhnstein Ullenhbeck noise process parameters. 
		self.theta = 0.15
		self.sigma = 0.2		

		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])


		self.reset_statistics()

	def create_networks(self):

		if self.args.MLP_policy:
			self.policy_network = ContinuousMLP(self.input_size, self.args.hidden_size, self.output_size, self.args).to(device)
			self.critic_network = CriticMLP(self.input_size, self.args.hidden_size, 1, self.args).to(device)
		else:
			# Create policy and critic. 		
			self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers, small_init=True).to(device)			
			self.critic_network = CriticNetwork(self.input_size, self.args.hidden_size, 1, self.args, self.args.number_layers).to(device)

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

	def load_all_models(self, path, critic=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if critic:
			self.critic_network.load_state_dict(load_object['Critic_Network'])

	def setup(self):
		# Calling a special RL setup function. This is because downstream classes inherit (and may override setup), but will still inherit RL_setup intact.
		self.RL_setup()

	def RL_setup(self):
		# Create Mujoco environment. 
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		# self.input_size = self.state_size + self.output_size		
		self.input_size = self.state_size + self.output_size*2
		
		# Create networks. 
		self.create_networks()
		self.create_training_ops()		
		self.initialize_plots()

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size, min_sigma=self.args.OU_min_sigma, max_sigma=self.args.OU_max_sigma)

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

	def get_action(self, hidden=None, random=True, counter=0, evaluate=False):

		# Change this to epsilon greedy...
		whether_greedy = np.random.binomial(n=1,p=0.8)

		if random or not(whether_greedy):
			action = 2*np.random.random((self.output_size))-1
			return action, hidden	

		# The rest of this will only be evaluated or run when random is false and when whether_greedy is true.
		# Assemble states of current input row.
		current_input_row = self.get_current_input_row()

		# Using the incremental get actions. Still get action greedily, then add noise. 		
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).to(device).float(), greedy=True, hidden=hidden)

		if evaluate:
			noise = torch.zeros_like(predicted_action).to(device).float()
		else:
			# Get noise from noise process. 					
			noise = torch.randn_like(predicted_action).to(device).float()*self.epsilon

		# Perturb action with noise. 			
		perturbed_action = predicted_action + noise

		if self.args.MLP_policy:
			action = perturbed_action[-1].detach().cpu().numpy()
		else:
			action = perturbed_action[-1].squeeze(0).detach().cpu().numpy()		

		return action, hidden

	def get_OU_action(self, hidden=None, random=False, counter=0, evaluate=False):

		if random==True:
			action = 2*np.random.random((self.output_size))-1
			return action, hidden
		
		# Assemble states of current input row.
		current_input_row = self.get_current_input_row()
		# Using the incremental get actions. Still get action greedily, then add noise. 		
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(current_input_row).to(device).float(), greedy=True, hidden=hidden)

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
		# Reset the noise process! We forgot to do this! :( 
		self.NoiseProcess.reset()

		if visualize:			
			image = self.environment.sim.render(600,600, camera_name='frontview')
			self.image_trajectory.append(np.flipud(image))
		
		self.state_trajectory.append(state)
		# self.terminal_trajectory.append(terminal)
		# self.reward_trajectory.append(0.)		

		hidden = None

		while not(terminal) and counter<self.max_timesteps:

			if self.args.OU:
				action, hidden = self.get_OU_action(hidden=hidden,random=random,counter=counter, evaluate=test)
			else:
				action, hidden = self.get_action(hidden=hidden,random=random,counter=counter, evaluate=test)			
				
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

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)			

		finger_diff = gripper_values[1]-gripper_values[0]
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))
		return gripper_value

	def get_current_input_row(self):
		# Addiong joint states, gripper, actions, and conditional info in addition to just conditional and actions.
		gripper_finger_values = self.state_trajectory[-1]['gripper_qpos']
		conditional = np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1))],axis=1)

		if len(self.action_trajectory)>0:				
			state_action = np.concatenate([self.state_trajectory[-1]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[-1].reshape((1,-1))],axis=1)
		else:			
			# state_action = np.concatenate([self.state_trajectory[-1]['robot-state'].reshape((1,-1)),self.state_trajectory[-1]['object-state'].reshape((1,-1)),np.zeros((1,self.output_size))],axis=1)
			state_action = np.concatenate([self.state_trajectory[-1]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), np.zeros((1,self.output_size))],axis=1)
		return np.concatenate([state_action, conditional],axis=1)

	def assemble_inputs(self):
		conditional_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1) for t in range(len(self.state_trajectory))],axis=0)

		state_action_sequence = np.concatenate([np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(self.state_trajectory[t]['gripper_qpos']), self.action_trajectory[t-1].reshape((1,-1))],axis=1) for t in range(1,len(self.state_trajectory))],axis=0)		
		initial_state_action = np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(self.state_trajectory[0]['gripper_qpos']), np.zeros((1, self.output_size))],axis=1)

		# Copy initial state to front of state_action seq. 
		state_action_sequence = np.concatenate([state_action_sequence, initial_state_action],axis=0)

		inputs = np.concatenate([state_action_sequence, conditional_sequence],axis=1)
		
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
		self.state_action_inputs = torch.tensor(assembled_inputs).to(device).float()	

		# Get summed reward for statistics. 
		self.batch_reward_statistics += sum(self.reward_trajectory)

	def set_differentiable_critic_inputs(self):
		# Get policy's predicted actions by getting action greedily, then add noise. 				
		predicted_action = self.policy_network.reparameterized_get_actions(self.state_action_inputs, greedy=True).squeeze(1)
		noise = torch.zeros_like(predicted_action).to(device).float()
		
		# Get noise from noise process. 					
		noise = torch.randn_like(predicted_action).to(device).float()*self.epsilon

		# Concatenate the states from policy inputs and the predicted actions. 
		self.critic_inputs = torch.cat([self.state_action_inputs[:,:self.output_size], predicted_action, self.state_action_inputs[:,2*self.output_size:]],axis=1).to(device).float()

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
			self.critic_targets = torch.tensor(self.TD_targets).to(device).float()
		else:
			self.cummulative_rewards = copy.deepcopy(np.cumsum(np.array(self.reward_trajectory)[::-1])[::-1])
			self.critic_targets = torch.tensor(self.cummulative_rewards).to(device).float()

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

			# Reset the noise process! We forgot to do this! :( 
			self.NoiseProcess.reset()

			print("Initializing Memory Episode: ", episode_counter)
			# Rollout an episode.
			self.rollout(random=self.args.random_memory_burn_in)

			# Add episode to memory.
			self.memory.append_to_memory(self.episode)

			episode_counter += 1			

	def evaluate(self, epoch=None, model=None):

		if model is not None:
			print("Loading model in training.")
			self.load_all_models(model)
			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		else:
			model_epoch = epoch

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

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.total_rewards.mean())

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
			
			# Reset the noise process! We forgot to do this! :( 
			self.NoiseProcess.reset()

			if e%self.args.save_freq==0:
				self.save_all_models("epoch{0}".format(e))

			self.run_iteration(e)
			print("#############################")
			print("Running Episode: ",e)

			if e%self.args.eval_freq==0:
				self.evaluate(epoch=e, model=None)

class PolicyManager_DownstreamRL(PolicyManager_BaselineRL):

	def __init__(self, number_policies=4, dataset=None, args=None):

		super(PolicyManager_DownstreamRL, self).__init__(number_policies=4, dataset=dataset, args=args)

	def setup(self):
		# Create Mujoco environment. 
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.environment.action_spec[0].shape[0]
		self.conditional_info_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]
		# If we are loading policies....
		if self.args.model:
			# Padded conditional info.
			self.conditional_info_size = 53		
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
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.args.hidden_size, self.output_size, self.args, self.args.number_layers).to(device)				
		self.critic_network = CriticNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, 1, self.args, self.args.number_layers).to(device)

		if self.args.constrained_b_prior:
			self.latent_policy = ContinuousLatentPolicyNetwork_ConstrainedBPrior(self.input_size+self.conditional_info_size, self.args.hidden_size, self.args, self.args.number_layers).to(device)
		else:
			self.latent_policy = ContinuousLatentPolicyNetwork(self.input_size+self.conditional_info_size, self.args.hidden_size, self.args, self.args.number_layers).to(device)

	def create_training_ops(self):
		
		self.NLL_Loss = torch.nn.NLLLoss(reduction='none')
		self.MSE_Loss = torch.nn.MSELoss(reduction='none')
		
		# If we are using reparameterization, use a global optimizer for both policies, and a global loss function.
		parameter_list = list(self.latent_policy.parameters())
		if not(self.args.fix_subpolicy):
			parameter_list = parameter_list + list(self.policy_network.parameters())		
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

	def load_all_models(self, path, critic=False):
		load_object = torch.load(path)
		self.policy_network.load_state_dict(load_object['Policy_Network'])
		if self.args.load_latent:
			self.latent_policy.load_state_dict(load_object['Latent_Policy'])
		if critic:
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
		conditional_info_row = np.zeros((1,self.conditional_info_size))
		info_value = np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1)		
		conditional_info_row[0,:info_value.shape[1]] = info_value

		return conditional_info_row

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
			return torch.cat([torch.tensor(self.get_current_input_row(t)).to(device).float(), latent_z.reshape((1,-1))],dim=1)
		else:
			# Remember, get_latent_input_row isn't operating on something that needs to be differentiable, so just use numpy and then wrap with torch tensor. 
			# return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_latent_input_row(t)[:,:-1]],axis=1)).to(device).float()
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.latent_z_trajectory[t].reshape((1,-1))],axis=1)).to(device).float()

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
			return torch.cat([torch.tensor(self.state_trajectory[t]['joint_pos']).to(device).float().reshape((1,-1)), torch.tensor(gripper_value).to(device).float(), action.reshape((1,-1)), torch.tensor(self.get_conditional_information_row(t)).to(device).float()],dim=1)
		else:		
			# Just use actions that were used in the trajectory. This doesn't need to be differentiable, because it's going to be used for the critic targets, so just make a torch tensor from numpy. 
			return torch.tensor(np.concatenate([self.get_current_input_row(t), self.get_conditional_information_row(t)],axis=1)).to(device).float()

	def assemble_state_action_inputs(self, action_list=None):
		# return np.concatenate([self.assemble_state_action_row(t) for t in range(len(self.state_trajectory))],axis=0)
		
		# Instead of numpy use torch.
		if action_list is not None:
			return torch.cat([self.assemble_state_action_row(t=t, action=action_list[t]) for t in range(len(self.state_trajectory))],dim=0)
		else:
			return torch.cat([self.assemble_state_action_row(t=t) for t in range(len(self.state_trajectory))],dim=0)

	def get_OU_action_latents(self, policy_hidden=None, latent_hidden=None, random=False, counter=0, previous_z=None, test=False, delta_t=0):

		# if random==True:
		# 	action = 2*np.random.random((self.output_size))-1
		# 	return action, 

		# Get latent policy inputs.
		latent_policy_inputs = self.assemble_latent_input_row()
		
		# Feed in latent policy inputs and get the latent policy outputs (z, b, and hidden)
		latent_z, latent_b, latent_hidden = self.latent_policy.incremental_reparam_get_actions(torch.tensor(latent_policy_inputs).to(device).float(), greedy=True, hidden=latent_hidden, previous_z=previous_z, delta_t=delta_t)

		# Perturb latent_z with some noise. 
		z_noise = self.epsilon*torch.randn_like(latent_z)
		# Add noise to z.
		latent_z = latent_z + z_noise

		if latent_b[-1]==1:
			delta_t = 0
		else:
			delta_t += 1

		# Now get subpolicy inputs.
		# subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z.detach().cpu().numpy())
		subpolicy_inputs = self.assemble_subpolicy_input_row(latent_z=latent_z)

		# Feed in subpolicy inputs and get the subpolicy outputs (a, hidden)
		predicted_action, hidden = self.policy_network.incremental_reparam_get_actions(torch.tensor(subpolicy_inputs).to(device).float(), greedy=True, hidden=policy_hidden)

		# Numpy action
		action = predicted_action[-1].squeeze(0).detach().cpu().numpy()		
		
		if test:
			perturbed_action = action
		else:	
			# Perturb action with noise. 			
			if self.args.OU:
				perturbed_action = self.NoiseProcess.get_action(action, counter)

			else:
				# Just regular epsilon
				perturbed_action = action + self.epsilon*np.random.randn(action.shape[-1])

		return perturbed_action, latent_z, latent_b, policy_hidden, latent_hidden, delta_t

	def rollout(self, random=False, test=False, visualize=False):
		
		# Reset the noise process! We forgot to do this! :( 
		self.NoiseProcess.reset()

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

		delta_t = 0		

		# For number of steps / while we don't terminate:
		while not(terminal) and counter<self.max_timesteps:

			# Get the action to execute, b, z, and hidden states. 
			action, latent_z, latent_b, policy_hidden, latent_hidden, delta_t = self.get_OU_action_latents(policy_hidden=policy_hidden, latent_hidden=latent_hidden, random=random, counter=counter, previous_z=latent_z, test=test, delta_t=delta_t)

			if self.args.debug:
				print("Embed in Trajectory Rollout.")
				embed()

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
		self.state_action_inputs = torch.tensor(self.assemble_state_action_inputs()).to(device).float()

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
		self.latent_policy_inputs = torch.tensor(self.assemble_latent_inputs()).to(device).float()		

		# 2) Feed this into latent policy. 
		latent_z, latent_b, _ = self.latent_policy.incremental_reparam_get_actions(torch.tensor(self.latent_policy_inputs).to(device).float(), greedy=True)

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

	# Make it inherit joint policy manager init.
	def __init__(self, number_policies=4, dataset=None, args=None):
		super(PolicyManager_DMPBaselines, self).__init__(number_policies, dataset, args)

	def setup_DMP_parameters(self):
		self.output_size 
		self.number_kernels = 15
		self.window = 15
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
		
		if len(segmentation)==0:
			segmentation = np.array([0,len(sample_traj)])
		else:
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

		self.setup_DMP_parameters()
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
				# elif self.args.data=='Mocap':
				# 	sample_traj = sample_traj
					
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
		super(PolicyManager_Imitation, self).__init__(number_policies=number_policies, dataset=dataset, args=args)
		# Explicitly run inits to make sure inheritance is good.
		# PolicyManager_Pretrain.__init__(self, number_policies, dataset, args)
		# PolicyManager_BaselineRL.__init__(self, args)

		# Set train only policy to true.
		self.args.train_only_policy = 1

		# Get task index from task name.
		self.demo_task_index = np.where(np.array(self.dataset.environment_names)==self.args.environment)[0][0]

	def setup(self):
		# Fixing seeds.
		np.random.seed(seed=0)
		torch.manual_seed(0)
		np.set_printoptions(suppress=True,precision=2)

		# Create index list.
		extent = self.dataset.get_number_task_demos(self.demo_task_index)
		self.index_list = np.arange(0,extent)	

		# Create Mujoco environment. 
		self.environment = robosuite.make(self.args.environment, has_renderer=False, use_camera_obs=False, reward_shaping=self.args.shaped_reward)
		
		self.gripper_open = np.array([0.0115, -0.0115])
		self.gripper_closed = np.array([-0.020833, 0.020833])

		# Get input and output sizes from these environments, etc. 
		self.obs = self.environment.reset()		
		self.output_size = self.environment.action_spec[0].shape[0]
		self.state_size = self.obs['robot-state'].shape[0] + self.obs['object-state'].shape[0]

		self.conditional_info_size = self.state_size
		# Input size.. state, action, conditional
		self.input_size = self.state_size + self.output_size*2

		# Create networks. 
		self.create_networks()
		self.create_training_ops()		
		self.initialize_plots()

		self.total_rewards = 0.

		# Create Noise process. 
		self.NoiseProcess = RLUtils.OUNoise(self.output_size)

	def create_networks(self):

		# We don't need a decoder.
		# Policy Network is the only thing we need.
		self.policy_network = ContinuousPolicyNetwork(self.input_size, self.hidden_size, self.output_size, self.args, self.number_layers).to(device)

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

		if counter%self.args.display_freq==0:

			# print("Embedding in Update Plots.")
			
			# Rollout policy.
			self.rollout(random=False, test=True, visualize=True)
			self.tf_logger.gif_summary("Rollout Trajectory", [np.array(self.image_trajectory)], counter)

	def run_iteration(self, counter, i):

		self.set_epoch(counter)	
		self.iter = counter

		############# (0) #############
		# Get sample we're going to train on.		
		sample_traj, sample_action_seq, concatenated_traj, old_concatenated_traj = self.collect_inputs(i)

		if sample_traj is not None:			
			# Now concatenate info with... conditional_information
			policy_inputs = np.concatenate([concatenated_traj, self.conditional_information], axis=1) 	

			# Add zeros to the last action, so that we evaluate likelihood correctly. Since we're using demo actions, no need.
			# padded_action_seq = np.concatenate([sample_action_seq, np.zeros((1,self.output_size))],axis=0)

			# Feed concatenated trajectory into the policy. 
			logprobabilities, _ = self.policy_network.forward(torch.tensor(policy_inputs).to(device).float(), sample_action_seq)

			if self.args.train:
				if self.args.debug:
					if self.iter%self.args.debug==0:
						print("Embedding in Train Function.")
						embed()
				
				# Update policy. 						
				self.update_policies(logprobabilities)

				# Update plots.
				self.update_plots(counter, logprobabilities)

	def get_transformed_gripper_value(self, gripper_finger_values):
		gripper_values = (gripper_finger_values - self.gripper_open)/(self.gripper_closed - self.gripper_open)					
		finger_diff = gripper_values[1]-gripper_values[0]	
		gripper_value = np.array(2*finger_diff-1).reshape((1,-1))

		return gripper_value

	def get_state_action_row(self, t=-1):

		# The state that we want is ... joint state?
		gripper_finger_values = self.state_trajectory[t]['gripper_qpos']

		if len(self.action_trajectory)==0 or t==0:
			return np.concatenate([self.state_trajectory[0]['joint_pos'].reshape((1,-1)), np.zeros((1,1)), np.zeros((1,self.output_size))],axis=1)
		elif t==-1:
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t].reshape((1,-1))],axis=1)
		else: 
			return np.concatenate([self.state_trajectory[t]['joint_pos'].reshape((1,-1)), self.get_transformed_gripper_value(gripper_finger_values), self.action_trajectory[t-1].reshape((1,-1))],axis=1)

	def get_current_input_row(self, t=-1):
		# Rewrite this funciton so that the baselineRL Rollout class can still use it here...
		# First get conditional information.

		# Get robot and object state.
		conditional_info = np.concatenate([self.state_trajectory[t]['robot-state'].reshape((1,-1)),self.state_trajectory[t]['object-state'].reshape((1,-1))],axis=1)		

		# Get state actions..
		state_action = self.get_state_action_row()

		# Concatenate.
		input_row = np.concatenate([state_action, conditional_info],axis=1)

		return input_row

	def evaluate(self, epoch=None, model=None):

		if model is not None:
			self.load_all_models(model)
			model_epoch = int(os.path.split(self.args.model)[1].lstrip("Model_epoch"))
		else:
			model_epoch = epoch

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

		self.dir_name = os.path.join(self.args.logdir,self.args.name,"MEval","m{0}".format(model_epoch))
		if not(os.path.isdir(self.dir_name)):
			os.mkdir(self.dir_name)

		np.save(os.path.join(self.dir_name,"Total_Rewards_{0}.npy".format(self.args.name)),self.total_rewards)
		np.save(os.path.join(self.dir_name,"Mean_Reward_{0}.npy".format(self.args.name)),self.total_rewards.mean())

		# Add average reward to tensorboard.
		self.tf_logger.scalar_summary('Average Reward', self.total_rewards.mean(), model_epoch)

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
				self.evaluate(e)

		self.write_and_close()

class PolicyManager_Transfer(PolicyManager_BaseClass):

	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_Transfer, self).__init__()

		# The inherited functions refer to self.args. Also making this to make inheritance go smooth.
		self.args = args

		# Before instantiating policy managers of source or target domains; create copies of args with data attribute changed. 		
		self.source_args = copy.deepcopy(args)
		self.source_args.data = self.source_args.source_domain
		self.source_dataset = source_dataset

		self.target_args = copy.deepcopy(args)
		self.target_args.data = self.target_args.target_domain
		self.target_dataset = target_dataset

		# Now create two instances of policy managers for each domain. Call them source and target domain policy managers. 
		self.source_manager = PolicyManager_Pretrain(dataset=self.source_dataset, args=self.source_args)
		self.target_manager = PolicyManager_Pretrain(dataset=self.target_dataset, args=self.target_args)		

		self.source_dataset_size = len(self.source_manager.dataset) - self.source_manager.test_set_size
		self.target_dataset_size = len(self.target_manager.dataset) - self.target_manager.test_set_size

		# Now create variables that we need. 
		self.number_epochs = 200
		self.extent = max(self.source_dataset_size, self.target_dataset_size)		

		# Now setup networks for these PolicyManagers. 		
		self.source_manager.setup()
		self.target_manager.setup()

		# Now define other parameters that will be required for the discriminator, etc. 
		self.input_size = self.args.z_dimensions
		self.hidden_size = self.args.hidden_size
		self.output_size = 2
		self.learning_rate = self.args.learning_rate

	def set_iteration(self, counter):

		# Based on what phase of training we are in, set discriminability loss weight, etc. 
		
		# Phase 1 of training: Don't train discriminator at all, set discriminability loss weight to 0.
		if counter<self.args.training_phase_size:
			self.discriminability_loss_weight = 0.
			self.vae_loss_weight = 1.
			self.training_phase = 1
			self.skip_vae = False
			self.skip_discriminator = True

		# Phase 2 of training: Train the discriminator, and set discriminability loss weight to original.
		else:
			self.discriminability_loss_weight = self.args.discriminability_weight
			self.vae_loss_weight = self.args.vae_loss_weight

			# Now make discriminator and vae train in alternating fashion. 
			# Set number of iterations of alteration. 
			# self.alternating_phase_size = self.args.alternating_phase_size*self.extent

			# # If odd epoch, train discriminator. (Just so that we start training discriminator first).
			# if (counter/self.alternating_phase_size)%2==1:			
			# 	self.skip_discriminator = False
			# 	self.skip_vae = True
			# # Otherwise train VAE.
			# else:
			# 	self.skip_discriminator = True
			# 	self.skip_vae = False		

			# Train discriminator for k times as many steps as VAE. Set args.alternating_phase_size as 1 for this. 
			if (counter/self.args.alternating_phase_size)%(self.args.discriminator_phase_size+1)>=1:
				print("Training Discriminator.")
				self.skip_discriminator = False
				self.skip_vae = True
			# Otherwise train VAE.
			else:
				print("Training VAE.")
				self.skip_discriminator = True
				self.skip_vae = False		

			self.training_phase = 2


		self.source_manager.set_epoch(counter)
		self.target_manager.set_epoch(counter)

	def create_networks(self):

		# Call create networks from each of the policy managers. 
		self.source_manager.create_networks()
		self.target_manager.create_networks()

		# Now must also create discriminator.
		self.discriminator_network = DiscreteMLP(self.input_size, self.hidden_size, self.output_size).to(device)

	def create_training_ops(self):

		# # Call create training ops from each of the policy managers. Need these optimizers, because the encoder-decoders get a different loss than the discriminator. 
		self.source_manager.create_training_ops()
		self.target_manager.create_training_ops()

		# Create BCE loss object. 
		# self.BCE_loss = torch.nn.BCELoss(reduction='None')		
		self.negative_log_likelihood_loss_function = torch.nn.NLLLoss(reduction='none')
		
		# Create common optimizer for source, target, and discriminator networks. 
		self.discriminator_optimizer = torch.optim.Adam(self.discriminator_network.parameters(),lr=self.learning_rate)

	def save_all_models(self, suffix):
		self.logdir = os.path.join(self.args.logdir, self.args.name)
		self.savedir = os.path.join(self.logdir,"saved_models")
		if not(os.path.isdir(self.savedir)):
			os.mkdir(self.savedir)
		self.save_object = {}

		# Source
		self.save_object['Source_Policy_Network'] = self.source_manager.policy_network.state_dict()
		self.save_object['Source_Encoder_Network'] = self.source_manager.encoder_network.state_dict()
		# Target
		self.save_object['Target_Policy_Network'] = self.target_manager.policy_network.state_dict()
		self.save_object['Target_Encoder_Network'] = self.target_manager.encoder_network.state_dict()
		# Discriminator
		self.save_object['Discriminator_Network'] = self.discriminator_network.state_dict()				

		torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	def load_all_models(self, path):
		self.load_object = torch.load(path)

		# Source
		self.source_manager.policy_network.load_state_dict(self.load_object['Source_Policy_Network'])
		self.source_manager.encoder_network.load_state_dict(self.load_object['Source_Encoder_Network'])
		# Target
		self.target_manager.policy_network.load_state_dict(self.load_object['Target_Policy_Network'])
		self.target_manager.encoder_network.load_state_dict(self.load_object['Target_Encoder_Network'])
		# Discriminator
		self.discriminator_network.load_state_dict(self.load_object['Discriminator_Network'])

	def get_domain_manager(self, domain):
		# Create a list, and just index into this list. 
		domain_manager_list = [self.source_manager, self.target_manager]
		return domain_manager_list[domain]

	def get_trajectory_segment_tuple(self, source_manager, target_manager):

		# Sample indices. 
		source_index = np.random.randint(0, high=self.source_dataset_size)
		target_index = np.random.randint(0, high=self.target_dataset_size)

		# Get trajectory segments. 
		source_trajectory_segment, source_action_seq, _ = source_manager.get_trajectory_segment(source_manager.index_list[source_index])
		target_trajectory_segment, target_action_seq, _ = target_manager.get_trajectory_segment(target_manager.index_list[target_index])

		return source_trajectory_segment, source_action_seq, target_trajectory_segment, target_action_seq

	def encode_decode_trajectory(self, policy_manager, i, return_trajectory=False, trajectory_input=None):

		# This should basically replicate the encode-decode steps in run_iteration of the Pretrain_PolicyManager. 

		############# (0) #############
		# Sample trajectory segment from dataset. 			

		# Check if the index is too big. If yes, just sample randomly.
		if i >= len(policy_manager.dataset):
			i = np.random.randint(0, len(policy_manager.dataset))

		if trajectory_input is not None: 

			# Grab trajectory segment from tuple. 
			torch_traj_seg = trajectory_input['target_trajectory_rollout']
			
		else: 
			trajectory_segment, sample_action_seq, sample_traj = policy_manager.get_trajectory_segment(i)
			# Torchify trajectory segment.
			torch_traj_seg = torch.tensor(trajectory_segment).to(device).float()

		if trajectory_segment is not None:
			############# (1) #############
			# Encode trajectory segment into latent z. 		
			latent_z, encoder_loglikelihood, encoder_entropy, kl_divergence = policy_manager.encoder_network.forward(torch_traj_seg, policy_manager.epsilon)

			########## (2) & (3) ##########
			# Feed latent z and trajectory segment into policy network and evaluate likelihood. 
			latent_z_seq, latent_b = policy_manager.construct_dummy_latents(latent_z)

			# If we are using the pre-computed trajectory input, (in second encode_decode call, from target trajectory to target latent z.)
			# Don't assemble trajectory in numpy, just take the previous subpolicy_inputs, and then clone it and replace the latent z in it.
			if trajectory_input is not None: 

				# Now assigned trajectory_input['target_subpolicy_inputs'].clone() to SubPolicy_inputs, and then replace the latent z's.
				subpolicy_inputs = trajectory_input['target_subpolicy_inputs'].clone()
				subpolicy_inputs[:,2*self.state_dim:-1] = latent_z_seq

				# Now get "sample_action_seq" for forward function. 
				sample_action_seq = subpolicy_inputs[:,self.state_dim:2*self.state_dim].clone()

			else:
				_, subpolicy_inputs, sample_action_seq = policy_manager.assemble_inputs(trajectory_segment, latent_z_seq, latent_b, sample_action_seq)

			# Policy net doesn't use the decay epislon. (Because we never sample from it in training, only rollouts.)
			loglikelihoods, _ = policy_manager.policy_network.forward(subpolicy_inputs, sample_action_seq)
			loglikelihood = loglikelihoods[:-1].mean()

			if return_trajectory:
				return sample_traj, latent_z
			else:
				return subpolicy_inputs, latent_z, loglikelihood, kl_divergence

		if return_trajectory:
			return None, None
		else:
			return None, None, None, None

	def update_plots(self, counter, viz_dict):

		# VAE Losses. 
		self.tf_logger.scalar_summary('Policy LogLikelihood', self.likelihood_loss, counter)
		self.tf_logger.scalar_summary('Discriminability Loss', self.discriminability_loss, counter)
		self.tf_logger.scalar_summary('Encoder KL', self.encoder_KL, counter)
		self.tf_logger.scalar_summary('VAE Loss', self.VAE_loss, counter)
		self.tf_logger.scalar_summary('Total VAE Loss', self.total_VAE_loss, counter)
		self.tf_logger.scalar_summary('Domain', viz_dict['domain'], counter)

		# Plot discriminator values after we've started training it. 
		if self.training_phase>1:
			# Discriminator Loss. 
			self.tf_logger.scalar_summary('Discriminator Loss', self.discriminator_loss, counter)
			# Compute discriminator prob of right action for logging. 
			self.tf_logger.scalar_summary('Discriminator Probability', viz_dict['discriminator_probs'], counter)
		
		# If we are displaying things: 
		if counter%self.args.display_freq==0:

			self.gt_gif_list = []
			self.rollout_gif_list = []

			# Now using both TSNE and PCA. 
			# Plot source, target, and shared embeddings via TSNE.
			tsne_source_embedding, tsne_target_embedding, tsne_combined_embeddings, tsne_combined_traj_embeddings = self.get_embeddings(projection='tsne')

			# Now actually plot the images.			
			self.tf_logger.image_summary("TSNE Source Embedding", [tsne_source_embedding], counter)
			self.tf_logger.image_summary("TSNE Target Embedding", [tsne_target_embedding], counter)
			self.tf_logger.image_summary("TSNE Combined Embeddings", [tsne_combined_embeddings], counter)			

			# Plot source, target, and shared embeddings via PCA. 
			pca_source_embedding, pca_target_embedding, pca_combined_embeddings, pca_combined_traj_embeddings = self.get_embeddings(projection='pca')

			# Now actually plot the images.			
			self.tf_logger.image_summary("PCA Source Embedding", [pca_source_embedding], counter)
			self.tf_logger.image_summary("PCA Target Embedding", [pca_target_embedding], counter)
			self.tf_logger.image_summary("PCA Combined Embeddings", [pca_combined_embeddings], counter)			

			if self.args.source_domain=='ContinuousNonZero' and self.args.target_domain=='ContinuousNonZero':
				self.tf_logger.image_summary("PCA Combined Trajectory Embeddings", [pca_combined_traj_embeddings], counter)
				self.tf_logger.image_summary("TSNE Combined Trajectory Embeddings", [tsne_combined_traj_embeddings], counter)

			# We are also going to log Ground Truth trajectories and their reconstructions in each of the domains, to make sure our networks are learning. 		
			# Should be able to use the policy manager's functions to do this.
			source_trajectory, source_reconstruction, target_trajectory, target_reconstruction = self.get_trajectory_visuals()

			if source_trajectory is not None:
				# Now actually plot the images.

				if self.args.source_domain=='ContinuousNonZero':
					self.tf_logger.image_summary("Source Trajectory", [source_trajectory], counter)
					self.tf_logger.image_summary("Source Reconstruction", [source_reconstruction], counter)
				else:
					self.tf_logger.gif_summary("Source Trajectory", [source_trajectory], counter)
					self.tf_logger.gif_summary("Source Reconstruction", [source_reconstruction], counter)

				if self.args.target_domain=='ContinuousNonZero':
					self.tf_logger.image_summary("Target Trajectory", [target_trajectory], counter)
					self.tf_logger.image_summary("Target Reconstruction", [target_reconstruction], counter)
				else:
					self.tf_logger.gif_summary("Target Trajectory", [target_trajectory], counter)
					self.tf_logger.gif_summary("Target Reconstruction", [target_reconstruction], counter)

			if self.args.source_domain=='ContinuousNonZero' and self.args.target_domain=='ContinuousNonZero':
				# Evaluate metrics and plot them. 
				# self.evaluate_correspondence_metrics(computed_sets=False)
				# Actually, we've probably computed trajectory and latent sets. 
				self.evaluate_correspondence_metrics()

				self.tf_logger.scalar_summary('Source To Target Trajectory Distance', self.source_target_trajectory_distance, counter)		
				self.tf_logger.scalar_summary('Target To Source Trajectory Distance', self.target_source_trajectory_distance, counter)

	def get_transform(self, latent_z_set, projection='tsne', shared=False):

		if shared:
			# If this set of z's contains z's from both source and target domains, mean-std normalize them independently. 
			normed_z = np.zeros_like(latent_z_set)
			# Normalize source.
			source_mean = latent_z_set[:self.N].mean(axis=0)
			source_std = latent_z_set[:self.N].std(axis=0)
			normed_z[:self.N] = (latent_z_set[:self.N]-source_mean)/source_std
			# Normalize target.
			target_mean = latent_z_set[self.N:].mean(axis=0)
			target_std = latent_z_set[self.N:].std(axis=0)
			normed_z[self.N:] = (latent_z_set[self.N:]-target_mean)/target_std			

		else:
			# Just normalize z's.
			mean = latent_z_set.mean(axis=0)
			std = latent_z_set.std(axis=0)
			normed_z = (latent_z_set-mean)/std
		
		if projection=='tsne':
			# Use TSNE to project the data:
			tsne = skl_manifold.TSNE(n_components=2,random_state=0)
			embedded_zs = tsne.fit_transform(normed_z)

			scale_factor = 1
			scaled_embedded_zs = scale_factor*embedded_zs

			return scaled_embedded_zs, tsne

		elif projection=='pca':
			# Use PCA to project the data:
			pca_object = PCA(n_components=2)
			embedded_zs = pca_object.fit_transform(normed_z)

			return embedded_zs, pca_object

	def transform_zs(self, latent_z_set, transforming_object):
		# Simply just transform according to a fit transforming_object.
		return transforming_object.transform(latent_z_set)

	# @profile
	def get_embeddings(self, projection='tsne'):
		# Function to visualize source, target, and combined embeddings: 

		self.N = 100
		self.source_latent_zs = np.zeros((self.N,self.args.z_dimensions))
		self.target_latent_zs = np.zeros((self.N,self.args.z_dimensions))
		self.shared_latent_zs = np.zeros((2*self.N,self.args.z_dimensions))

		# For N data points:
		for i in range(self.N):

			# Get corresponding latent z's of source and target domains.
			_, source_z, _, _ = self.encode_decode_trajectory(self.source_manager, i)
			_, target_z, _, _ = self.encode_decode_trajectory(self.target_manager, i)

			if source_z is not None:
				self.source_latent_zs[i] = source_z.detach().cpu().numpy()
				self.shared_latent_zs[i] = source_z.detach().cpu().numpy()
			if target_z is not None:
				self.target_latent_zs[i] = target_z.detach().cpu().numpy()
				self.shared_latent_zs[self.N+i] = target_z.detach().cpu().numpy()

		if projection=='tsne':
			# Use TSNE to transform data.		
			source_embedded_zs, _ = self.get_transform(self.source_latent_zs, projection)
			target_embedded_zs, _ = self.get_transform(self.target_latent_zs, projection)
			shared_embedded_zs, _ = self.get_transform(self.shared_latent_zs, projection, shared=True)		

		elif projection=='pca':
			# Now fit PCA to source.
			source_embedded_zs, pca = self.get_transform(self.source_latent_zs, projection)
			target_embedded_zs = self.transform_zs(self.target_latent_zs, pca)
			shared_embedded_zs = np.concatenate([source_embedded_zs, target_embedded_zs],axis=0)

		source_image = self.plot_embedding(source_embedded_zs, "Source_Embedding")
		target_image = self.plot_embedding(target_embedded_zs, "Target_Embedding")
		shared_image = self.plot_embedding(shared_embedded_zs, "Shared_Embedding", shared=True)	

		toy_shared_embedding_image = None
		if self.args.source_domain=='ContinuousNonZero' and self.args.target_domain=='ContinuousNonZero':			
			toy_shared_embedding_image = self.plot_embedding(shared_embedded_zs, "Toy_Shared_Traj_Embedding", shared=True, trajectory=True)

		return source_image, target_image, shared_image, toy_shared_embedding_image

	# @profile
	def plot_embedding(self, embedded_zs, title, shared=False, trajectory=False):
	
		fig = plt.figure()
		ax = fig.gca()
		
		if shared:
			colors = 0.2*np.ones((2*self.N))
			colors[self.N:] = 0.8
		else:
			colors = 0.2*np.ones((self.N))

		if trajectory:
			# Create a scatter plot of the embedding.

			self.source_manager.get_trajectory_and_latent_sets()
			self.target_manager.get_trajectory_and_latent_sets()

			ratio = 0.4
			color_scaling = 15

			# Assemble shared trajectory set. 
			traj_length = len(self.source_manager.trajectory_set[0,:,0])
			self.shared_trajectory_set = np.zeros((2*self.N, traj_length, 2))
			
			self.shared_trajectory_set[:self.N] = self.source_manager.trajectory_set
			self.shared_trajectory_set[self.N:] = self.target_manager.trajectory_set
			
			color_range_min = 0.2*color_scaling
			color_range_max = 0.8*color_scaling+traj_length-1

			for i in range(2*self.N):
				ax.scatter(embedded_zs[i,0]+ratio*self.shared_trajectory_set[i,:,0],embedded_zs[i,1]+ratio*self.shared_trajectory_set[i,:,1],c=colors[i]*color_scaling+range(traj_length),cmap='jet',vmin=color_range_min,vmax=color_range_max)

		else:
			# Create a scatter plot of the embedding.
			ax.scatter(embedded_zs[:,0],embedded_zs[:,1],c=colors,vmin=0,vmax=1,cmap='jet')
		
		# Title. 
		ax.set_title("{0}".format(title),fontdict={'fontsize':40})
		fig.canvas.draw()
		# Grab image.
		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image

	def get_trajectory_visuals(self):

		i = np.random.randint(0,high=self.extent)

		# First get a trajectory, starting point, and latent z.
		source_trajectory, source_latent_z = self.encode_decode_trajectory(self.source_manager, i, return_trajectory=True)

		if source_trajectory is not None:
			# Reconstruct using the source domain manager. 
			_, source_trajectory_image, source_reconstruction_image = self.source_manager.get_robot_visuals(0, source_latent_z, source_trajectory, return_image=True)		

			# Now repeat the same for target domain - First get a trajectory, starting point, and latent z.
			target_trajectory, target_latent_z = self.encode_decode_trajectory(self.target_manager, i, return_trajectory=True)
			# Reconstruct using the target domain manager. 
			_, target_trajectory_image, target_reconstruction_image = self.target_manager.get_robot_visuals(0, target_latent_z, target_trajectory, return_image=True)		

			return np.array(source_trajectory_image), np.array(source_reconstruction_image), np.array(target_trajectory_image), np.array(target_reconstruction_image)
			
		else: 
			return None, None, None, None

	def update_networks(self, domain, policy_manager, policy_loglikelihood, encoder_KL, discriminator_loglikelihood, latent_z):

		#######################
		# Update VAE portion. 
		#######################

		# Zero out gradients of encoder and decoder (policy).
		policy_manager.optimizer.zero_grad()		

		# Compute VAE loss on the current domain as likelihood plus weighted KL.  
		self.likelihood_loss = -policy_loglikelihood.mean()
		self.encoder_KL = encoder_KL.mean()
		self.VAE_loss = self.likelihood_loss + self.args.kl_weight*self.encoder_KL

		# Compute discriminability loss for encoder (implicitly ignores decoder).
		# Pretend the label was the opposite of what it is, and train the encoder to make the discriminator think this was what was true. 
		# I.e. train encoder to make discriminator maximize likelihood of wrong label.

		self.discriminability_loss = self.negative_log_likelihood_loss_function(discriminator_loglikelihood.squeeze(1), torch.tensor(1-domain).to(device).long().view(1,))

		# Total encoder loss: 
		self.total_VAE_loss = self.vae_loss_weight*self.VAE_loss + self.discriminability_loss_weight*self.discriminability_loss	

		if not(self.skip_vae):
			# Go backward through the generator (encoder / decoder), and take a step. 
			self.total_VAE_loss.backward()
			policy_manager.optimizer.step()

		#######################
		# Update Discriminator. 
		#######################

		# Zero gradients of discriminator.
		self.discriminator_optimizer.zero_grad()

		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		discriminator_logprob, discriminator_prob = self.discriminator_network(latent_z.detach())

		# Compute discriminator loss for discriminator. 
		self.discriminator_loss = self.negative_log_likelihood_loss_function(discriminator_logprob.squeeze(1), torch.tensor(domain).to(device).long().view(1,))		
		
		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.discriminator_loss.backward()
			self.discriminator_optimizer.step()

	# @profile
	def run_iteration(self, counter, i):

		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and discriminator concurrently. 

		# Algorithm: 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use (source or target, i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		# 		# 3) Encode trajectory segments into latent z's and compute likelihood of trajectory actions under the decoder.
		# 		# 4) Feed into discriminator, get likelihood of each domain.
		# 		# 5) Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 

		# (0) Setup things like training phases, epislon values, etc.
		self.set_iteration(counter)

		# (1) Select which domain to run on. This is supervision of discriminator.
		domain = np.random.binomial(1,0.5)

		# (1.5) Get domain policy manager. 
		policy_manager = self.get_domain_manager(domain)
		
		# (2) & (3) Get trajectory segment and encode and decode. 
		subpolicy_inputs, latent_z, loglikelihood, kl_divergence = self.encode_decode_trajectory(policy_manager, i)

		if latent_z is not None:
			# (4) Feed latent z's to discriminator, and get discriminator likelihoods. 
			discriminator_logprob, discriminator_prob = self.discriminator_network(latent_z)

			# (5) Compute and apply gradient updates. 
			self.update_networks(domain, policy_manager, loglikelihood, kl_divergence, discriminator_logprob, latent_z)

			# Now update Plots. 
			viz_dict = {'domain': domain, 'discriminator_probs': discriminator_prob.squeeze(0).squeeze(0)[domain].detach().cpu().numpy()}			
			self.update_plots(counter, viz_dict)

	# Run memory profiling.
	# @profile 

	def set_neighbor_objects(self, computed_sets=False):
		if not(computed_sets):
			self.source_manager.get_trajectory_and_latent_sets()
			self.target_manager.get_trajectory_and_latent_sets()

		# Compute nearest neighbors for each set. First build KD-Trees / Ball-Trees. 
		self.source_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.source_manager.latent_z_set)
		self.target_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.target_manager.latent_z_set)

		self.neighbor_obj_set = True

	def evaluate_correspondence_metrics(self, computed_sets=True):

		print("Evaluating correspondence metrics.")
		# Evaluate the correspondence and alignment metrics. 
		# Whether latent_z_sets and trajectory_sets are already computed for each manager.
		self.set_neighbor_objects(computed_sets)

		# if not(computed_sets):
		# 	self.source_manager.get_trajectory_and_latent_sets()
		# 	self.target_manager.get_trajectory_and_latent_sets()

		# # Compute nearest neighbors for each set. First build KD-Trees / Ball-Trees. 
		# self.source_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.source_manager.latent_z_set)
		# self.target_neighbors_object = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(self.target_manager.latent_z_set)

		# Compute neighbors. 
		_, source_target_neighbors = self.source_neighbors_object.kneighbors(self.target_manager.latent_z_set)
		_, target_source_neighbors = self.target_neighbors_object.kneighbors(self.source_manager.latent_z_set)

		# # Now compute trajectory distances for neighbors. 
		# source_target_trajectory_diffs = (self.source_manager.trajectory_set - self.target_manager.trajectory_set[source_target_neighbors.squeeze(1)])
		# self.source_target_trajectory_distance = copy.deepcopy(np.linalg.norm(source_target_trajectory_diffs,axis=(1,2)).mean())

		# target_source_trajectory_diffs = (self.target_manager.trajectory_set - self.source_manager.trajectory_set[target_source_neighbors.squeeze(1)])
		# self.target_source_trajectory_distance = copy.deepcopy(np.linalg.norm(target_source_trajectory_diffs,axis=(1,2)).mean())

		# Remember, absolute trajectory differences is meaningless, since the data is randomly initialized across the state space. 
		# Instead, compare actions. I.e. first compute differences along the time dimension. 
		source_traj_actions = np.diff(self.source_manager.trajectory_set,axis=1)
		target_traj_actions = np.diff(self.target_manager.trajectory_set,axis=1)

		source_target_trajectory_diffs = (source_traj_actions - target_traj_actions[source_target_neighbors.squeeze(1)])
		self.source_target_trajectory_distance = copy.deepcopy(np.linalg.norm(source_target_trajectory_diffs,axis=(1,2)).mean())

		target_source_trajectory_diffs = (target_traj_actions - source_traj_actions[target_source_neighbors.squeeze(1)])
		self.target_source_trajectory_distance = copy.deepcopy(np.linalg.norm(target_source_trajectory_diffs,axis=(1,2)).mean())

		# Reset variables to prevent memory leaks.
		# source_neighbors_object = None
		# target_neighbors_object = None
		del self.source_neighbors_object
		del self.target_neighbors_object

	def evaluate(self, model=None):

		# Evaluating Transfer - we just want embeddings of both source and target; so run evaluate of both source and target policy managers. 		

		# Instead of parsing and passing model to individual source and target policy managers, just load using the transfer policy manager, and then run eval. 
		if model is not None: 
			self.load_all_models(model)

		# Run source policy manager evaluate. 
		self.source_manager.evaluate(suffix="Source")

		# Run target policy manager evaluate. 
		self.target_manager.evaluate(suffix="Target")

		# Evaluate metrics. 
		self.evaluate_correspondence_metrics()

	def automatic_evaluation(self, e):

		pass

# Writing a cycle consistency transfer PM class.
class PolicyManager_CycleConsistencyTransfer(PolicyManager_Transfer):

	# Inherit from transfer. 
	def __init__(self, args=None, source_dataset=None, target_dataset=None):

		super(PolicyManager_CycleConsistencyTransfer, self).__init__(args, source_dataset, target_dataset)

		self.neighbor_obj_set = False

	# Don't actually need to define these functions since they perform same steps as super functions.
	# def create_networks(self):

	# 	super().create_networks()

	# 	# Must also create two discriminator networks; one for source --> target --> source, one for target --> source --> target. 
	# 	# Remember, since these discriminator networks are operating on the trajectory space, we have to 
	# 	# make them LSTM networks, rather than MLPs. 

	# 	# # We have the encoder network class that's perfect for this. Output size is 2. 
	# 	# self.source_discriminator = EncoderNetwork(self.source_manager.input_size, self.hidden_size, self.output_size).to(device)
	# 	# self.target_discriminator = EncoderNetwork(self.source_manager.input_size, self.hidden_size, self.output_size).to(device)

	def create_training_ops(self):

		# Call super training ops. 
		super().create_training_ops()

		# # Now create discriminator optimizers. 
		# self.source_discriminator_optimizer = torch.optim.Adam(self.source_discriminator_network.parameters(),lr=self.learning_rate)
		# self.target_discriminator_optimizer = torch.optim.Adam(self.target_discriminator_network.parameters(),lr=self.learning_rate)

		# Instead of using the individuals policy manager optimizers, use one single optimizer. 
		self.parameter_list = self.source_manager.parameter_list + self.target_manager.parameter_list
		self.optimizer = torch.optim.Adam(self.parameter_list, lr=self.learning_rate)

	# def save_all_models(self, suffix):

	# 	# Call super save model. 
	# 	super().save_all_models(suffix)

	# 	# Now save the individual source / target discriminators. 
	# 	self.save_object['Source_Discriminator_Network'] = self.source_discriminator_network.state_dict()
	# 	self.save_object['Target_Discriminator_Network'] = self.target_discriminator_network.state_dict()

	# 	# Overwrite the save from super. 
	# 	torch.save(self.save_object,os.path.join(self.savedir,"Model_"+suffix))

	# def load_all_models(self, path):

	# 	# Call super load. 
	# 	super().load_all_models(path)

	# 	# Now load the individual source and target discriminators. 
	# 	self.source_discriminator.load_state_dict(self.load_object['Source_Discriminator_Network'])
	# 	self.target_discriminator.load_state_dict(self.load_object['Target_Discriminator_Network'])

	# A bunch of functions should just be directly usable:
	# get_domain_manager, get_trajectory_segment_tuple, encode_decode_trajectory, update_plots, get_transform, 
	# transform_zs, get_embeddings, plot_embeddings, get_trajectory_visuals, evaluate_correspondence_metrics, 
	# evaluate, automatic_evaluation

	def get_start_state(self, domain, source_latent_z):

		# Function to retrieve the start state for differentiable decoding from target domain. 
		# How we do this is first to retrieve the target domain latent z closest to the source_latent_z. 
		# We then select the trajectory corresponding to this target_domain latent_z.
		# We then copy the start state of this trajectory. 

		if not(self.neighbor_obj_set):
			self.set_neighbor_objects()

		# First get neighbor object and trajectory sets. 
		neighbor_object_list = [self.source_neighbors_object, self.target_neighbors_object]
		trajectory_set_list = [self.source_manager.trajectory_set, self.target_manager.trajectory_set]
		
		# Remember, we need _target_ domain. So use 1-domain instead of domain.
		neighbor_object = neighbor_object_list[1-domain]
		trajectory_set = trajectory_set_list[1-domain]

		# Next get closest target z. 
		_ , target_latent_z_index = neighbor_object.kneighbors(source_latent_z)

		# Don't actually need the target_latent_z, unless we're doing differentiable nearest neighbor transfer. 
		# Now get the corresponding trajectory. 
		trajectory = trajectory_set[target_latent_z_index]

		# Finally, pick up first state. 
		start_state = trajectory[0]

		return start_state

	def differentiable_rollout(self, trajectory_start, latent_z, rollout_length=None):
		# Copying over from rollout_robot_trajectory. This function should provide rollout template, but may need modifications for differentiability. 

		# Remember, the differentiable rollout is required because the backtranslation / cycle-consistency loss needs to be propagated through multiple sets of translations. 
		# Therefore it must pass through the decoder network(s), and through the latent_z's. (It doesn't actually pass through the states / actions?).		

		subpolicy_inputs = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
		subpolicy_inputs[0,:self.state_dim] = torch.tensor(trajectory_start).to(device).float()
		subpolicy_inputs[:,2*self.state_dim:] = torch.tensor(latent_z).to(device).float()	

		if rollout_length is not None: 
			length = rollout_length-1
		else:
			length = self.rollout_timesteps-1

		for t in range(length):

			# Get actions from the policy.
			actions = self.policy_network.reparameterized_get_actions(subpolicy_inputs, greedy=True)

			# Select last action to execute. 
			action_to_execute = actions[-1].squeeze(1)

			# Downscale the actions by action_scale_factor.
			action_to_execute = action_to_execute/self.args.action_scale_factor

			# Compute next state. 
			new_state = subpolicy_inputs[t,:self.state_dim]+action_to_execute

			# New input row. 
			input_row = torch.zeros((1,2*self.state_dim+self.latent_z_dimensionality)).to(device).float()
			input_row[0,:self.state_dim] = new_state
			# Feed in the ORIGINAL prediction from the network as input. Not the downscaled thing. 
			input_row[0,self.state_dim:2*self.state_dim] = actions[-1].squeeze(1)
			input_row[0,2*self.state_dim:] = latent_z

			# Now that we have assembled the new input row, concatenate it along temporal dimension with previous inputs. 
			subpolicy_inputs = torch.cat([subpolicy_inputs,input_row],dim=0)

		trajectory = subpolicy_inputs[:,:self.state_dim].detach().cpu().numpy()
		differentiable_trajectory = subpolicy_inputs[:,:self.state_dim]
		differentiable_action_seq = subpolicy_inputs[:,self.state_dim:2*self.state_dim]
		differentiable_state_action_seq = subpolicy_inputs[:,:2*self.state_dim]

		# return trajectory

		# For differentiabiity, return tuple of trajectory, actions, state actions, and subpolicy_inputs. 
		return [differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs]

	def get_source_target_domain_managers(self):

		domain = np.random.binomial(1,0.5)
		# Also Get domain policy manager. 
		source_policy_manager = self.get_domain_manager(domain) 
		target_policy_manager = self.get_domain_manager(1-domain) 

		return domain, source_policy_manager, target_policy_manager

	def cross_domain_decoding(self, domain, domain_manager, latent_z, start_state=None):

		# If start state is none, first get start state, else use the argument. 
		if start_state is None: 
			start_state = self.get_start_state(domain, latent_z)

		# Now rollout in target domain.
		differentiable_trajectory, differentiable_action_seq, differentiable_state_action_seq, subpolicy_inputs = self.differentiable_rollout(start_state, latent_z)

		return differentiable_trajectory, subpolicy_inputs

	def update_networks(self, dictionary, source_policy_manager):

		# Here are the objectives we have to be considering. 
		# 	1) Reconstruction of inputs under single domain encoding / decoding. 
		#		In this implementation, we just have to use the source_loglikelihood for this. 
		#	2) Discriminability of Z space. This is taken care of from the compute_discriminator_losses function.
		# 	3) Cycle-consistency. This may be implemented as regression (L2), loglikelihood of cycle-reconstructed traj, or discriminability of trajectories.
		#		In this implementation, we just have to use the cross domain decoded loglikelihood. 

		####################################
		# First update encoder decoder networks. Don't train discriminator.
		####################################

		# Zero gradients.
		self.optimizer.zero_grad()		

		####################################
		# (1) Compute single-domain reconstruction loss.
		####################################

		# Compute VAE loss on the current domain as negative log likelihood likelihood plus weighted KL.  
		self.source_likelihood_loss = -dictionary['source_loglikelihood'].mean()
		self.source_encoder_KL = dictionary['source_kl_divergence'].mean()
		self.source_reconstruction_loss = self.source_likelihood_loss + self.args.kl_weight*self.source_encoder_KL

		####################################
		# (2) Compute discriminability losses.
		####################################

		#	This block first computes discriminability losses:
		#	# a) First, feeds the latent_z into the z_discriminator, that is being trained to discriminate between z's of source and target domains. 
		#	# 	 Gets and returns the loglikelihood of the discriminator predicting the true domain. 
		#	# 	 Also returns discriminability loss, that is used to train the _encoders_ of both domains. 
		#	#		
		#	# b) ####### DON'T NEED TO DO THIS YET: ####### Also feeds either the cycle reconstructed trajectory, or the original trajectory from the source domain, into a separate discriminator. 
		#	# 	 This second discriminator is specific to the domain we are operating in. This discriminator is discriminating between the reconstructed and original trajectories. 
		#	# 	 Basically standard GAN adversarial training, except the generative model here is the entire cycle-consistency translation model.
		#
		#	In addition to this, must also compute discriminator losses to train discriminators themselves. 
		# 	# a) For the z discriminator (and if we're using trajectory discriminators, those too), clone and detach the inputs of the discriminator and compute a discriminator loss with the right domain used in targets / supervision. 
		#	#	 This discriminator loss is what is used to actually train the discriminators.		

		# Get z discriminator logprobabilities.
		z_discriminator_logprob, z_discriminator_prob = self.discriminator_network(dictionary['source_latent_z'])
		# Compute discriminability loss. Remember, this is not used for training the discriminator, but rather the encoders.
		self.z_discriminability_loss = self.negative_log_likelihood_loss_function(z_discriminator_logprob.squeeze(1), torch.tensor(1-domain).to(device).long().view(1,))

		###### Block that computes discriminability losses assuming we are using trjaectory discriminators. ######

		# # Get the right trajectory discriminator network.
		# discriminator_list = [self.source_discriminator, self.target_discriminator]		
		# source_discriminator = discriminator_list[domain]

		# # Now feed trajectory to the trajectory discriminator, based on whether it is the source of target discriminator.
		# traj_discriminator_logprob, traj_discriminator_prob = source_discriminator(trajectory)

		# # Compute trajectory discriminability loss, based on whether the trajectory was original or reconstructed.
		# self.traj_discriminability_loss = self.negative_log_likelihood_loss_function(traj_discriminator_logprob.squeeze(1), torch.tensor(1-original_or_reconstructed).to(device).long().view(1,))

		####################################
		# (3) Compute cycle-consistency losses.
		####################################

		# Must compute likelihoods of original actions under the cycle reconstructed trajectory states. 
		# I.e. evaluate likelihood of original actions under source_decoder (i.e. source subpolicy), with the subpolicy inputs constructed from cycle-reconstruction.
		
		# Get the original action sequence.
		original_action_sequence = dictionary['source_subpolicy_inputs_original'][:,self.state_dim:2*self.state_dim]

		# Now evaluate likelihood of actions under the source decoder.
		cycle_reconstructed_loglikelihood, _ = source_policy_manager.forward(dictionary['source_subpolicy_inputs_crossdomain'], original_action_sequence)
		# Reweight the cycle reconstructed likelihood to construct the loss.
		self.cycle_reconstruction_loss = -self.args.cycle_reconstruction_loss_weight*cycle_reconstruction_loss.mean()

		####################################
		# Now that individual losses are computed, compute total loss, compute gradients, and then step.
		####################################

		# First combine losses.
		self.total_VAE_loss = self.source_reconstruction_loss + self.z_discriminability_loss + self.cycle_reconstruction_loss

		# If we are in a encoder / decoder training phase, compute gradients and step.  
		if not(self.skip_vae):
			self.total_VAE_loss.backward()
			self.optimizer.step()

		####################################
		# Now compute discriminator losses and update discriminator network(s).
		####################################

		# First zero out the discriminator gradients. 
		self.discriminator_optimizer.zero_grad()

		# Detach the latent z that is fed to the discriminator, and then compute discriminator loss.
		# If we tried to zero grad the discriminator and then use NLL loss on it again, Pytorch would cry about going backward through a part of the graph that we already \ 
		# went backward through. Instead, just pass things through the discriminator again, but this time detaching latent_z. 
		z_discriminator_detach_logprob, z_discriminator_detach_prob = self.discriminator_network(dictionary['source_latent_z'].detach())

		# Compute discriminator loss for discriminator. 
		self.z_discriminator_loss = self.negative_log_likelihood_loss_function(z_discriminator_detach_logprob.squeeze(1), torch.tensor(domain).to(device).long().view(1,))		
		
		if not(self.skip_discriminator):
			# Now go backward and take a step.
			self.z_discriminator_loss.backward()
			self.discriminator_optimizer.step()

	def run_iteration(self, counter, i):

		# Phases: 
		# Phase 1:  Train encoder-decoder for both domains initially, so that discriminator is not fed garbage. 
		# Phase 2:  Train encoder, decoder for each domain, and Z discriminator concurrently. 
		# Phase 3:  Train encoder, decoder for each domain, and the individual source and target discriminators, concurrently.

		# Algorithm (joint training): 
		# For every epoch:
		# 	# For every datapoint: 
		# 		# 1) Select which domain to use as source (i.e. with 50% chance, select either domain).
		# 		# 2) Get trajectory segments from desired domain. 
		#		# 3) Transfer Steps: 
		#	 		# a) Encode trajectory as latent z (domain 1). 
		#			# b) Use domain 2 decoder to decode latent z into trajectory (domain 2).
		#			# c) Use domain 2 encoder to encode trajectory into latent z (domain 2).
		#			# d) Use domain 1 decoder to decode latent z (domain 2) into trajectory (domain 1).
		# 		# 4) Feed cycle-reconstructed trajectory and original trajectory (both domain 1) into discriminator. 
		#		# 5) Train discriminators to predict whether original or cycle reconstructed trajectory. 
		#		# 	 Alternate: Remember, don't actually need to use trajectory level discriminator networks, can just use loglikelihood cycle-reconstruction loss. Try this first.
		#		# 	 Train z discriminator to predict which domain the latentz sample came from. 
		# 		# 	 Train encoder / decoder architectures with mix of reconstruction loss and discriminator confusing objective. 
		# 		# 	 Compute and apply gradient updates. 

		# Remember to make domain agnostic function calls to encode, feed into discriminator, get likelihoods, etc. 

		####################################
		# (0) Setup things like training phases, epislon values, etc.
		####################################

		self.set_iteration(counter)
		dictionary = {}
		target_dict = {}

		####################################
		# (1) Select which domain to use as source domain (also supervision of z discriminator for this iteration). 
		####################################

		domain, source_policy_manager, target_policy_manager = self.get_source_target_domain_managers()

		####################################
		# (2) & (3 a) Get source trajectory (segment) and encode into latent z. Decode using source decoder, to get loglikelihood for reconstruction objectve. 
		####################################

		dictionary['source_subpolicy_inputs_original'], dictionary['source_latent_z'], dictionary['source_loglikelihood'], dictionary['source_kl_divergence'] = self.encode_decode_trajectory(source_policy_manager, i)

		####################################
		# (3 b) Cross domain decoding. 
		####################################
		
		target_dict['target_trajectory_rollout'], target_dict['target_subpolicy_inputs'] = self.cross_domain_decoding(domain, target_policy_manager, dictionary['source_latent_z'])

		####################################
		# (3 c) Cross domain encoding of target_trajectory_rollout into target latent_z. 
		####################################

		dictionary['target_subpolicy_inputs'], dictionary['target_latent_z'], dictionary['target_loglikelihood'], dictionary['target_kl_divergence'] = self.encode_decode_trajectory(target_policy_manager, i, trajectory_input=target_dict)

		####################################
		# (3 d) Cross domain decoding of target_latent_z into source trajectory. 
		# Can use the original start state, or also use the reverse trick for start state. Try both maybe.
		####################################

		source_trajectory_rollout, dictionary['source_subpolicy_inputs_crossdomain'] = self.cross_domain_decoding(domain, source_policy_manager, dictionary['target_latent_z'], start_state=dictionary['source_subpolicy_inputs'][0,:self.state_dim].detach().cpu().numpy())

		####################################
		# (4) Feed source and target latent z's to z_discriminator.
		####################################

		self.compute_discriminator_losses(domain, dictionary['source_latent_z'])

		####################################
		# (5) Compute all losses, reweight, and take gradient steps.
		####################################

		self.update_networks(dictionary, source_policy_manager)

		# viz_dict = {'domain': domain, 'discriminator_probs': discriminator_prob.squeeze(0).squeeze(0)[domain].detach().cpu().numpy()}			
		# self.update_plots(counter, viz_dict)

		# Encode decode function: First encodes, takes trajectory segment, and outputs latent z. The latent z is then provided to decoder (along with initial state), and then we get SOURCE domain subpolicy inputs. 
		# Cross domain decoding function: Takes encoded latent z (and start state), and then rolls out with target decoder. Function returns, target trajectory, action sequence, and TARGET domain subpolicy inputs. 

