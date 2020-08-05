# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""

# For both arms and grippers. 
python -m SkillsfromDemonstrations.Experiments.UseSkillsRL.TrainZPolicyRL --train --transformer --nz=64 --nh=64 --variable_nseg=False --network_dir=saved_models/T356_fnseg_vae_sl2pt0_kldwt0pt002_finetune --variable_ns=False --st_space=joint_both_gripper --vae_enc 
"""

from __future__ import absolute_import

import os, sys, torch
import matplotlib.pyplot as plt
from ...DataLoaders import MIME_DataLoader
from ..abstraction import mime_eval
from ..abstraction.abstraction_utils import ScoreFunctionEstimator
from .PolicyNet import PolicyNetwork, PolicyNetworkSingleTimestep, AltPolicyNetworkSingleTimestep
from absl import app, flags
import imageio, numpy as np, copy, os, shutil
from IPython import embed
import robosuite
import tensorboard, tensorboardX

flags.DEFINE_boolean('train',False,'Whether to run train.')
flags.DEFINE_boolean('debug',False,'Whether to debug.')
# flags.DEFINE_float('sf_loss_wt', 0.1, 'Weight of pseudo loss for SF estimator')
# flags.DEFINE_float('kld_loss_wt', 0, 'Weight for KL Divergence loss if using VAE encoder.')
flags.DEFINE_float('reinforce_loss_wt', 1., 'Weight for primary reinforce loss.')
# flags.DEFINE_string('name',None,'Name to give run.')

class ZPolicyTrainer(object):

	def __init__(self, opts):

		self.opts = opts

		self.input_size = self.opts.n_state
		self.zpolicy_input_size = 85
		self.hidden_size = 20
		self.output_size = self.opts.nz

		self.primitive_length = 10
		self.learning_rate = 1e-4
		self.number_epochs = 200
		self.number_episodes = 500
		self.save_every_epoch = 5
		self.maximum_skills = 6

	def initialize_plots(self):
		self.log_dir = os.path.join("SkillsfromDemonstrations/cachedir/logs/RL",self.opts.name)
		if not(os.path.isdir(self.log_dir)):
			os.mkdir(self.log_dir)
		self.writer = tensorboardX.SummaryWriter(self.log_dir)

	def setup_networks(self):
		# Set up evaluator to load mime model and stuff.
		self.evaluator = mime_eval.PrimitiveDiscoverEvaluator(self.opts)
		self.evaluator.setup_testing(split='val')

		# Also create a ZPolicy.
		# self.z_policy = PolicyNetworkSingleTimestep(opts=self.opts, input_size=self.zpolicy_input_size, hidden_size=self.hidden_size, output_size=self.output_size).cuda()
		self.z_policy = AltPolicyNetworkSingleTimestep(opts=self.opts, input_size=self.zpolicy_input_size, hidden_size=self.hidden_size, output_size=self.output_size).cuda()

		if self.opts.variable_nseg:
			self.sf_loss_fn = ScoreFunctionEstimator()
		
		# Creating optimizer. 
		self.z_policy_optimizer = torch.optim.Adam(self.z_policy.parameters(), lr=self.learning_rate)
		
	def load_network(self, network_dir):
		# Load the evaluator networks (Abstraction network and skill network)    
		self.evaluator.load_network(self.evaluator.model, 'pred', 'latest', network_dir=network_dir)

		# Freeze parameters of the IntendedTrajectoryPredictorModel.
		for parameter in self.evaluator.model.parameters():
			parameter.require_grad = False

	def save_zpolicy_model(self, path, suffix):
		if not(os.path.isdir(path)):
			os.mkdir(path)
		save_object = {}
		save_object['ZPolicy'] = self.z_policy.state_dict()
		torch.save(save_object,os.path.join(path,"ZPolicyModel"+suffix))

	def load_all_models(self, path):
		load_object = torch.load(path)
		self.z_policy.load_state_dict(load_object['ZPolicy'])

	# def update_plots(self, counter, sample_map, loglikelihood):
	def update_plots(self, counter):

		if self.opts.variable_nseg:
			self.writer.add_scalar('Stop_Prob_Reinforce_Loss', torch.mean(self.stop_prob_reinforce_loss), counter)
		self.writer.add_scalar('Predicted_Zs_Reinforce_Loss', torch.mean(self.reinforce_predicted_Zs), counter)
		self.writer.add_scalar('KL_Divergence_Loss', torch.mean(self.kld_loss_seq), counter)
		self.writer.add_scalar('Total_Loss', torch.mean(self.total_loss), counter)
		
	def assemble_input(self, trajectory):
		traj_start = trajectory[0]
		traj_end = trajectory[-1]
		return torch.cat([torch.tensor(traj_start).cuda(),torch.tensor(traj_end).cuda()],dim=0)

	# def update_networks(self, state_traj, reward_traj, predicted_Zs):
	def update_networks(self, state_traj_torch, reward_traj, latent_z_seq, log_prob_seq, stop_prob_seq, stop_seq, kld_loss_seq):
		# embed()
		# Get cummulative rewards corresponding to actions executed after selecting a particular Z. -# This is basically adding up the rewards from the end of the array. 
		# cumm_reward_to_go = torch.cumsum(torch.tensor(reward_traj[::-1]).cuda().float())[::-1]
		cumm_reward_to_go_numpy = copy.deepcopy(np.cumsum(copy.deepcopy(reward_traj[::-1]))[::-1])
		cumm_reward_to_go = torch.tensor(cumm_reward_to_go_numpy).cuda().float()

		self.total_loss = 0.

		if self.opts.variable_nseg:
			# Remember, this stop probability loss is for stopping predicting Z's, #NOT INTERMEDIATE TIMESTEPS! 
			# So we still use cumm_reward_to_go rather than cumm_reward_to_go_array

			self.stop_prob_reinforce_loss = self.sf_loss_fn.forward(cumm_reward_to_go, stop_prob_seq.unsqueeze(1), stop_seq.long()) 
			# Add reinforce loss and loss value.             
			self.total_loss += self.opts.sf_loss_wt*self.stop_prob_reinforce_loss

		# Now adding the reinforce loss associated with predicted Zs. 
		# (Remember, we want to maximize reward times log prob, so multiply by -1 to minimize.)

		self.reinforce_predicted_Zs = (self.opts.reinforce_loss_wt * -1. * cumm_reward_to_go*log_prob_seq.view(-1)).sum()
		self.total_loss += self.reinforce_predicted_Zs

		# Add loss term with KL Divergence between 0 mean Gaussian and predicted Zs. 

		self.kld_loss_seq = kld_loss_seq
		self.total_loss += self.opts.kld_loss_wt*self.kld_loss_seq[0]

		# Zero gradients of optimizer, compute backward, then step optimizer. 
		self.z_policy_optimizer.zero_grad()
		self.total_loss.sum().backward()
		self.z_policy_optimizer.step()

	def reorder_actions(self, actions):

		# Assume that the actions are 16 dimensional, and are ordered as: 
		# 7 DoF for left arm, 7 DoF for right arm, 1 for left gripper, and 1 for right gripper. 

		# The original trajectory has gripper values from 0 (Close) to 1 (Open), but we've to rescale to -1 (Open) to 1 (Close) for Mujoco. 
		# And handle joint velocities.
		# MIME Gripper values are from 0 to 100 (Close to Open), but we assume actions has values from 0 to 1 (Close to Open), and then rescale to (-1 Open to 1 Close) for Mujoco.
		# Mujoco needs them flipped.

		indices = np.array([ 7,  8,  9, 10, 11, 12, 13,  0,  1,  2,  3,  4,  5,  6, 15, 14])
		reordered_actions = actions[:,indices]
		reordered_actions[:,14:] = 1 - 2*reordered_actions[:,14:]
		return reordered_actions

	def run_episode(self, counter):

		# For number of epochs:
		#   # 1) Given start and goal (for reaching task, say)
		#   # 2) Run Z_Policy on start and goal to retrieve predicted Zs.
		#   # 3) Decode predicted Zs into trajectory. 
		#   # 4) Retrieve "actions" from trajectory. 
		#   # 5) Feed "actions" into RL environment and collect reward. 
		#   # 6) Train ZPolicy to maximize cummulative reward with favorite RL algorithm. 

		# Reset environment. 
		state = self.environment.reset()
		terminal = False
		reward_traj = None		
		state_traj_torch = None
		t_out = 0
		stop = False
		hidden = None
		latent_z_seq = None
		stop_prob_seq = None
		stop_seq = None
		log_prob_seq = None
		kld_loss_seq = 0.
		previous_state = None

		while terminal==False and stop==False:
			
			########################################################
			######## 1) Collect input for first timestep. ##########
			########################################################
			zpolicy_input = np.concatenate([state['robot-state'],state['object-state']]).reshape(1,self.zpolicy_input_size)

			########################################################
			# 2) Feed into the Z policy to retrieve the predicted Z.
			######################################################## 
			latent_z, stop_probability, stop, log_prob, kld_loss, hidden = self.z_policy.forward(zpolicy_input, hidden=hidden)
			latent_z = latent_z.squeeze(1)

			########################################################
			############## 3) Decode into trajectory. ##############
			########################################################

			primitive_and_skill_stop_prob = self.evaluator.model.primitive_decoder(latent_z)			
			traj_seg = primitive_and_skill_stop_prob[0].squeeze(1).detach().cpu().numpy()					

			if previous_state is None:
				previous_state = traj_seg[-1].reshape(1,self.opts.n_state)
			else:
				# Concatenate previous state to trajectory, so that when we take actions we get an action from previous segment to the current one. 
				traj_seg = np.concatenate([previous_state,traj_seg],axis=0)
				previous_state = traj_seg[-1].reshape(-1,self.opts.n_state)

			########################################################
			## 4) Finite diff along time axis to retrieve actions ##
			########################################################
			actions = np.diff(traj_seg,axis=0)
			actions = self.reorder_actions(actions)
			actions_torch = torch.tensor(actions).cuda().float()
		
			cummulative_reward_in_segment = 0.			
			# Run step into evironment for all actions in this segment. 
			t = 0
			while t<actions_torch.shape[0] and terminal==False:
				
				# Step. 
				state, onestep_reward, terminal, success = self.environment.step(actions[t])		

				# Collect onestep_rewards within this segment. 
				cummulative_reward_in_segment += float(onestep_reward)
				# Assuming we have fixed_ns (i.e. novariable_ns), we can use the set decoding length of primitives to assign cummulative reward-to-go values to the various predicted Z variables. 
				# (This is also why we need the reward history, and not just the cummulative rewards obtained over the course of training.
				
				t+=1 

			# Everything is going to be set to None, so set variables. 
			# Do some bookkeeping in life.
			if t_out==0:
				state_traj_torch = torch.tensor(zpolicy_input).cuda().float().view(-1,self.zpolicy_input_size)
				latent_z_seq = latent_z.view(-1,self.opts.nz)
				stop_seq = stop.clone().detach().view(-1,1)
				stop_prob_seq = stop_probability.view(-1,2)
				log_prob_seq = log_prob.view(-1,1)
				# reward_traj = torch.tensor(copy.deepcopy(cummulative_reward_in_segment)).cuda().float().view(-1,1)
				reward_traj = np.array(cummulative_reward_in_segment).reshape((1,1))
			else:
				state_traj_torch = torch.cat([state_traj_torch, torch.tensor(zpolicy_input).cuda().float().view(-1,self.zpolicy_input_size)],dim=0)
				latent_z_seq = torch.cat([latent_z_seq, latent_z.view(-1,self.opts.nz)], dim=0)				
				stop_seq = torch.cat([stop_seq, stop.view(-1,1)], dim=0)
				stop_prob_seq = torch.cat([stop_prob_seq, stop_probability.view(-1,2)], dim=0)
				log_prob_seq = torch.cat([log_prob_seq, log_prob.view(-1,1)], dim=0)
				# reward_traj = torch.cat([reward_traj.view(-1,1), torch.tensor(copy.deepcopy(cummulative_reward_in_segment)).cuda().float().view(-1,1)])
				reward_traj = np.concatenate([reward_traj, np.array(cummulative_reward_in_segment).reshape((1,1))], axis=0)

			# Either way: 
			kld_loss_seq += kld_loss
			t_out += 1 	
			# print(t_out)			

			# Set to false by default. 
			if self.opts.variable_nseg==False:
				stop = False

			if t_out>=self.maximum_skills:
				stop = True

			# if self.opts.debug==True:
			# 	embed()

		if self.opts.train:
			# 6) Feed states, actions, reward, and predicted Zs to update. (These are all lists of tensors.)
			# self.update_networks(state_traj_torch, action_torch, reward_traj, latent_zs)
			self.update_networks(state_traj_torch, reward_traj, latent_z_seq, log_prob_seq, stop_prob_seq, stop_seq, kld_loss_seq)
			self.update_plots(counter)

	def setup_RL_environment(self, has_display=False):
		
		# Create Mujoco environment. 
		self.environment = robosuite.make("BaxterLift", has_renderer=has_display)
		self.initialize_plots()

	def trainRL(self):


		# Basic function to train.		
		counter = 0

		for e in range(self.number_epochs):

			# Number of episodes per epoch.
			for i in range(self.number_episodes):

				print("#########################################")
				print("Epoch: ",e,"Traj: ",i)

				# Run an episode.
				self.run_episode(counter)               

				counter += 1

			if self.opts.train and e%self.save_every_epoch==0:
				self.save_zpolicy_model(os.path.join("saved_models/RL",self.opts.name), "epoch{0}".format(e))

def main(_):
	
	# This is only to be executed for notebooks. 
	# flags.FLAGS([''])    
	opts = flags.FLAGS

	# Set state space. 
	if opts.st_space == 'ee_r' or opts.st_space == 'ee_l':
		opts.n_state = 7
	if opts.st_space == 'joint_ra' or opts.st_space == 'joint_la':
		opts.n_state = 7
	if opts.st_space == 'joint_both':
		opts.n_state = 14
	elif opts.st_space == 'ee_all':
		opts.n_state = 14
	elif opts.st_space == 'joint':
		opts.n_state = 17
	elif opts.st_space =='joint_both_gripper':
		opts.n_state = 16

	opts.logging_dir = os.path.join(opts.logging_dir, 'mime')
	opts.transformer = True

	torch.manual_seed(0)  

	# Create instance of class.
	zpolicy_trainer = ZPolicyTrainer(opts)
	zpolicy_trainer.setup_networks()
	zpolicy_trainer.setup_RL_environment()
	# Still need this to load primitive decoder network.
	zpolicy_trainer.load_network(opts.network_dir)
	zpolicy_trainer.trainRL()    
	

if __name__ == '__main__':
	app.run(main)
