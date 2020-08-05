from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags, app
import copy, os, imageio, scipy.misc, pdb, math, time, numpy as np

import robosuite, threading
from robosuite.wrappers import IKWrapper
import MocapVisualizationUtils
from mocap_processing.motion.pfnn import Animation, BVH
import matplotlib.pyplot as plt
from IPython import embed

class SawyerVisualizer():

	def __init__(self, has_display=False):

		# Create environment.
		print("Do I have a display?", has_display)
		# self.base_env = robosuite.make('BaxterLift', has_renderer=has_display)
		self.base_env = robosuite.make("SawyerViz",has_renderer=has_display)

		# Create kinematics object. 
		self.sawyer_IK_object = IKWrapper(self.base_env)
		self.environment = self.sawyer_IK_object.env        

	def update_state(self):
		# Updates all joint states
		self.full_state = self.environment._get_observation()

	def set_joint_pose_return_image(self, joint_angles, arm='both', gripper=False):

		# In the roboturk dataset, we've the following joint angles: 
		# ('time','right_j0', 'head_pan', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6', 'r_gripper_l_finger_joint', 'r_gripper_r_finger_joint')

		# Set usual joint angles through set joint positions API.
		self.environment.reset()
		self.environment.set_robot_joint_positions(joint_angles[:7])

		# For gripper, use "step". 
		# Mujoco requires actions that are -1 for Open and 1 for Close.

		# [l,r]
		# gripper_open = [0.0115, -0.0115]
		# gripper_closed = [-0.020833, 0.020833]
		# In mujoco, -1 is open, and 1 is closed.
		
		actions = np.zeros((8))
		actions[-1] = joint_angles[-1]

		# Move gripper positions.
		self.environment.step(actions)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None):

		image_list = []
		for t in range(trajectory.shape[0]):
			new_image = self.set_joint_pose_return_image(trajectory[t])
			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)            

class BaxterVisualizer():

	def __init__(self, has_display=False):

		# Create environment.
		print("Do I have a display?", has_display)
		# self.base_env = robosuite.make('BaxterLift', has_renderer=has_display)
		self.base_env = robosuite.make("BaxterViz",has_renderer=has_display)

		# Create kinematics object. 
		self.baxter_IK_object = IKWrapper(self.base_env)
		self.environment = self.baxter_IK_object.env        
	
	def update_state(self):
		# Updates all joint states
		self.full_state = self.environment._get_observation()

	def set_ee_pose_return_image(self, ee_pose, arm='right', seed=None):

		# Assumes EE pose is Position in the first three elements, and quaternion in last 4 elements. 
		self.update_state()

		if seed is None:
			# Set seed to current state.
			seed = self.full_state['joint_pos']

		if arm == 'right':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=ee_pose[:3],
				target_orientation_right=ee_pose[3:],
				target_position_left=self.full_state['left_eef_pos'],
				target_orientation_left=self.full_state['left_eef_quat'],
				rest_poses=seed
			)

		elif arm == 'left':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=self.full_state['right_eef_pos'],
				target_orientation_right=self.full_state['right_eef_quat'],
				target_position_left=ee_pose[:3],
				target_orientation_left=ee_pose[3:],
				rest_poses=seed
			)

		elif arm == 'both':
			joint_positions = self.baxter_IK_object.controller.inverse_kinematics(
				target_position_right=ee_pose[:3],
				target_orientation_right=ee_pose[3:7],
				target_position_left=ee_pose[7:10],
				target_orientation_left=ee_pose[10:],
				rest_poses=seed
			)
		image = self.set_joint_pose_return_image(joint_positions, arm=arm, gripper=False)
		return image

	def set_joint_pose_return_image(self, joint_pose, arm='both', gripper=False):

		# FOR FULL 16 DOF STATE: ASSUMES JOINT_POSE IS <LEFT_JA, RIGHT_JA, LEFT_GRIPPER, RIGHT_GRIPPER>.

		self.update_state()
		self.state = copy.deepcopy(self.full_state['joint_pos'])
		# THE FIRST 7 JOINT ANGLES IN MUJOCO ARE THE RIGHT HAND. 
		# THE LAST 7 JOINT ANGLES IN MUJOCO ARE THE LEFT HAND. 
		
		if arm=='right':
			# Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
			self.state[:7] = copy.deepcopy(joint_pose[:7])
		elif arm=='left':    
			# Assume joint_pose is 8 DoF - 7 for the arm, and 1 for the gripper.
			self.state[7:] = copy.deepcopy(joint_pose[:7])
		elif arm=='both':
			# The Plans were generated as: Left arm, Right arm, left gripper, right gripper.
			# Assume joint_pose is 16 DoF. 7 DoF for left arm, 7 DoF for right arm. (These need to be flipped)., 1 for left gripper. 1 for right gripper.            
			# First right hand. 
			self.state[:7] = joint_pose[7:14]
			# Now left hand. 
			self.state[7:] = joint_pose[:7]
		# Set the joint angles magically. 
		self.environment.set_robot_joint_positions(self.state)

		action = np.zeros((16))
		if gripper:
			# Left gripper is 15. Right gripper is 14. 
			# MIME Gripper values are from 0 to 100 (Close to Open), but we treat the inputs to this function as 0 to 1 (Close to Open), and then rescale to (-1 Open to 1 Close) for Mujoco.
			if arm=='right':
				action[14] = -joint_pose[-1]*2+1
			elif arm=='left':                        
				action[15] = -joint_pose[-1]*2+1
			elif arm=='both':
				action[14] = -joint_pose[15]*2+1
				action[15] = -joint_pose[14]*2+1
			# Move gripper positions.
			self.environment.step(action)

		image = np.flipud(self.environment.sim.render(600, 600, camera_name='vizview1'))
		return image

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None):

		image_list = []
		for t in range(trajectory.shape[0]):
			new_image = self.set_joint_pose_return_image(trajectory[t])
			image_list.append(new_image)

			# Insert white 
			if segmentations is not None:
				if t>0 and segmentations[t]==1:
					image_list.append(255*np.ones_like(new_image)+new_image)

		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

class MocapVisualizer():

	def __init__(self, has_display=False, args=None):

		# Load some things from the MocapVisualizationUtils and set things up so that they're ready to go. 
		# self.cam_cur = MocapVisualizationUtils.camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
		# 						origin=np.array([0.0, 0.0, 0.0]), 
		# 						vup=np.array([0.0, 0.0, 1.0]), 
		# 						fov=45.0)

		self.args = args

		# Default is local data. 
		self.global_data = False

		self.cam_cur = MocapVisualizationUtils.camera.Camera(pos=np.array([4.5, 0.0, 2.0]),
								origin=np.array([0.0, 0.0, 0.0]), 
								vup=np.array([0.0, 0.0, 1.0]), 
								fov=45.0)

		# Path to dummy file that is going to populate joint_parents, initial global positions, etc. 
		bvh_filename = "/private/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  

		# Run init before loading animation.
		MocapVisualizationUtils.init()
		MocapVisualizationUtils.global_positions, MocapVisualizationUtils.joint_parents, MocapVisualizationUtils.time_per_frame = MocapVisualizationUtils.load_animation(bvh_filename)

		# State sizes. 
		self.number_joints = 22
		self.number_dimensions = 3
		self.total_dimensions = self.number_joints*self.number_dimensions

		# Run thread of viewer, so that callbacks start running. 
		thread = threading.Thread(target=self.run_thread)
		thread.start()

		# Also create dummy animation object. 
		self.animation_object, _, _ = BVH.load(bvh_filename)

	def run_thread(self):
		MocapVisualizationUtils.viewer.run(
			title='BVH viewer',
			cam=self.cam_cur,
			size=(1280, 720),
			keyboard_callback=None,
			render_callback=MocapVisualizationUtils.render_callback_time_independent,
			idle_callback=MocapVisualizationUtils.idle_callback_return,
		) 

	def get_global_positions(self, positions, animation_object=None):
		# Function to get global positions corresponding to predicted or actual local positions.

		traj_len = positions.shape[0]

		def resample(original_trajectory, desired_number_timepoints):
			original_traj_len = len(original_trajectory)
			new_timepoints = np.linspace(0, original_traj_len-1, desired_number_timepoints, dtype=int)
			return original_trajectory[new_timepoints]

		if animation_object is not None:
			# Now copy over from animation_object instead of just dummy animation object.
			new_animation_object = Animation.Animation(resample(animation_object.rotations, traj_len), positions, animation_object.orients, animation_object.offsets, animation_object.parents)
		else:	
			# Create a dummy animation object. 
			new_animation_object = Animation.Animation(self.animation_object.rotations[:traj_len], positions, self.animation_object.orients, self.animation_object.offsets, self.animation_object.parents)

		# Then transform them.
		transformed_global_positions = Animation.positions_global(new_animation_object)

		# Now return coordinates. 
		return transformed_global_positions

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None):

		image_list = []

		if self.global_data:
			# If we predicted in the global setting, just reshape.
			predicted_global_positions = np.reshape(trajectory, (-1,self.number_joints,self.number_dimensions)) 
		else:
			# If it's local data, then transform to global. 
			# Assume trajectory is number of timesteps x number_dimensions. 
			# Convert to number_of_timesteps x number_of_joints x 3.
			predicted_local_positions = np.reshape(trajectory, (-1,self.number_joints,self.number_dimensions))

			# Assume trajectory was predicted in local coordinates. Transform to global for visualization.
			predicted_global_positions = self.get_global_positions(predicted_local_positions, animation_object=additional_info)

		# Copy into the global variable.
		MocapVisualizationUtils.global_positions = predicted_global_positions

		# Reset Image List. 
		MocapVisualizationUtils.image_list = []
		# Set save_path and prefix.
		MocapVisualizationUtils.save_path = gif_path
		MocapVisualizationUtils.name_prefix = gif_name.rstrip('.gif')
		# Now set the whether_to_render as true. 
		MocapVisualizationUtils.whether_to_render = True

		# Wait till rendering is complete. 
		x_count = 0
		while MocapVisualizationUtils.done_with_render==False and MocapVisualizationUtils.whether_to_render==True:
			x_count += 1
			time.sleep(1)
			
		# Now that rendering is complete, load images.
		image_list = MocapVisualizationUtils.image_list

		# Now actually save the GIF or return.
		if return_and_save:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)
			return image_list
		elif return_gif:
			return image_list
		else:
			imageio.mimsave(os.path.join(gif_path,gif_name), image_list)

class ToyDataVisualizer():

	def __init__(self):

		pass

	def visualize_joint_trajectory(self, trajectory, return_gif=False, gif_path=None, gif_name="Traj.gif", segmentations=None, return_and_save=False, additional_info=None):

		fig = plt.figure()		
		ax = fig.gca()
		ax.scatter(trajectory[:,0],trajectory[:,1],c=range(len(trajectory)),cmap='jet')
		plt.xlim(-10,10)
		plt.ylim(-10,10)

		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(int(height), int(width), 3)
		image = np.transpose(image, axes=[2,0,1])

		return image


if __name__ == '__main__':
	# end_eff_pose = [0.3, -0.3, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394]
	# end_eff_pose = [0.53303758, -0.59997265,  0.09359371,  0.77337391,  0.34998901, 0.46797516, -0.24576358]
	# end_eff_pose = np.array([0.64, -0.83, 0.09798524029948213, 0.38044099037703677, 0.9228975092885654, -0.021717379118030174, 0.05525572942370394])
	visualizer = MujocoVisualizer()
	# img = visualizer.set_ee_pose_return_image(end_eff_pose, arm='right')
	# scipy.misc.imsave('mj_vis.png', img)
