#!/usr/bin/env python
from mocap_processing.motion.pfnn import Animation, BVH
from basecode.render import glut_viewer as viewer
from basecode.render import gl_render, camera
from basecode.utils import basics
from basecode.math import mmMath

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import time, threading, pyautogui
from IPython import embed

global whether_to_render
whether_to_render = False

def init():
	global whether_to_render
	whether_to_render = False
	global_positions = None

# Define function to load animation file. 
def load_animation(bvh_filename):
	animation, joint_names, time_per_frame = BVH.load(bvh_filename)
	joint_parents = animation.parents
	global_positions = Animation.positions_global(animation)
	return global_positions, joint_parents, time_per_frame

# Function that draws body of animated character from the global positions.
def render_pose_by_capsule(global_positions, frame_num, joint_parents, scale=1.0, color=[0.5, 0.5, 0.5, 1], radius=0.05):
	glPushMatrix()
	glScalef(scale, scale, scale)

	for i in range(len(joint_parents)):
		pos = global_positions[frame_num][i]
		# gl_render.render_point(pos, radius=radius, color=color)
		j = joint_parents[i]
		if j!=-1:
			pos_parent = global_positions[frame_num][j]
			p = 0.5 * (pos_parent + pos)
			l = np.linalg.norm(pos_parent-pos)
			R = mmMath.getSO3FromVectors(np.array([0, 0, 1]), pos_parent-pos)
			gl_render.render_capsule(mmMath.Rp2T(R,p), l, radius, color=color, slice=16)        
	glPopMatrix()

def render_callback_time_independent():	
	global global_positions, joint_parents, counter

	gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='z', origin=True, use_arrow=True)

	# Render Shadow of Character

	glEnable(GL_DEPTH_TEST)
	glDisable(GL_LIGHTING)
	glPushMatrix()
	glTranslatef(0, 0, 0.001)
	glScalef(1, 1, 0)
	render_pose_by_capsule(global_positions, counter, joint_parents, color=[0.5,0.5,0.5,1.0])	
	glPopMatrix()

	# Render Character
	glEnable(GL_LIGHTING)
	render_pose_by_capsule(global_positions, counter, joint_parents, color=np.array([85, 160, 173, 255])/255.0)
			
def render_callback_trajectory():
	global global_positions, joint_parents, image_list

	gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='z', origin=True, use_arrow=True)

	# Render Shadow of Character

	# For now, use the first timestep of the posiitons.
	time = 0
	image_list = []

	for time in range(global_positions.shape[0]):
		print("Entering Loop.")

		glEnable(GL_DEPTH_TEST)
		glDisable(GL_LIGHTING)
		glPushMatrix()
		glTranslatef(0, 0, 0.001)
		glScalef(1, 1, 0)
		render_pose_by_capsule(global_positions, time, joint_parents, color=[0.5,0.5,0.5,1.0])	
		glPopMatrix()

		# Render Character
		glEnable(GL_LIGHTING)
		render_pose_by_capsule(global_positions, time, joint_parents, color=np.array([85, 160, 173, 255])/255.0)
			
		# 
		# viewer.drawGL()
		viewer.save_screen("/home/tanmayshankar/Research/Code/","TRY_VIZ_{}".format(time))
		viewer.return_screen_image()
		image_list.append(image)

def render_callback():
	global whether_to_render

	if whether_to_render:
		print("Whether to render is actually true.")

		render_callback_trajectory()

# def idle_callback():
# 	global whether_to_render

# 	if whether_to_render:
# 		print("Whether to render is actually true.")

# 		render_callback_trajectory()

# 	# Increment counter
# 	# Set frame number of trajectory to be rendered
# 	# Using the time independent rendering. 
# 	# Call drawGL and savescreen. 
# 	# Since this is an idle callback, drawGL won't call itself (only calls render callback).

def idle_callback():

	global whether_to_render, counter, global_positions
	if whether_to_render:
		print("Whether to render is actually true, with counter:",counter)
		# render_callback_time_independent()
		viewer.drawGL()
		viewer.save_screen("/home/tanmayshankar/Research/Code/","Visualize_Image_{}".format(counter))

		counter += 1

	# If whether to render is false, reset the counter.
	else:
		counter = 0

def keyboard_callback(key):
	print("Entering the keyboard callback.", key)
	global filenames, file_num, global_positions, joint_parents, time_per_frame

	if key == b'.':
		file_num += 1
		global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])
		print(filenames[file_num])
	if key == b',':
		file_num -= 1
		global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])
		print(filenames[file_num])

	# This branch shows how to save the environment to file.
	if key == b'm':
		print("Entering Callback.")
		global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])

		viewer.drawGL()
		viewer.save_screen("/home/tanmayshankar/Research/Code/","TRY_VIZ")
 
	return

# bvh_filename = "/checkpoint/dgopinath/amass/CMU/01/01_01_poses.bvh"  
bvh_filename = "/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  
filenames = [bvh_filename]
file_num = 0
global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])

print("About to run viewer.")

cam_cur = camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
								origin=np.array([0.0, 0.0, 0.0]), 
								vup=np.array([0.0, 0.0, 1.0]), 
								fov=45.0)


def run_thread():
	viewer.run(
		title='BVH viewer',
		# cam_pos=cam_pos,
		# cam_origin=cam_origin,
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=keyboard_callback,
		render_callback=render_callback_time_independent,
	)

def run_thread():
	viewer.run(
		title='BVH viewer',
		# cam_pos=cam_pos,
		# cam_origin=cam_origin,
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=keyboard_callback,
		render_callback=render_callback_trajectory,
	)

def run_thread():
	viewer.run(
		title='BVH viewer',
		# cam_pos=cam_pos,
		# cam_origin=cam_origin,
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=keyboard_callback,
		render_callback=render_callback_time_independent,
		idle_callback=idle_callback,
	)

thread = threading.Thread(target=run_thread)
thread.start()


print("OKAY! PAY ATTENTION!")
time.sleep(5)
whether_to_render = True
time.sleep(5)
whether_to_render = False