#!/usr/bin/env python
from mocap_processing.motion.pfnn import Animation, BVH
from basecode.render import glut_viewer as viewer
from basecode.render import gl_render, camera
from basecode.utils import basics
from basecode.math import mmMath

import numpy as np, imageio

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import time, threading
from IPython import embed

global whether_to_render
whether_to_render = False

def init():
	global whether_to_render, global_positions, counter, joint_parents, done_with_render, save_path, name_prefix, image_list
	whether_to_render = False
	done_with_render = False
	global_positions = None
	joint_parents = None
	save_path = "/home/tanmayshankar/Research/Code/"
	name_prefix = "Viz_Image"
	image_list = []
	counter = 0

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

# Callback that renders one pose. 
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
			
# Callback that runs rendering when the global variable is set to true.
def idle_callback():
	# 	# Increment counter
	# 	# Set frame number of trajectory to be rendered
	# 	# Using the time independent rendering. 
	# 	# Call drawGL and savescreen. 
	# 	# Since this is an idle callback, drawGL won't call itself (only calls render callback).

	global whether_to_render, counter, global_positions, done_with_render, save_path, name_prefix
	done_with_render = False

	# if whether_to_render and counter<global_positions.shape[0]:	
	if whether_to_render and counter<10:	

		print("Whether to render is actually true, with counter:",counter)
		# render_callback_time_independent()
		viewer.drawGL()
		viewer.save_screen(save_path, "Image_{}_{}".format(name_prefix, counter))
		# viewer.save_screen("/home/tanmayshankar/Research/Code/","Visualize_Image_{}".format(counter))

		counter += 1

		# Set whether to render to false if counter exceeded. 
		# if counter>=global_positions.shape[0]:
		if counter>=10:
			whether_to_render = False
			done_with_render = True

	# If whether to render is false, reset the counter.
	else:
		counter = 0

def idle_callback_return():
	# 	# Increment counter
	# 	# Set frame number of trajectory to be rendered
	# 	# Using the time independent rendering. 
	# 	# Call drawGL and savescreen. 
	# 	# Since this is an idle callback, drawGL won't call itself (only calls render callback).

	global whether_to_render, counter, global_positions, done_with_render, save_path, name_prefix, image_list
	done_with_render = False

	if whether_to_render and counter<global_positions.shape[0]:	
	# if whether_to_render and counter<10:	

		print("Whether to render is actually true, with counter:",counter)
		# render_callback_time_independent()
		viewer.drawGL()
		name = "Image_{}_{}".format(name_prefix, counter)
		viewer.save_screen(save_path, name)
		img = imageio.imread(os.path.join(save_path, name+".png"))
		image_list.append(img)

		counter += 1

		# Set whether to render to false if counter exceeded. 
		if counter>=global_positions.shape[0]:
		# if counter>=10:
			whether_to_render = False
			done_with_render = True

	# If whether to render is false, reset the counter.
	else:
		counter = 0