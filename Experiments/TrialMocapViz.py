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

import time
from IPython import embed

# Define function to load animation file. 
def load_animation(bvh_filename):
	animation, joint_names, time_per_frame = BVH.load(bvh_filename)
	joint_parents = animation.parents
	global_positions = Animation.positions_global(animation)
	return global_positions, joint_parents, time_per_frame


def render_callback():
	global start_time, global_positions, joint_parents, time_per_frame

	gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='y', origin=True, use_arrow=True)
	time_elapsed = time.time() - start_time
	frame_num = int(time_elapsed / time_per_frame) % len(global_positions)
	
	glPushMatrix()
	# glRotatef(90, -1, 0, 0)
	# glScalef(0.1, 0.1, 0.1)
	glScalef(1., 1., 1.)
	glEnable(GL_LIGHTING)
	for i in range(len(joint_parents)):
		pos = global_positions[frame_num][i]
		gl_render.render_point(pos, radius=0.25, color=[0.8, 0.8, 0.0, 1.0])
		j = joint_parents[i]
		if j!=-1:
			pos_parent = global_positions[frame_num][j]
			gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])            
	glPopMatrix()

##############################
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
	# print("Running time independent callback.")
	global global_positions, joint_parents

	gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='z', origin=True, use_arrow=True)

	# glPushMatrix()
	# # glRotatef(90, -1, 0, 0)
	# # glScalef(0.1, 0.1, 0.1)
	# glScalef(1., 1., 1.)
	# glEnable(GL_LIGHTING)

	# t = 0
	# for i in range(len(joint_parents)):
	# 	pos = global_positions[t][i]
	# 	gl_render.render_point(pos, radius=0.25, color=[0.8, 0.8, 0.0, 1.0])
	# 	j = joint_parents[i]
	# 	if j!=-1:
	# 		pos_parent = global_positions[t][j]
	# 		gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])            
	# glPopMatrix()

	##############################
	# Render Shadow of Character
	glEnable(GL_DEPTH_TEST)
	glDisable(GL_LIGHTING)
	glPushMatrix()
	glTranslatef(0, 0, 0.001)
	glScalef(1, 1, 0)
	render_pose_by_capsule(global_positions, 0, joint_parents, color=[0.5,0.5,0.5,1.0])	
	glPopMatrix()

	# Render Character
	glEnable(GL_LIGHTING)
	render_pose_by_capsule(global_positions, 0, joint_parents, color=np.array([85, 160, 173, 255])/255.0)
			
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

	if key == b'm':
		print("Entering Callback.")
		global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])

		viewer.drawGL()
		viewer.save_screen("/home/tanmayshankar/Research/Code/","TRY_VIZ")
 
	return

start_time = time.time()

cam_origin = 0.01*np.array([0, 50, 0])
cam_pos = cam_origin + np.array([0.0, 1.0, 3.5])

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

viewer.run(
	title='BVH viewer',
	# cam_pos=cam_pos,
	# cam_origin=cam_origin,
	cam=cam_cur,
	size=(1280, 720),
	keyboard_callback=keyboard_callback,
	render_callback=render_callback_time_independent,
)
