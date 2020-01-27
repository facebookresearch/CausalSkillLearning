#!/usr/bin/env python
from mocap_processing.motion.pfnn import Animation, BVH
from basecode.render import glut_viewer as viewer
from basecode.render import gl_render
from basecode.utils import basics

import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import time
from IPython import embed

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
	glScalef(0.1, 0.1, 0.1)

	glEnable(GL_LIGHTING)
	for i in range(len(joint_parents)):
		pos = global_positions[frame_num][i]
		gl_render.render_point(pos, radius=0.25, color=[0.8, 0.8, 0.0, 1.0])
		j = joint_parents[i]
		if j!=-1:
			pos_parent = global_positions[frame_num][j]
			gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])            
	glPopMatrix()

def render_callback_time_independent():
	# print("Running time independent callback.")
	global global_positions, joint_parents

	gl_render.render_ground(size=[100, 100], color=[0.8, 0.8, 0.8], axis='y', origin=True, use_arrow=True)

	glPushMatrix()
	# glRotatef(90, -1, 0, 0)
	glScalef(0.1, 0.1, 0.1)

	glEnable(GL_LIGHTING)

	t = 0
	for i in range(len(joint_parents)):
		pos = global_positions[t][i]
		gl_render.render_point(pos, radius=0.25, color=[0.8, 0.8, 0.0, 1.0])
		j = joint_parents[i]
		if j!=-1:
			pos_parent = global_positions[t][j]
			gl_render.render_line(p1=pos_parent, p2=pos, color=[0, 0, 0, 1])            
	glPopMatrix()
			
def keyboard_callback(key):
	print("Entering the keyboard clalba.", key)
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
		viewer.save_screen(".","TRY_VIZ")

	return

start_time = time.time()

cam_origin = 0.01*np.array([0, 50, 0])
cam_pos = cam_origin + np.array([0.0, 1.0, 3.5])

bvh_filename = "/checkpoint/dgopinath/amass/CMU/01/01_01_poses.bvh"  
filenames = [bvh_filename]
file_num = 0
global_positions, joint_parents, time_per_frame = load_animation(filenames[file_num])

print("About to run viewer.")

embed()

viewer.run(
    title='BVH viewer',
    # cam_pos=cam_pos,
    # cam_origin=cam_origin,
    size=(1280, 720),
    keyboard_callback=keyboard_callback,
    render_callback=render_callback_time_independent,
)