#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import MocapVisualizationUtils
import threading, time, numpy as np

# bvh_filename = "/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  
bvh_filename = "/private/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  
filenames = [bvh_filename]
file_num = 0

print("About to run viewer.")

cam_cur = MocapVisualizationUtils.camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
								origin=np.array([0.0, 0.0, 0.0]), 
								vup=np.array([0.0, 0.0, 1.0]), 
								fov=45.0)

def run_thread():
	MocapVisualizationUtils.viewer.run(
		title='BVH viewer',
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=None,
		render_callback=MocapVisualizationUtils.render_callback_time_independent,
		idle_callback=MocapVisualizationUtils.idle_callback,
	)

def run_thread():
	MocapVisualizationUtils.viewer.run(
		title='BVH viewer',
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=None,
		render_callback=MocapVisualizationUtils.render_callback_time_independent,
		idle_callback=MocapVisualizationUtils.idle_callback_return,
	)


# Run init before loading animation.
MocapVisualizationUtils.init()
MocapVisualizationUtils.global_positions, MocapVisualizationUtils.joint_parents, MocapVisualizationUtils.time_per_frame = MocapVisualizationUtils.load_animation(filenames[file_num])
thread = threading.Thread(target=run_thread)
thread.start()

print("Going to actually call callback now.")
MocapVisualizationUtils.whether_to_render = True

x_count = 0
while MocapVisualizationUtils.done_with_render==False and MocapVisualizationUtils.whether_to_render==True:
	x_count += 1
	time.sleep(1)
	print("x_count is now: ",x_count)

print("We finished with the visualization!")
