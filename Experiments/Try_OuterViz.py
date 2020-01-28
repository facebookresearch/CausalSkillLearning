#!/usr/bin/env python
import TrialMocapViz
import threading, time, numpy as np

bvh_filename = "/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  
filenames = [bvh_filename]
file_num = 0

print("About to run viewer.")

cam_cur = TrialMocapViz.camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
								origin=np.array([0.0, 0.0, 0.0]), 
								vup=np.array([0.0, 0.0, 1.0]), 
								fov=45.0)

def run_thread():
	TrialMocapViz.viewer.run(
		title='BVH viewer',
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=None,
		render_callback=TrialMocapViz.render_callback_time_independent,
		idle_callback=TrialMocapViz.idle_callback,
	)

# Run init before loading animation.
TrialMocapViz.init()
TrialMocapViz.global_positions, TrialMocapViz.joint_parents, TrialMocapViz.time_per_frame = TrialMocapViz.load_animation(filenames[file_num])
thread = threading.Thread(target=run_thread)
thread.start()

print("Going to actually call callback now.")
TrialMocapViz.whether_to_render = True