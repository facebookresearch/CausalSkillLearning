#!/usr/bin/env python
import MocapVisualizer
import threading, time, numpy as np

bvh_filename = "/home/tanmayshankar/Research/Code/CausalSkillLearning/Experiments/01_01_poses.bvh"  
filenames = [bvh_filename]
file_num = 0

print("About to run viewer.")

cam_cur = MocapVisualizer.camera.Camera(pos=np.array([6.0, 0.0, 2.0]),
								origin=np.array([0.0, 0.0, 0.0]), 
								vup=np.array([0.0, 0.0, 1.0]), 
								fov=45.0)

def run_thread():
	MocapVisualizer.viewer.run(
		title='BVH viewer',
		cam=cam_cur,
		size=(1280, 720),
		keyboard_callback=None,
		render_callback=MocapVisualizer.render_callback_time_independent,
		idle_callback=MocapVisualizer.idle_callback,
	)

# Run init before loading animation.
MocapVisualizer.init()
MocapVisualizer.global_positions, MocapVisualizer.joint_parents, MocapVisualizer.time_per_frame = MocapVisualizer.load_animation(filenames[file_num])
thread = threading.Thread(target=run_thread)
thread.start()

print("Going to actually call callback now.")
MocapVisualizer.whether_to_render = True

