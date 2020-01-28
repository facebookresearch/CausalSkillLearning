#!/usr/bin/env python
from TrialMocapViz import *
import threading, pyautogui

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


# viewer.run(
# 	title='BVH viewer',
# 	# cam_pos=cam_pos,
# 	# cam_origin=cam_origin,
# 	cam=cam_cur,
# 	size=(1280, 720),
# 	keyboard_callback=keyboard_callback,
# 	render_callback=render_callback_time_independent,
# )

init()

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

thread = threading.Thread(target=run_thread)
thread.start()

