#!/usr/bin/env python
import mocap_processing
from mocap_processing.motion.pfnn import Animation, BVH
from mocap_processing.motion.pfnn import Animation, BVH

filename = "/checkpoint/dgopinath/amass/CMU/01/01_01_poses.bvh"

# def load_animation(bvh_filename):
#     animation, joint_names, time_per_frame = BVH.load(bvh_filename)
#     joint_parents = animation.parents
#     global_positions = Animation.positions_global(animation)
#     return global_positions, joint_parents, time_per_frame

def load_animation_data(bvh_filename):
    animation, joint_names, time_per_frame = BVH.load(bvh_filename)
    global_positions = Animation.positions_global(animation)
    # return global_positions, joint_parents, time_per_frame
    return global_positions, animation.positions, animation.rotations

# load_animation(filename)
# load_animation(filename)[0]
# load_animation(filename)[0].shape
# load_animation(filename)[1]
# load_animation(filename)[2]

# 1/load_animation(filename)[2]

# animation, joint_names, time_per_frame = BVH.load(filename)

# animation
# animation.positions
# animation.rotations
# animation.rotations.shape
# animation.position.shape
# animation.positions
# animation.positions.shape
# animation.rotations
# animation.rotations[0]
# animation.rotations[0][0]
# animation.rotations[0][1]
# get_ipython().run_line_magic('pwd', '')

directory = "/checkpoint/dgopinath/amass/CMU"
filelist = glob.glob("*/*.bvh")

for i in range(len(filelist)):
	filename = os.path.join(directory, filelist[i])
	global_positions, local_positions, local_rotations = load_animation_data(filename)

	data_element = {}
	data_element['global_positions'] = global_positions
	data_element['local_positions'] = global_positions
	data_element['local_rotations'] = global_positions