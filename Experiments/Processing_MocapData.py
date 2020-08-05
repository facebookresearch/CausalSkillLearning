# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mocap_processing, glob, numpy as np, os
from mocap_processing.motion.pfnn import Animation, BVH
from mocap_processing.motion.pfnn import Animation, BVH
from IPython import embed

# Define function that loads global and local posiitons, and the rotations from a datafile.
def load_animation_data(bvh_filename):
    animation, joint_names, time_per_frame = BVH.load(bvh_filename)
    global_positions = Animation.positions_global(animation)
    # return global_positions, joint_parents, time_per_frame
    return global_positions, animation.positions, animation.rotations, animation

# Set directory. 
directory = "/checkpoint/dgopinath/amass/CMU"
save_directory = "/checkpoint/tanmayshankar/Mocap"
# Get file list. 
filelist = glob.glob(os.path.join(directory,"*/*.bvh"))

demo_list = []

print("Starting to preprocess data.")

for i in range(len(filelist)):

	print("Processing file number: ",i, " of ",len(filelist))
	# Get filename. 
	filename = os.path.join(directory, filelist[i])
	# Actually load file. 
	global_positions, local_positions, local_rotations, animation = load_animation_data(filename)

	# Create data element object.
	data_element = {}
	data_element['global_positions'] = global_positions
	data_element['local_positions'] = local_positions
	# Get quaternion as array.
	data_element['local_rotations'] = local_rotations.qs	
	data_element['animation'] = animation

	demo_list.append(data_element)

demo_array = np.array(demo_list)
np.save(os.path.join(save_directory,"Demo_Array.npy"),demo_array)