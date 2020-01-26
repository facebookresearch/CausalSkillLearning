#!/usr/bin/env python
import mocap_processing, glob, numpy as np
from mocap_processing.motion.pfnn import Animation, BVH
from mocap_processing.motion.pfnn import Animation, BVH

# Define function that loads global and local posiitons, and the rotations from a datafile.
def load_animation_data(bvh_filename):
    animation, joint_names, time_per_frame = BVH.load(bvh_filename)
    global_positions = Animation.positions_global(animation)
    # return global_positions, joint_parents, time_per_frame
    return global_positions, animation.positions, animation.rotations

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
	global_positions, local_positions, local_rotations = load_animation_data(filename)

	# Create data element object.
	data_element = {}
	data_element['global_positions'] = global_positions
	data_element['local_positions'] = global_positions
	data_element['local_rotations'] = global_positions

	demo_list.append(data_element)

demo_array = np.array(demo_list)
np.save(os.path.join(save_directory,"Demo_Array.npy"),demo_array)