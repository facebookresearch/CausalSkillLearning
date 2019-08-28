import sys
import copy
import rospy
import subprocess
import glob
import ast
import struct
import json
import os
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from natsort import natsorted 
import shutil

from moveit_msgs.srv import GetPositionFK
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from IPython import embed
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
            SolvePositionIK,
                SolvePositionIKRequest,
                )
import baxter_interface

"""This code computes the Forward Kinematic solution using ROS service calls. Given the joint positions, FK solution, if available is found via Compute_FKin() and is written to local directory via the DumpFiles(). Inverse Kineematics is implemented as well Compute_IKin() """

#Initialization
joint_names = []
joint_val = []
#work_dir = '/home/dhiraj/Desktop/Sub_DataSet/'
work_dir = '/mnt/backUp/DataSubset/'

flag = 0
prev_name_left = []
prev_name_right = []
prev_pos_right = []
prev_pos_left = []

def DumpFiles(left, right,count,file_name):
        #This function writes the computed end-effector pose
        #Args: left, right: respective joint FK solution  

        test_dir = work_dir+file_name+"/RD "+file_name+"/fk/"
        try:
                os.makedirs(test_dir)
                os.makedirs(test_dir+"left/")
                os.makedirs(test_dir+"right/")
        except OSError:
            pass
        print "WRITING IT FOR: ",test_dir
        with open(test_dir+"left/"+str(count)+".txt",'wb') as fp:
            fp.write("position\n")
            fp.write(str(left.pose_stamped[0].pose.position.x)+"\n")
            fp.write(str(left.pose_stamped[0].pose.position.y)+"\n")
            fp.write(str(left.pose_stamped[0].pose.position.z)+"\n")
            fp.write("orientation\n")
            fp.write(str(left.pose_stamped[0].pose.orientation.x)+"\n")
            fp.write(str(left.pose_stamped[0].pose.orientation.y)+"\n")  
            fp.write(str(left.pose_stamped[0].pose.orientation.z)+"\n")
            fp.write(str(left.pose_stamped[0].pose.orientation.w)+"\n")

        with open(test_dir+"right/"+str(count)+".txt",'wb') as fp:
            fp.write("position\n")
            fp.write(str(right.pose_stamped[0].pose.position.x)+"\n")
            fp.write(str(right.pose_stamped[0].pose.position.y)+"\n")
            fp.write(str(right.pose_stamped[0].pose.position.z)+"\n")
            fp.write("orientation\n")
            fp.write(str(right.pose_stamped[0].pose.orientation.x)+"\n")
            fp.write(str(right.pose_stamped[0].pose.orientation.y)+"\n")  
            fp.write(str(right.pose_stamped[0].pose.orientation.z)+"\n")
            fp.write(str(right.pose_stamped[0].pose.orientation.w)+"\n")

def Compute_IKin(limb, seed_name, seed_pos, pose):
    # ROS Params initalization
    ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    seed = JointState()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    limb_joints = {}

    ikreq.pose_stamp.append(pose[0])
    seed.name = seed_name
    seed.position = seed_pos
    ikreq.seed_angles.append(seed) 
    
    try:
        rospy.wait_for_service(ns, 5.0)
        resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
        rospy.logerr("Service call failed: %s" % (e,))
        return 1
    
    #Check if result is valid + type of seed
    resp_seeds = struct.unpack('<%dB' % len(resp.result_type),resp.result_type)
    if (resp_seeds[0] != resp.RESULT_INVALID):
        seed_str = {
            ikreq.SEED_USER: 'User Provided Seed',
            ikreq.SEED_CURRENT: 'Current Joint Angles',
            ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                    }.get(resp_seeds[0], 'None')
        print("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %(seed_str,))
        #Format solution
        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
        #print "\nIK Joint Solution:\n", limb_joints 
        prev_name = resp.joints[0].name
        prev_pos = resp.joints[0].position
        
    else:
        print ("INVALID POSE")

    return limb_joints,prev_name,prev_pos


def Compute_FKin():
    #ROS Params
    rospy.init_node("kinematics_2")
    robot = moveit_commander.RobotCommander()

    #Initialize MoveGroupCpmmanders for both the arms. 
    right_arm = moveit_commander.MoveGroupCommander("right_arm")    
    left_arm = moveit_commander.MoveGroupCommander("left_arm")
    fk_right = [right_arm.get_end_effector_link()]
    fk_left = [left_arm.get_end_effector_link()]

    #Service Call
    header = Header(0,rospy.Time.now(),"/base") 
    joints_info = RobotState()
    rospy.wait_for_service('compute_fk')

    try:
        moveit_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
    except rospy.ServiceException as e:
        rospy.logerror("Service call failed: %s"%e)

    #Read and sort file names
    dirs = glob.glob(work_dir+"*") 
    for d in dirs:
        count = 0 #reset count to 0 after 60 time steps
        print "For directory: ",d," , count: ",count
        files = glob.glob(d+"/RD "+os.path.basename(d)+"/baxter_right_joints/*.txt")
        files = natsorted(files, key=lambda y: y.lower())
        global flag
        global prev_pos_left
        global prev_pos_right
        global prev_name_left
        global prev_name_right
        file_name = os.path.basename(d) 

        if os.path.isdir( work_dir+file_name+"/RD "+file_name+"/fk/"):
            print "file_name exists: ", work_dir+file_name+"/RD "+file_name+"/fk/"
        else:
            print "NEED to write: ",file_name
            for f in files:
                joints = open(f).read()
                joints_dict = ast.literal_eval(joints)   #All 17 joint angles
                joints_dict.pop("head_nod",None)
                joints_dict.pop("head_pan",None)
                joints_dict.pop("torso_t0",None)
                joints_info.joint_state.name = joints_dict.keys() 
                joints_info.joint_state.position = joints_dict.values()
     
                #Rosservice Call for moving the Baxter Arms. 
                #Make sure Baxter arms are above the table to avoid collision if you're commanding
                pose_right = moveit_fk(header, fk_right, joints_info)
                pose_left = moveit_fk(header, fk_left, joints_info)
                if flag == 0:
                    prev_name_left = joints_dict.keys()
                    prev_name_right = joints_dict.keys()
                    prev_pos_left = joints_dict.values()
                    prev_pos_right = joints_dict.values()
                 
                flag = 1
                DumpFiles(pose_left,pose_right,count,os.path.basename(d))
                count = count+1
            
                 
#Main
if __name__ == "__main__":
    #Forward Kinematics
    Compute_FKin()