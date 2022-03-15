# Grasp.msg
# a message definition containing a single two-fingered grasp
# identified by the grasp detection network.
# The approach vector, baseline vector, and width 
# are as defined in Fig. 3 of (Sundermeyer et. al 2021)
# https://arxiv.org/pdf/2103.14127.pdf

# The poses and baselines are expressed in a reference frame
# defined in the header of the Grasps.msg (plural) message 

geometry_msgs/Pose pose
geometry_msgs/Vector3 approach
geometry_msgs/Vector3 baseline
float32 width