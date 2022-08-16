# Grasps2.msg
# an alternative formulation of the Grasps message, used as a stopgap measure (a hack)
# so that we can speedily implement kinematic feasibility filtering, which expects 
# this message type

Header header # sequence number, timestamp,frame ID
geometry_msgs/Pose[] poses # N grasp poses expressed in this frame
geometry_msgs/Pose[] orbital_poses # N orbital_poses, only published occasionally
float32[] widths
float32[] confs # N corresponding grasp confidences in [0, 1]