#! /usr/bin/python3
# Read in the predicted grasp poses and publish markers to visualize.
# Each marker shows the grasp pose and the quality relative to the 
# other reported potential grasp poses.

import rospy
from grasp_synth.msg import Grasps
from visualization_msgs.msg import Marker, MarkerArray
from matplotlib import cm

marker_array_pub = rospy.Publisher('/grasp_synth/filtered_grasp_pose_markers', MarkerArray, queue_size=100)
    
def gripper_marker():
    marker = Marker()
    marker.type = marker.MESH_RESOURCE
    marker.action = marker.MODIFY
    marker.mesh_resource = "package://grasp_synth/urdf/gripper_yawed.stl"
    marker.scale.x = 1.0
    marker.scale.y = 1.0
    marker.scale.z = 1.0
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    return marker

def poses_cb(msg):
    
    viridis = cm.get_cmap('viridis', 12)

    _range = max(msg.confs) - min(msg.confs)
    _range = 1 if _range == 0 else _range
    cmap = lambda x: viridis(
        (x - min(msg.confs))/(_range)
    ) # normalize linearly
    
    marker_array = MarkerArray()

    poses = [grasp.pose for grasp in msg.grasps]
    for i, pose in enumerate(poses):
        marker = gripper_marker()
        marker.id = i
        marker.color.a = 0.5
        color = cmap(msg.confs[i])
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.header = msg.header
        marker.pose.orientation = pose.orientation
        marker.pose.position = pose.position
        marker_array.markers.append(marker)

    marker_array_pub.publish(marker_array)

rospy.init_node('publish_grasp_markers')
rospy.Subscriber('/filtered_grasps', Grasps, poses_cb, queue_size=1)

rospy.loginfo('Ready to grasp markers.')
rospy.spin()