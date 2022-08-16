#! /usr/bin/python3
# find_grasp_client.py
# A dummy client node that requests 5 grasps.

import rospy
from grasp_synth.srv import FindGrasps
from grasp_synth.msg import Grasps
from std_msgs.msg import UInt32
import sys

def find_grasps_client(top_k: int):
    try:
        find_grasps = rospy.ServiceProxy('find_grasps', FindGrasps)
        return find_grasps(top_k)
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    try:
        top_k = int(sys.argv[1])
    except Exception as e:
        print("Argument parsing did not work. Expected a command such as \
        'rosrun grasp_synth grasp_client.py chair.bag 50'. ")
        raise(e)

    rospy.init_node('grasp_client')
    grasp_pub = rospy.Publisher('grasp_synth/grasps', Grasps, queue_size=10)
    rospy.wait_for_service('find_grasps')

    import time; time.sleep(4)

    # request grasps using the service
    resp = find_grasps_client(top_k)

    # Publish the poses for visualization
    grasp_pub.publish(resp.grasps)

