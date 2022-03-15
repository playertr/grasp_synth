#! /usr/bin/python3
# find_grasp_client.py
# A dummy client node that requests 5 grasps for a bagged
# point cloud.

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
from grasp_synth.srv import FindGrasps
from grasp_synth.msg import Grasps
from std_msgs.msg import UInt32
import pathlib
import sys

def find_grasps_client(pcl_msg: PointCloud2, top_k: int):
    try:
        find_grasps = rospy.ServiceProxy('find_grasps', FindGrasps)
        return find_grasps(pcl_msg, UInt32(top_k))
    except rospy.ServiceException as e:
        print(e)

if __name__ == "__main__":
    # parse the single argument, e.g., "chair.bag"
    try:
        bag_name = sys.argv[1]
        top_k = int(sys.argv[2])
    except Exception as e:
        print("Argument parsing did not work. Expected a command such as \
        'rosrun grasp_synth grasp_client.py chair.bag 50'. ")
        raise(e)

    # get rosbag from ../bags/point_bloud.bag
    this_path = pathlib.Path(__file__).parent.absolute()
    bag_path = this_path.parent / 'bags' / bag_name

    rospy.init_node('grasp_client')
    grasp_pub = rospy.Publisher('grasp_synth/grasps', Grasps, queue_size=10)
    pcl_pub = rospy.Publisher('grasp_synth/points', PointCloud2, queue_size=10)
    rospy.wait_for_service('find_grasps')

    # read point cloud and find grasps using the service
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, pcl_msg, t in bag.read_messages(
            topics=['/camera/depth/points']):

            # Request a response
            resp = find_grasps_client(pcl_msg, top_k)
            print(resp)

            # Publish the input point cloud for visualization
            pcl_pub.publish(pcl_msg)

            # Publish the poses for visualization
            grasp_pub.publish(resp.grasps)

