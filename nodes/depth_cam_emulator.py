#! /usr/bin/python3
# find_grasp_client.py
# A node that pretends to be a depth camera by loading PointCloud2 msgs from a
# bag file to `/camera/depth/points`

import rospy
import rosbag
from sensor_msgs.msg import PointCloud2
import pathlib
import sys

if __name__ == "__main__":
    # parse the single argument, e.g., "chair.bag"
    try:
        bag_name = sys.argv[1]
    except Exception as e:
        print("Argument parsing did not work. Expected a command such as \
        'rosrun grasp_synth depth_cam_emulator.py chair.bag'. ")
        raise(e)

    # get rosbag from ../bags/point_bloud.bag
    this_path = pathlib.Path(__file__).parent.absolute()
    bag_path = this_path.parent / 'bags' / bag_name

    rospy.init_node('depth_cam_emulator')
    pcl_pub = rospy.Publisher('camera/depth/points', PointCloud2, queue_size=10)

    # read point cloud and find grasps using the service
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, pcl_msg, t in bag.read_messages(
            topics=['/camera/depth/points']):

            r = rospy.Rate(30)

            while True:
                # Publish the input point cloud for visualization
                pcl_pub.publish(pcl_msg)
                r.sleep()