#! /usr/bin/python3
# read_pc.py 
# A trial script to see what is the data in the bag file

import rosbag
import pathlib



if __name__ == "__main__":
        this_path = pathlib.Path(__file__).parent.absolute()
        bag_path = this_path.parent / 'bags' / 'chair.bag'
        count = 0
        
        with rosbag.Bag(bag_path, 'r') as bag:
            # print("length = ",bag.read_messages(topics=['/camera/depth/points']))
            output = bag.read_messages(topics=['/camera/depth/points'])
            print("msg = ",output)
            for topic, pcl_msg, t in bag.read_messages(topics=['/camera/depth/points']):
                count +=1
                # print("pcl_msg = ",pcl_msg)
                # print("length = ",len(pcl_msg))
                # print("type = ", type(pcl_msg))
            print("count = ",count)

