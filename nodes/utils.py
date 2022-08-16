import rospy
import tf2_ros
import numpy as np

## tf2 abstraction
# https://gitlab.msu.edu/av/av_notes/-/blob/master/ROS/Coordinate_Transforms.md
class TFHelper():
    def __init__(self):
        ''' Create a buffer of transforms and update it with TransformListener '''
        self.tfBuffer = tf2_ros.Buffer()           # Creates a frame buffer
        tf2_ros.TransformListener(self.tfBuffer)   # TransformListener fills the buffer as background task
    
    def get_transform(self, source_frame, target_frame):
        ''' Lookup latest transform between source_frame and target_frame from the buffer '''
        try:
            trans = self.tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(0.2) )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f'Cannot find transformation from {source_frame} to {target_frame}.')
            raise e
        return trans     # Type: TransformStamped