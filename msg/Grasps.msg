# Grasps.msg
# a message definition containing many grasps
# identified by the grasp detection network.

Header header # sequence number, timestamp,frame ID
Grasp[] grasps # parametrizations of identified grasps
float32[] confs # N corresponding grasp confidences in [0, 1]