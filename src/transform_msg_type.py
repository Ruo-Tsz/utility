# It's modifying the topic pub time to header time for syn all data to visualize
# It's usefull when you have markerarray but cannot pub/record online at the header timestamp because of markerarry doesn't have header 
import rosbag 
import os
import rospy
from itri_msgs.msg import DetectedObject, DetectedObjectArray
outPutFile = '/data/bags/2021-05-05-15-00-25-small_v2_tra.bag'
inPutFile = '/data/bags/2021-05-05-15-00-25-small_v2.bag'


def convertTotraOne(msg):
    new_msg = DetectedObjectArray()
    new_msg.sensor_type = msg.sensor_type
    new_msg.header = msg.header
    for obj in msg.objects:
        new_obj = DetectedObject()
        new_obj = obj
        new_obj.past_variance = []
        new_msg.objects.append(new_obj)
    return new_msg


with rosbag.Bag( outPutFile ,'w') as outbag:
    for topic, msg, t in rosbag.Bag( inPutFile ).read_messages():
        try:
            # not target topic
            if topic.find('/center_track/pylon_cameras/gige_26_0/vision_objects') == -1:
                outbag.write(topic, msg, msg.header.stamp)
                # print(topic, msg.header.stamp)
            else:
                new_msg = convertTotraOne(msg)
                outbag.write(topic, new_msg, new_msg.header.stamp)
                print(topic, new_msg.header.stamp)
        except Exception as e:
            # print(e)
            continue
        
