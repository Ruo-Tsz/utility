# It's modifying the topic pub time to header time for syn all data to visualize
# It's usefull when you have markerarray but cannot pub/record online at the header timestamp because of markerarry doesn't have header 
import rosbag 
import os
outPutFile = '/media/ee904/Samsung_T5/bags/20200911-nctu/2020-09-11-17-37-12_det_livox_all_sync.bag'
inPutFile = '/media/ee904/Samsung_T5/bags/20200911-nctu/2020-09-11-17-37-12_det_livox_all.bag'


with rosbag.Bag( outPutFile ,'w') as outbag:
    for topic, msg, t in rosbag.Bag( inPutFile ).read_messages():
        try:
            if topic.find('pointcloud') != -1:
                outbag.write(topic, msg, msg.header.stamp)
                print(topic, msg.header.stamp)
            else:
                outbag.write(topic, msg, msg.markers[0].header.stamp)
                print(topic, msg.markers[0].header.stamp)
        except Exception as e:
            print(e)
            continue
        
