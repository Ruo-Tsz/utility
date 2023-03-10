#! /usr/bin/env python2
import rospy
import json
import numpy as np
import tf
import rosbag
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix
import os

# record localization in the bag


bags_path = '/data/itri_output/tracking_output/bags/intersection_bags/'
bag_path = bags_path + '2020-09-11-17-37-12_1.bag'
out_path = '/data/itri_output/tracking_output/tf_localization/2020-09-11-17-37-12_1.json'

# baselink_to_v = [x, y, z, w]
bl_v_qua = [0.000, 0.000, -0.013, 1.000]
(roll, pitch, yaw) = euler_from_quaternion (bl_v_qua)

T_b_v = tf.transformations.quaternion_matrix(
    np.array([ bl_v_qua[0], bl_v_qua[1], bl_v_qua[2], bl_v_qua[3] ]))
T_b_v[0, 3] = 0
T_b_v[1, 3] = 0
T_b_v[2, 3] = 0


bag = rosbag.Bag(bag_path)

localization_result = {}
baseLinkData = []
baseLinkTime = []
# retrieve all tf of baselink (velodyne to baselind is in static_tf)
for topic, msg, t in bag.read_messages(topics=['/tf']):
    for transf in msg.transforms:
        if (transf.header.frame_id == str('/map') and transf.child_frame_id == str('/base_link')):
            time = transf.header.stamp.secs * 10**(9) +  transf.header.stamp.nsecs
            x = transf.transform.translation.x
            y = transf.transform.translation.y
            z = transf.transform.translation.z
            global roll, pitch, yaw
            orientation_q = transf.transform.rotation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (roll, pitch, yaw) = euler_from_quaternion (orientation_list)
            baseLinkTime.append(time)
            baseLinkData.append([x, y, z, roll, pitch, yaw])
            
            T_m_b = tf.transformations.quaternion_matrix(
                np.array([ orientation_list[0], orientation_list[1], orientation_list[2], orientation_list[3] ]))
            T_m_b[0, 3] = x
            T_m_b[1, 3] = y
            T_m_b[2, 3] = z
            
            T_m_v = T_m_b.dot(T_b_v)

            location = {
                'header': {
                    'frame_id': '/map',
                    'child_frame_id': '/velodyne', 
                    'stamp': {
                        'nsecs': transf.header.stamp.nsecs, 
                        'secs': transf.header.stamp.secs}}, 
                'pose':{
                    'translation':{
                        'x': T_m_v[0, 3],
                        'y': T_m_v[1, 3],
                        'z': T_m_v[2, 3]},
                    'rotation':{
                        'x': quaternion_from_matrix(T_m_v)[0],
                        'y': quaternion_from_matrix(T_m_v)[1],
                        'z': quaternion_from_matrix(T_m_v)[2],
                        'w': quaternion_from_matrix(T_m_v)[3]}}
            }
            localization_result.update({time: location})


lidarTime = []
# retrieve all velodyne stamp
for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
    time = msg.header.stamp.secs * 10**(9) +  msg.header.stamp.nsecs
    # print(time)
    lidarTime.append(time)


print(len(localization_result))
print(len(lidarTime))

# search 
get_time = 0
for l_t in lidarTime:
    if l_t in localization_result.keys():
        get_time += 1

print(get_time)




ret = json.dumps(localization_result, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
with open(out_path, 'w') as fp:
    fp.write(ret)

