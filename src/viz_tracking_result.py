#! /usr/bin/python2
# It's for visualizing the tracking result of kuang-fu rd
import rospy
import json
import os
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from tf import transformations
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header

# follow the original result structure
# 'PATH_TO_ROOT_DIR_OF_RESULT/kuang-fu-rd/2020-09-11-17-31-33'
result_path = '/data/itri_output/tracking_output/kuang-fu-rd_v2/2020-09-11-17-31-33'
viz_segment = '2020-09-11-17-31-33_13'
# list of sorted string in micro-sec
result_stamp = []
result = {}


def load_result(result_file):
    global result, result_stamp
    with open (result_file, mode='r') as f:
        result = json.load(f)['frames']

    result_stamp = sorted(list(result.keys())) 



def load_pc(pc_path):
    files_list = sorted(os.listdir(pc_path))
    lidar_scans_list = [lidar_scan.split('.')[0] for lidar_scan in files_list]
    # print(lidar_scans_list)

    scans_dict = {}
    for index, f in enumerate(files_list):
        scanPath = os.path.join(pc_path, f)
        raw_scan = np.fromfile(scanPath, dtype=np.float32)
        # (N, 4)
        scan = raw_scan.reshape(-1, 4)
        scans_dict[lidar_scans_list[index]] = scan
    # print(scans_dict.keys())
    # print(len(scans_dict.keys()))

    return scans_dict

def create_boxes_msg(objs_dict, header):
    obj_markers = BoundingBoxArray()
    obj_markers.header = header

    T_m_v = transformations.quaternion_matrix(
        np.array([
            objs_dict['pose']['rotation']['x'],
            objs_dict['pose']['rotation']['y'],
            objs_dict['pose']['rotation']['z'],
            objs_dict['pose']['rotation']['w']]))
    
    T_m_v[0, 3] = objs_dict['pose']['position']['x']
    T_m_v[1, 3] = objs_dict['pose']['position']['y']
    T_m_v[2, 3] = objs_dict['pose']['position']['z']

    for obj in objs_dict['objects']:
        obj_marker = BoundingBox()
        obj_marker.header = header

        T_m = transformations.quaternion_matrix(
            np.array([
                obj['rotation']['x'],
                obj['rotation']['y'],
                obj['rotation']['z'],
                obj['rotation']['w']]))
    
        T_m[0, 3] = obj['translation']['x']
        T_m[1, 3] = obj['translation']['y']
        T_m[2, 3] = obj['translation']['z']

        T_v = np.linalg.inv(T_m_v).dot(T_m) 

        obj_marker.label = int(obj['tracking_id'])
        obj_marker.pose.position.x = T_v[0, 3]
        obj_marker.pose.position.y = T_v[1, 3]
        obj_marker.pose.position.z = T_v[2, 3]
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_v)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_v)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_v)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_v)[3]
        obj_marker.dimensions.x = obj['size']['l']
        obj_marker.dimensions.y = obj['size']['w']
        obj_marker.dimensions.z = obj['size']['h']

        obj_markers.boxes.append(obj_marker)
        # print(obj_markers.boxes[-1].pose.position.x, obj_markers.boxes[-1].pose.position.y, obj_markers.boxes[-1].pose.position.z)

    return obj_markers


def create_pc(pc, header):
    fields = []
    # fill sensor_msg with density
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)]

    pc_msg = pcl2.create_cloud(header, fields, pc)
    return pc_msg


if __name__ == "__main__":    
    rospy.init_node("visualize_node", anonymous=True)
    mPubBoxes = rospy.Publisher('result_box', BoundingBoxArray, queue_size=100)
    mPubScans = rospy.Publisher('velodynes_points', PointCloud2, queue_size=100)
    
    load_result(os.path.join(result_path, 'result', viz_segment + '_ImmResult.json'))
    scans_dict = load_pc(os.path.join(result_path, 'pointcloud', viz_segment))

    header = Header()
    header.frame_id = 'velodyne'

    while not rospy.is_shutdown():
        rate = rospy.Rate(10) 
        for stamp in result_stamp:
            # print(int(stamp)*1000)
            
            # pub based on result stamp
            if str(int(stamp)*1000) not in scans_dict.keys():
                print('skip at ', int(stamp)*1000)
                continue
            
            if len(result[stamp]['objects']) == 0:
                print('skip for no objects ')
                continue

            header.stamp.secs = int(int(stamp)*1e3 / 1e9)
            header.stamp.nsecs= int(stamp)*1e3 % 1e9

            objs_dict = result[stamp]
            # print('Get {} objs at {}'.format(len(objs_dict['objects']), header))
            boxes_msg = create_boxes_msg(objs_dict, header)
            pc_msg = create_pc(scans_dict[str(int(stamp)*1000)], header)

            # print('create {}'.format(len(boxes_msg.boxes)))
            mPubBoxes.publish(boxes_msg)
            mPubScans.publish(pc_msg)
            
            rate.sleep()
        