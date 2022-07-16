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
from threading import Thread


# follow the original result structure
# 'PATH_TO_ROOT_DIR_OF_RESULT/kuang-fu-rd/2020-09-11-17-31-33'

result_path = '/data/itri_output/tracking_output/kuang-fu-rd_livox_public/ego compensation/kuang-fu-rd_v3/2020-09-11-17-31-33/result/'
result_path = '/data/itri_output/tracking_output/output/livox_gt_annotate_velodyne_raw/ego_compensation/2020-09-11-17-31-33_sync_frame(v3)/result/'
viz_segment = '2020-09-11-17-31-33_7'
cloud_path = '/data/itri_output/tracking_output/output/livox_gt_annotate_velodyne_raw/ego_compensation/2020-09-11-17-31-33_sync_frame(v3)/pointcloud/'
# list of sorted string in micro-sec
result_stamp = []
result = {}

shut_down = False
block = False

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

        # T_v = np.linalg.inv(T_m_v).dot(T_m) 
        T_v = T_m 

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

def interrupt():
    global block
    while not shut_down: 
        # raw input would block and keep waiting(stall) for input
        raw_input('press any key to pause or resume:\n')
        block = not block
        print('pause' if block else 'resume')

    print('shut_down')

if __name__ == "__main__":    
    # rospy.init_node("visualize_node", anonymous=True)
    # mPubBoxes = rospy.Publisher('result_box', BoundingBoxArray, queue_size=100)
    # mPubScans = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    
    # load_result(os.path.join(result_path, 'result', viz_segment + '_ImmResult.json'))
    # scans_dict = load_pc(os.path.join(result_path, 'pointcloud', viz_segment))

    load_result(os.path.join(result_path, viz_segment + '_ImmResult.json'))
    scans_dict = load_pc(os.path.join(cloud_path, viz_segment))

    header = Header()
    header.frame_id = 'velodyne'

    lidar_matched = []
    result_matched = []
    for stamp in result_stamp:
        # pub based on result stamp
        if str(int(stamp)*1000) not in scans_dict.keys():
            print('no lidar: %lf', int(stamp/1e6))
        else:
            result_matched.append(str(int(stamp)*1000))
        
    for stamp in scans_dict.keys():
        if str(int(stamp)/1000) not in result_stamp:
            print('no result: %lf', int(stamp/1e9))
        else:
            lidar_matched.append(str(stamp))

    print('result matched: {}'.format(len(result_matched)))
    print('lidar matched: {}'.format(len(lidar_matched)))
    print('result total: {}'.format(len(result_stamp)))
    print('lidar total: {}'.format(len(scans_dict.keys())))
    print(viz_segment)
    
        