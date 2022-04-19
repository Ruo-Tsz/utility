#! /usr/bin/python2
# It's for visualizing the tracking result of kuang-fu rd
import rospy
import json
import os, sys
import time
from threading import Thread
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from tf import transformations
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header

# follow the original result structure
# 'PATH_TO_ROOT_DIR_OF_RESULT/kuang-fu-rd/2020-09-11-17-31-33'
result_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/2022-04-18_01-35-24'
result_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/v2/2022-04-18_04-23-25'
viz_segment = '2020-09-11-17-37-12_4'

gt = {}
det = {}
fn_gt = {}
fp_hyp = {}
gt_stamp = []
scans_dict = {}

block = False

# mydata = raw_input('Prompt :')
# print (mydata)
# if mydata =='p':
#     print('p')
# exit(-1)

def load_result(result_file):
    with open (result_file, mode='r') as f:
        result = json.load(f)
    return result

def load_details(detail_file):
    with open (detail_file, mode='r') as f:
        details = json.load(f)
        fn_gt = details['fn_gt']
        fp_hyp = details['fp_hypotheses']
    return fn_gt, fp_hyp

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

    for obj in objs_dict:
        obj_marker = BoundingBox()
        obj_marker.header = header

        T_m = transformations.quaternion_matrix(
            np.array([
                obj['track']['rotation']['x'],
                obj['track']['rotation']['y'],
                obj['track']['rotation']['z'],
                obj['track']['rotation']['w']]))
    
        T_m[0, 3] = obj['track']['translation']['x']
        T_m[1, 3] = obj['track']['translation']['y']
        T_m[2, 3] = obj['track']['translation']['z']

        # T_v = np.linalg.inv(T_m_v).dot(T_m) 
        T_v = T_m 

        obj_marker.label = int(obj['id'])
        obj_marker.pose.position.x = T_v[0, 3]
        obj_marker.pose.position.y = T_v[1, 3]
        obj_marker.pose.position.z = T_v[2, 3]
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_v)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_v)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_v)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_v)[3]
        obj_marker.dimensions.x = obj['track']['box']['length']
        obj_marker.dimensions.y = obj['track']['box']['width']
        obj_marker.dimensions.z = obj['track']['box']['height']

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
    while not rospy.is_shutdown(): 
        # raw input would block and keep waiting for input
        raw_input('press any key to pause or resume:\n')
        block = not block
        print('pause' if block else 'resume')

    print('End interrupt')
    sys.exit()

if __name__ == "__main__":    
    rospy.init_node("visualize_node", anonymous=True)
    mPubBoxes = rospy.Publisher('result_box', BoundingBoxArray, queue_size=100)
    mPubGTBoxes = rospy.Publisher('gt_box', BoundingBoxArray, queue_size=100)
    mPubFNBoxes = rospy.Publisher('fn_box', BoundingBoxArray, queue_size=100)
    mPubFPBoxes = rospy.Publisher('fp_box', BoundingBoxArray, queue_size=100)
    mPubScans = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    
    gt = load_result(os.path.join(result_path, 'gt.json'))
    det = load_result(os.path.join(result_path, 'det.json'))
    fn_gt, fp_hyp = load_details(os.path.join(result_path, 'details.json'))

    print('load: {} det and {} gt'.format(len(det.keys()), len(gt.keys())))

    gt_stamp = sorted(list(gt.keys())) 
    scans_dict = load_pc(os.path.join('/data/itri_output/tracking_output', 'pointcloud/2020-09-11-17-37-12', viz_segment))

    header = Header()
    header.frame_id = 'velodyne'

    # keep listening to keyboard to pause or resume 
    thread = Thread(target = interrupt, args = [])
    thread.start()

    while not rospy.is_shutdown():
        rate = rospy.Rate(1) 
        for stamp in gt_stamp:
            if rospy.is_shutdown():
                break

            # pub based on gt stamp
            if str(int(stamp)) not in scans_dict.keys():
                print('skip at ', int(stamp))
                continue

            header.stamp.secs = int(int(stamp) / 1e9)
            header.stamp.nsecs= int(stamp) % 1e9

            objs_dict = det[stamp] if det.has_key(stamp) else []
            gt_dict = gt[stamp]
            fn_dict = fn_gt[stamp] if fn_gt.has_key(stamp) else []
            fp_dict = fp_hyp[stamp] if fp_hyp.has_key(stamp) else []
            # print('Get {} objs at {}'.format(len(objs_dict['objects']), header))
            det_boxes_msg = create_boxes_msg(objs_dict, header)
            gt_boxes_msg = create_boxes_msg(gt_dict, header)
            fn_boxes_msg = create_boxes_msg(fn_dict, header)
            fp_boxes_msg = create_boxes_msg(fp_dict, header)
            pc_msg = create_pc(scans_dict[str(int(stamp))], header)

            while block:
                # print('now block..')
                pass

            print('create det: {}; gt: {}'.format(len(det_boxes_msg.boxes), len(gt_boxes_msg.boxes)))
            mPubBoxes.publish(det_boxes_msg)
            mPubGTBoxes.publish(gt_boxes_msg)
            mPubScans.publish(pc_msg)
            mPubFNBoxes.publish(fn_boxes_msg)
            mPubFPBoxes.publish(fp_boxes_msg)
            
            rate.sleep()

    print('Out loop, thread.is_alive: {}, enter any key to stop program thread'.format(thread.is_alive()))
    