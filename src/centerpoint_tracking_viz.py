#! /usr/bin/python2
# It's for visualizing the detection/tracking result of CENTERPOINT at kuang-fu rd
import rospy
import json
import time
import os
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from tf import transformations
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header

# follow the original result structure
# 'PATH_TO_ROOT_DIR_OF_RESULT/kuang-fu-rd/2020-09-11-17-31-33'
result_path = '/home/user/repo/CenterPoint/itri/frame_num_4/tracking'
# list of sorted string in micro-sec
result_stamp = []
ego_pose = {}
result = {}
is_tracking = True


def load_result(result_file):
    global result, result_stamp, ego_pose
    with open (result_file, mode='r') as f:
        det_result = json.load(f)
    result = det_result['results']
    ego_pose = det_result['ego_poses']

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

def create_boxes_msg(objs_dict, header, stamp):
    """
        obj at map, but pc cloud at velodyne
    """
    obj_markers = BoundingBoxArray()
    obj_markers.header = header

    T_pose = np.eye(1)
    if ego_pose.has_key(str(stamp)):
        T_pose = transformations.quaternion_matrix(
            np.array([
                ego_pose[str(stamp)]['rotation'][0],
                ego_pose[str(stamp)]['rotation'][1],
                ego_pose[str(stamp)]['rotation'][2],
                ego_pose[str(stamp)]['rotation'][3]]))
        
        T_pose[0, 3] = ego_pose[str(stamp)]['translation'][0]
        T_pose[1, 3] = ego_pose[str(stamp)]['translation'][1]
        T_pose[2, 3] = ego_pose[str(stamp)]['translation'][2]
        print('get at {}'.format(T_pose[:3, 3]))
    else:
        print('not get pose {}'.format(stamp))

    for index, obj in enumerate(objs_dict):
        obj_marker = BoundingBox()
        obj_marker.header = header

        # global position
        T_m = transformations.quaternion_matrix(
            np.array([
                obj['rotation'][0],
                obj['rotation'][1],
                obj['rotation'][2],
                obj['rotation'][3]]))
    
        T_m[0, 3] = obj['translation'][0]
        T_m[1, 3] = obj['translation'][1]
        T_m[2, 3] = obj['translation'][2]

        # local
        # T_v = np.linalg.inv(T_pose).dot(T_m)
        T_v = T_m 

        obj_marker.label = int(obj['tracking_id']) if is_tracking else index
        obj_marker.pose.position.x = T_v[0, 3]
        obj_marker.pose.position.y = T_v[1, 3]
        obj_marker.pose.position.z = T_v[2, 3]
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_v)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_v)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_v)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_v)[3]
        obj_marker.dimensions.x = obj['size'][0]
        obj_marker.dimensions.y = obj['size'][1]
        obj_marker.dimensions.z = obj['size'][2]

        obj_markers.boxes.append(obj_marker)
        # print(obj_markers.boxes[-1].pose.position.x, obj_markers.boxes[-1].pose.position.y, obj_markers.boxes[-1].pose.position.z)

    return obj_markers

def create_vel(objs_dict, header, stamp):
    """
        pub global vel in global frame
    """
    vel_markers = MarkerArray()
    # delete last shadow first 
    marker = Marker()
    marker.id = 0
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    vel_markers.markers.append(marker)

    for index, obj in enumerate(objs_dict):
        vel_marker = Marker()
        vel_marker.header = header
        vel_marker.action = Marker.ADD
        vel_marker.ns = 'points_arrows'
        vel_marker.id = int(obj['tracking_id']) if is_tracking else index
        vel_marker.type = Marker.ARROW
        vel_marker.pose.orientation.y = 0
        vel_marker.pose.orientation.w = 1
        vel_marker.scale.x = 0.3
        vel_marker.scale.y = 1.0
        vel_marker.color.r = 1.0
        vel_marker.color.g = 0.0
        vel_marker.color.b = 1.0
        vel_marker.color.a = 1.0

        pt = Point()
        pt.x = obj['translation'][0]
        pt.y = obj['translation'][1]
        pt.z = obj['translation'][2]
        vel_marker.points.append(pt)

        # vel in global(map) frame
        pt = Point()
        pt.x = obj['translation'][0] + obj['velocity'][0]
        pt.y = obj['translation'][1] + obj['velocity'][1]
        pt.z = obj['translation'][2]
        vel_marker.points.append(pt)
        vel_markers.markers.append(vel_marker)
    return vel_markers


def create_pc(pc, header, stamp):
    fields = []
    # fill sensor_msg with density
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)]

    T_pose = np.eye(4)
    if ego_pose.has_key(str(stamp)):
        T_pose = transformations.quaternion_matrix(
            np.array([
                ego_pose[str(stamp)]['rotation'][0],
                ego_pose[str(stamp)]['rotation'][1],
                ego_pose[str(stamp)]['rotation'][2],
                ego_pose[str(stamp)]['rotation'][3]]))
        
        T_pose[0, 3] = ego_pose[str(stamp)]['translation'][0]
        T_pose[1, 3] = ego_pose[str(stamp)]['translation'][1]
        T_pose[2, 3] = ego_pose[str(stamp)]['translation'][2]

    # pc: N x 4 (x, y ,z , intensity)
    pc_local = np.hstack([pc[:, :3], np.ones([pc.shape[0], 1], dtype=np.float32)])
    global_pc = np.dot(T_pose, pc_local.T).T

    pc_msg = pcl2.create_cloud(header, fields, global_pc)
    return pc_msg


if __name__ == "__main__":    
    rospy.init_node("cp_visualize_node", anonymous=True)
    mPubBoxes = rospy.Publisher('result_box', BoundingBoxArray, queue_size=100)
    mPubScans = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    mPubVels = rospy.Publisher('vels', MarkerArray, queue_size=100)
    

    # load_result(os.path.join(result_path, 'tracking/tracking_result_wo_vel.json'))
    # load_result(os.path.join(result_path, 'tracking/tracking_result_with_vel.json'))
    load_result(os.path.join(result_path, 'tracking_result.json'))
    # load_result(os.path.join(result_path, 'det_output.json'))
    scans_dict = load_pc('/data/itri_output/tracking_output/pointcloud/2020-09-11-17-37-12/2020-09-11-17-37-12_4')

    header = Header()
    header.frame_id = 'map'

    while not rospy.is_shutdown():
        rate = rospy.Rate(10) 
        for stamp in result_stamp:
            if rospy.is_shutdown():
                break
            print(int(stamp))
            
            # pub based on result stamp
            if str(stamp) not in scans_dict.keys():
                print('skip at ', int(stamp))
                continue
            
            if len(result[stamp]) == 0:
                print('skip for no objects ')
                continue

            header.stamp.secs = int(int(stamp) / 1e9)
            header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3

            objs_dict = result[stamp]
            # print('Get {} objs at {}'.format(len(objs_dict['objects']), header))
            boxes_msg = create_boxes_msg(objs_dict, header, stamp)
            pc_msg = create_pc(scans_dict[str(int(stamp))], header, stamp)
            vel_msg = create_vel(objs_dict, header, stamp)

            # print('create {}'.format(len(boxes_msg.boxes)))
            mPubBoxes.publish(boxes_msg)
            mPubScans.publish(pc_msg)
            mPubVels.publish(vel_msg)
            
            rate.sleep()
        print('Restart')
        exit(-1)
        time.sleep(5)
        