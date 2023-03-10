#! /usr/bin/python2
# It's for visualizing the tracking result of kuang-fu rd
import rospy
import json
import os, sys
import time
import copy
from threading import Thread
from tf import transformations
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header
import cv2

# set resolution to image, scale mRangeResol to pixel_scale times, 1 pixel = 1/pixel_scale = 0.15 m
pixel_resol = 0.15
pixel_resol = 0.05
pixel_scale = 1/pixel_resol
img_size = 900
img = np.full((img_size, img_size, 3), 255, np.uint8)

rospy.set_param('use_sim_time', False)
play_rate = rospy.get_param('play_rate', 1)
shut_down = False
block = False
show_2D = True

gt = {}
det = {}
fn_gt = {}
fp_hyp = {}
gt_stamp = []
scans_dict = {}

gt_traj = {}
gt_frag_traj = {}
# {"gt1": {
#           t1: {'tp': {'id': , 'pt': tp1}, gt: {'id': , 'pt': gt1}}, t2: {tp: tp2, gt: gt2}
# }
id_color_map = {'det': [1, 1, 0], 'gt': [0, 1, 0], 'fp':[1, 0, 0], 'fn':[0, 0, 1]}

rl_markers = MarkerArray()

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
        if show_2D:
            scan[:,2] = np.zeros(scan.shape[0])
        scans_dict[lidar_scans_list[index]] = scan
    # print(scans_dict.keys())
    # print(len(scans_dict.keys()))

    return scans_dict

def create_boxes_msg(objs_dict, header, msg_type, focus_id=None):
    obj_markers = BoundingBoxArray()
    obj_markers.header = header

    id_markers = MarkerArray()
    # delete last shadow first 
    marker = Marker()
    marker.id = 0
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    id_markers.markers.append(marker)

    scale_param = 1

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
        if not show_2D:
            obj_marker.pose.position.z = T_v[2, 3]
        else:
            obj_marker.pose.position.z = 0
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_v)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_v)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_v)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_v)[3]
        obj_marker.dimensions.x = obj['track']['box']['length'] *scale_param
        obj_marker.dimensions.y = obj['track']['box']['width'] *scale_param
        if not show_2D:
            obj_marker.dimensions.z = obj['track']['box']['height'] *scale_param
        else:
            obj_marker.dimensions.z = 0.001

        obj_markers.boxes.append(obj_marker)
        # print(obj_markers.boxes[-1].pose.position.x, obj_markers.boxes[-1].pose.position.y, obj_markers.boxes[-1].pose.position.z)

        id_marker = Marker()
        id_marker.header = header
        id_marker.action = Marker.ADD
        id_marker.ns = msg_type + '_id'
        id_marker.type = Marker.TEXT_VIEW_FACING
        # id_marker.lifetime = rospy.Duration(1/play_rate)
 
        id_marker.color.r = id_color_map[msg_type][0] if id_color_map.has_key(msg_type) else 1
        id_marker.color.g = id_color_map[msg_type][1] if id_color_map.has_key(msg_type) else 1
        id_marker.color.b = id_color_map[msg_type][2] if id_color_map.has_key(msg_type) else 1
        id_marker.color.a = 1
        id_marker.scale.z = 1.2
        id_marker.id = int(obj['id'])
        id_marker.text = str((obj['id'])) 
        id_marker.pose.position.x = T_v[0, 3]
        id_marker.pose.position.y = T_v[1, 3]
        id_marker.pose.position.z = T_v[2, 3] + 2

        if focus_id != None and int(obj['id'] not in focus_id):
            continue
        # else:
        #     id_marker.pose.position.x = T_v[0, 3] + 0.5
        #     id_marker.pose.position.y = T_v[1, 3] - 1
        #     id_marker.pose.position.z = T_v[2, 3] + 3
        id_markers.markers.append(id_marker)

    return obj_markers, id_markers


def create_pc(pc, header):
    fields = []
    # fill sensor_msg with density
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)]

    pc_msg = pcl2.create_cloud(header, fields, pc)
    return pc_msg

def create_sw_msg(objs_dict, stamp, sw_id):
    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'
    obj_markers = BoundingBoxArray()
    obj_markers.header = header
    sw_id_list = map(int, sw_id.keys())

    id_markers = MarkerArray()
    
    # delete last shadow first 
    marker = Marker()
    marker.id = 0
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    id_markers.markers.append(marker)

    duration_time = 0.5
    for idx, obj in enumerate(objs_dict):
        if  obj['id'] not in sw_id_list:
            continue

        hyp_id = sw_id[str(obj['id'])]['history'][str(stamp)]

        # just show during duration_num [sec] before and after happening moment
        during_sw = False
        time_to_sw = False
        for m in sw_id[str(obj['id'])]['happened']:
            if (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) <= m) or \
                (abs(int(stamp) - m)/1e9 < 0.2 and int(stamp) > m):
                during_sw = True

                if int(stamp) == m:
                    time_to_sw = True
                break
        if not during_sw:
            continue

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
        T_v = T_m 

        if time_to_sw:
            obj_marker.label = 1
        else:
            obj_marker.label = 10000000
        # show tp association detection
        # obj_marker.label = int(hyp_id) if np.isfinite(hyp_id) else 1000000
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

        id_marker = Marker()
        id_marker.header = header
        id_marker.action = Marker.ADD
        id_marker.ns = 'sw_id'
        id_marker.type = Marker.TEXT_VIEW_FACING
        id_marker.lifetime = rospy.Duration(1/play_rate)

        id_marker.color.r = 0
        id_marker.color.g = 1
        id_marker.color.b = 0
        id_marker.color.a = 1
        id_marker.scale.z = 1
        id_marker.id = idx
        id_marker.text = str(hyp_id) 
        id_marker.pose.position.x = T_v[0, 3]
        id_marker.pose.position.y = T_v[1, 3]
        id_marker.pose.position.z = T_v[2, 3] + 1
        id_markers.markers.append(id_marker)

    return obj_markers, id_markers

def create_tra_msg(objs_dict, dets_dict, stamp, tp_id_history):
    global gt_traj, det_traj
    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'

    boxes = BoundingBoxArray()
    # boxes.header = header

    tras_markers = MarkerArray()
    # delete last shadow first 
    marker = Marker()
    marker.id = 1000
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    tras_markers.markers.append(marker)

    duration_time = 2
    # seg 4
    # focus_gt_id = 2
    # focus_gt_id = 197

    # seg 1
    focus_gt_id = 484
    # pub gt tra
    for idx, obj in enumerate(objs_dict):
        if not (obj['id'] == focus_gt_id):
            continue

        # gt trajectory
        gt_tras = Marker()
        gt_tras.header = header
        gt_tras.ns = 'gt_tras_' + str(obj['id'])
        gt_tras.action = Marker.ADD
        gt_tras.type = Marker.LINE_STRIP
        gt_tras.id = int(obj['id'])
        gt_tras.scale.x = 0.25
        gt_tras.color.a = 1
        gt_tras.color.g = 1

        T_gt = transformations.quaternion_matrix(
            np.array([
                obj['track']['rotation']['x'],
                obj['track']['rotation']['y'],
                obj['track']['rotation']['z'],
                obj['track']['rotation']['w']]))
        T_gt[0, 3] = obj['track']['translation']['x']
        T_gt[1, 3] = obj['track']['translation']['y']
        T_gt[2, 3] = obj['track']['translation']['z']
        pt = Point(T_gt[0, 3], T_gt[1, 3], T_gt[2, 3])

        if not gt_traj.has_key(obj['id']):
            gt_traj[obj['id']] = {}
        single_frame_dict = {'tp': {'id': -1, 'pt': np.nan}, 'gt': {'id': obj['id'], 'pt': pt}}
        gt_traj[obj['id']][stamp] = single_frame_dict

        # pub gt from now on
        for k, v in sorted(gt_traj[obj['id']].items()):
            if abs(int(stamp) - int(k))/1e9 < duration_time:
                gt_tras.points.append(v['gt']['pt'])
                if show_2D:
                    gt_tras.points[-1].z = 0
        tras_markers.markers.append(gt_tras)

        # register corresponding tp
        tp_det_id = tp_id_history[str(obj['id'])]['id'][str(stamp)]
        print('tp for {}: {}'.format(obj['id'], tp_det_id) )
        for det in dets_dict:
            if det['id'] != tp_det_id:
                continue

            T_obj = transformations.quaternion_matrix(
                np.array([
                    det['track']['rotation']['x'],
                    det['track']['rotation']['y'],
                    det['track']['rotation']['z'],
                    det['track']['rotation']['w']]))
        
            T_obj[0, 3] = det['track']['translation']['x']
            T_obj[1, 3] = det['track']['translation']['y']
            T_obj[2, 3] = det['track']['translation']['z']

            # register tp pos
            pt = Point(T_obj[0, 3], T_obj[1, 3], T_obj[2, 3]+1)
            gt_traj[obj['id']][stamp]['tp']['id'] = det['id']
            gt_traj[obj['id']][stamp]['tp']['pt'] = pt
            # print(gt_traj[obj['id']][stamp]['tp']['id'], ": ", T_obj[0, 3], T_obj[1, 3], T_obj[2, 3]+1)

            # tp_marker = BoundingBox()
            # tp_marker.header = header
            # tp_marker.label = int(det['id'])
            # tp_marker.pose.position.x = T_obj[0, 3]
            # tp_marker.pose.position.y = T_obj[1, 3]
            # if not show_2D:
            #     tp_marker.pose.position.z = T_obj[2, 3]
            # else:
            #     tp_marker.pose.position.z = 0
            # tp_marker.pose.orientation.x = quaternion_from_matrix(T_obj)[0]
            # tp_marker.pose.orientation.y = quaternion_from_matrix(T_obj)[1]
            # tp_marker.pose.orientation.z = quaternion_from_matrix(T_obj)[2]
            # tp_marker.pose.orientation.w = quaternion_from_matrix(T_obj)[3]
            # tp_marker.dimensions.x = det['track']['box']['length']
            # tp_marker.dimensions.y = det['track']['box']['width']
            # if not show_2D:
            #     tp_marker.dimensions.z = det['track']['box']['height']
            # else:
            #     tp_marker.dimensions.z = 0.001
            # boxes.boxes.append(tp_marker)

            obj_box = Marker()
            obj_box.header = header
            obj_box.ns = 'obj_box_' + str(det['id'])
            obj_box.action = Marker.ADD
            obj_box.type = Marker.LINE_STRIP
            obj_box.id = int(det['id'])
            obj_box.scale.x = 0.15
            obj_box.color.r = 1
            obj_box.color.b = 1
            obj_box.color.a = 1
            l = det['track']['box']['length']
            w = det['track']['box']['width']
            lengths = [l/2, -l/2, -l/2, l/2]
            widths = [w/2, w/2, -w/2, -w/2]
            corners = []
            for idx, pt in enumerate(lengths):
                corner = np.array([lengths[idx], widths[idx], 0, 1])
                pos_corner = T_obj.dot(corner)
                # print('det: {}, {}, {}'.format(pos_corner[0], pos_corner[1], pos_corner[2]))
                corners.append(Point(pos_corner[0], pos_corner[1], pos_corner[2]))

            for pt in corners:
                obj_box.points.append(pt)
            obj_box.points.append(corners[0])
            tras_markers.markers.append(obj_box)
            break


        # pub all tp det from now on
        traverse_tp_id = []
        markers = []
        for k, v in sorted(gt_traj[obj['id']].items()):
            if abs(int(stamp) - int(k))/1e9 < duration_time:
                if v['tp']['id'] == -1:
                    continue
                
                # register differ tra of tp det
                if v['tp']['id'] not in traverse_tp_id:
                    det_tras = Marker()
                    det_tras.header = header
                    det_tras.ns = 'det_tras_' + str(v['tp']['id'])
                    det_tras.action = Marker.ADD
                    det_tras.type = Marker.LINE_STRIP
                    det_tras.id = int(v['tp']['id'])
                    det_tras.scale.x = 0.2
                    det_tras.color.a = 1
                    det_tras.color.r = 1

                    if v['tp']['id'] == 412:
                        det_tras.color.r = 0
                        det_tras.color.b = 1
                    elif v['tp']['id']%4 == 1:
                        det_tras.color.r = 0
                        det_tras.color.b = 1
                    elif v['tp']['id']%4 == 2:
                        det_tras.color.g = 1
                    elif v['tp']['id'] == 764:
                        det_tras.color.r = 1
                        det_tras.color.g = 1

                    det_tras.points.append(v['tp']['pt'])
                    if show_2D:
                        det_tras.points[-1].z = 0.01
                    markers.append(det_tras)
                    traverse_tp_id.append(v['tp']['id'])
                else:
                    idx = traverse_tp_id.index(v['tp']['id'])
                    markers[idx].points.append(v['tp']['pt'])
                    if show_2D:
                        markers[idx].points[-1].z = 0.05
        # print('traverse_tp_id:', traverse_tp_id)
        # print('markers:', len(markers))
        for m in markers:
            tras_markers.markers.append(m)

        # obj_marker = BoundingBox()
        # obj_marker.header = header
        # obj_marker.label = int(obj['id'])
        # obj_marker.pose.position.x = T_gt[0, 3]
        # obj_marker.pose.position.y = T_gt[1, 3]
        # if not show_2D:
        #     obj_marker.pose.position.z = T_gt[2, 3]
        # else:
        #     obj_marker.pose.position.z = 0
        # obj_marker.pose.orientation.x = quaternion_from_matrix(T_gt)[0]
        # obj_marker.pose.orientation.y = quaternion_from_matrix(T_gt)[1]
        # obj_marker.pose.orientation.z = quaternion_from_matrix(T_gt)[2]
        # obj_marker.pose.orientation.w = quaternion_from_matrix(T_gt)[3]
        # obj_marker.dimensions.x = obj['track']['box']['length']
        # obj_marker.dimensions.y = obj['track']['box']['width']
        # if not show_2D:
        #     obj_marker.dimensions.z = obj['track']['box']['height']
        # else:
        #     obj_marker.dimensions.z = 0.001
        # boxes.boxes.append(obj_marker)

        gt_box = Marker()
        gt_box.header = header
        gt_box.ns = 'gt_box_' + str(obj['id'])
        gt_box.action = Marker.ADD
        gt_box.type = Marker.LINE_STRIP
        gt_box.id = int(obj['id'])
        gt_box.scale.x = 0.15
        gt_box.color.g = 1
        gt_box.color.a = 1
        l = obj['track']['box']['length']
        w = obj['track']['box']['width']
        lengths = [l/2, -l/2, -l/2, l/2]
        widths = [w/2, w/2, -w/2, -w/2]
        corners = []
        for idx, pt in enumerate(lengths):
            corner = np.array([lengths[idx], widths[idx], 0, 1])
            pos_corner = T_gt.dot(corner)
            corners.append(Point(pos_corner[0], pos_corner[1], pos_corner[2]))

        # for idx, pt in enumerate(corners):
        for pt in corners:
            gt_box.points.append(pt)
            # pt_msg = Point(corners[(idx+1)%len(corners)][0, 0], corners[(idx+1)%len(corners)][0, 1], corners[(idx+1)%len(corners)][0, 2])
            # gt_box.points.append(pt_msg)
        gt_box.points.append(corners[0])
        tras_markers.markers.append(gt_box)
        
    return tras_markers, boxes

    

def create_frag_msg(objs_dict, dets_dict, stamp, tp_id_history, frag_id):
    global gt_frag_traj, det_traj
    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'
    frag_id_list = [int(id) for id in frag_id.keys() if frag_id[id]['frag_num'] > 0]

    duration_time = 2
    frag_gt_markers = BoundingBoxArray()
    frag_gt_markers.header = header
    frag_det_markers = BoundingBoxArray()
    frag_det_markers.header = header
    frag_tras = MarkerArray()

    # delete last shadow first 
    marker = Marker()
    marker.id = 1000
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    frag_tras.markers.append(marker)

    for idx, obj in enumerate(objs_dict):
        if  obj['id'] not in frag_id_list:
            continue

        # if not (obj['id'] == 5 or obj['id'] == 2):
        # if not (obj['id'] == 5 ):
        # 1599817098792594000
        # if not (obj['id'] == 2 ):
        #     continue
        
        # during_frag = False
        # for m in frag_id[str(obj['id'])]['happened']:
        #     # if (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) <= int(m)) or \
        #     #     (abs(int(stamp) - int(m))/1e9 < 0.5 and int(stamp) > int(m)):
        #     if (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) <= int(m)) or \
        #         (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) > int(m)):
        #         during_frag = True
        #         # print('happened time: {}'.format(m))
        #         break

        # find duration of frag which is untracked duration (happened is the first frame of untracked)
        sort_t = sorted(tp_id_history[str(obj['id'])]['matched'])
        sort_matched = []
        for t in sort_t:
            sort_matched.append(tp_id_history[str(obj['id'])]['matched'][t])
    
        in_frag = False
        first_track = -1
        last_track = len(sort_matched)
        counted_flag = False
        for idx, tracked in enumerate(sort_matched):
            if tracked:
                first_track = idx
                break
        
        for idx, tracked in reversed(list(enumerate(sort_matched))):
            if tracked:
                last_track = idx
                break
        duration_ts = []
        duration_t = []

        # make sure would be tracked at least 1 frame
        if not(first_track == -1 and last_track == len(sort_matched)):
            for idx, m in enumerate(sort_matched):
                if idx < first_track:
                    continue

                # close duration
                if in_frag and m:
                    duration_ts.append(duration_t)
                    duration_t = []
                    in_frag = False
                # in duration
                elif in_frag and not m:
                    duration_t.append(sort_t[idx])
                # new duration
                if not m and not in_frag:
                    duration_t.append(sort_t[idx])
                    in_frag = True

                if idx > last_track:
                    break
    
        during_frag = False
        for frag_times in duration_ts:
            for m in frag_times:
                if (abs(int(stamp) - int(m))/1e9 < 0.5 and int(stamp) <= int(m)) or \
                (abs(int(stamp) - int(m))/1e9 < 0.2 and int(stamp) > int(m)):
                    during_frag = True
                    break
        
        if not during_frag:
            continue

        # gt box
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

        # 1000 -> not tracked as blue
        obj_marker.label = obj['id']
        obj_marker.pose.position.x = T_m[0, 3]
        obj_marker.pose.position.y = T_m[1, 3]
        if not show_2D:
            obj_marker.pose.position.z = T_m[2, 3]
        else:
            obj_marker.pose.position.z = 0
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
        obj_marker.dimensions.x = obj['track']['box']['length'] * 1.5
        obj_marker.dimensions.y = obj['track']['box']['width'] * 1.5
        if not show_2D:
            obj_marker.dimensions.z = obj['track']['box']['height']
        else:
            obj_marker.dimensions.z = 0.001
        frag_gt_markers.boxes.append(obj_marker)

        # gt trajectory
        frag_gt_tras = Marker()
        frag_gt_tras.header = header
        frag_gt_tras.ns = 'frag_gt_tras'
        frag_gt_tras.action = Marker.ADD
        frag_gt_tras.type = Marker.LINE_STRIP
        frag_gt_tras.id = int(obj['id'])
        frag_gt_tras.scale.x = 0.3
        frag_gt_tras.color.a = 1
        frag_gt_tras.color.g = 1
        pt = Point(T_m[0, 3], T_m[1, 3], T_m[2, 3])

        if not gt_frag_traj.has_key(obj['id']):
            gt_frag_traj[obj['id']] = {}

        single_frame_dict = {'tp': {'id': -1, 'pt': np.nan}, 'gt': {'id': obj['id'], 'pt': pt}}
        gt_frag_traj[obj['id']][stamp] = single_frame_dict

        current_tp_ids = []
        for k, v in sorted(gt_frag_traj[obj['id']].items()):
            if gt_frag_traj[obj['id']][k]['tp']['id'] not in current_tp_ids and gt_frag_traj[obj['id']][k]['tp']['id'] > 0:
                current_tp_ids.append(gt_frag_traj[obj['id']][k]['tp']['id'])
        # print('current_tp_ids:', current_tp_ids)
    
        # search current tp
        # dets_id_list = list(set(list(tp_id_history[str(obj['id'])]['id'].values())))
        tp_det_id = tp_id_history[str(obj['id'])]['id'][str(stamp)]
        # print(str(obj['id']), tp_det_id)
        for det in dets_dict:
            if det['id'] != tp_det_id:
                continue

            det_marker = BoundingBox()
            det_marker.header = header

            T_m = transformations.quaternion_matrix(
                np.array([
                    det['track']['rotation']['x'],
                    det['track']['rotation']['y'],
                    det['track']['rotation']['z'],
                    det['track']['rotation']['w']]))
        
            T_m[0, 3] = det['track']['translation']['x']
            T_m[1, 3] = det['track']['translation']['y']
            T_m[2, 3] = det['track']['translation']['z']

            # 1000 -> not tracked as blue
            det_marker.label = obj['id']
            det_marker.pose.position.x = T_m[0, 3]
            det_marker.pose.position.y = T_m[1, 3]
            if not show_2D:
                det_marker.pose.position.z = T_m[2, 3]
            else:
                det_marker.pose.position.z = 0
            det_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
            det_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
            det_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
            det_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
            det_marker.dimensions.x = det['track']['box']['length']
            det_marker.dimensions.y = det['track']['box']['width']
            if not show_2D:
                det_marker.dimensions.z = det['track']['box']['height']
            else:
                det_marker.dimensions.z = 0.001
            frag_det_markers.boxes.append(det_marker)

            # register tp pos
            pt = Point(T_m[0, 3], T_m[1, 3], T_m[2, 3]+1)
            gt_frag_traj[obj['id']][stamp]['tp']['id'] = det['id']
            gt_frag_traj[obj['id']][stamp]['tp']['pt'] = pt
            # print(gt_frag_traj[obj['id']][stamp]['tp']['id'], ": ", T_m[0, 3], T_m[1, 3], T_m[2, 3]+1)

        # pub gt from now on
        for k, v in sorted(gt_frag_traj[obj['id']].items()):
            # print(k)
            frag_gt_tras.points.append(v['gt']['pt'])
        frag_tras.markers.append(frag_gt_tras)

        # pub all tp det from now on
        traverse_tp_id = []
        markers = []
        for k, v in sorted(gt_frag_traj[obj['id']].items()):
            if v['tp']['id'] == -1:
                continue
            
            # register differ tra of tp det
            if v['tp']['id'] not in traverse_tp_id:
                frag_det_tras = Marker()
                frag_det_tras.header = header
                frag_det_tras.ns = 'frag_det_tras'
                frag_det_tras.action = Marker.ADD
                frag_det_tras.type = Marker.LINE_STRIP
                frag_det_tras.id = int(v['tp']['id'])
                frag_det_tras.scale.x = 0.2
                frag_det_tras.color.a = 1
                frag_det_tras.color.r = 1

                if v['tp']['id']%4 == 1:
                    # frag_det_tras.color.g = 1
                    frag_det_tras.color.b = 1
                elif v['tp']['id']%4 == 2:
                    frag_det_tras.color.r = 0
                    frag_det_tras.color.b = 1
                elif v['tp']['id']%4 == 3:
                    frag_det_tras.color.g = 1

                frag_det_tras.points.append(v['tp']['pt'])
                markers.append(frag_det_tras)
                traverse_tp_id.append(v['tp']['id'])
            else:
                idx = traverse_tp_id.index(v['tp']['id'])
                markers[idx].points.append(v['tp']['pt'])
        # print('traverse_tp_id:', traverse_tp_id)
        # print('markers:', len(markers))
        for m in markers:
            frag_tras.markers.append(m)

        # print(gt_frag_traj[obj['id']][stamp]['tp']['id'])

    return frag_gt_markers, frag_det_markers, frag_tras

def create_over_msg(objs_dict, over_seg_dict, stamp):
    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'
    obj_markers = BoundingBoxArray()
    obj_markers.header = header

    over_det_markers = BoundingBoxArray()
    over_det_markers.header = header

    if stamp not in over_seg_dict.keys():
        return obj_markers, over_det_markers
    
    gt_id_list = over_seg_dict[stamp].keys()
    for idx, obj in enumerate(objs_dict):
        if  obj['id'] not in gt_id_list:
            continue
        
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

        obj_marker.label = obj['id']
        obj_marker.pose.position.x = T_m[0, 3]
        obj_marker.pose.position.y = T_m[1, 3]
        obj_marker.pose.position.z = T_m[2, 3]
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
        obj_marker.dimensions.x = obj['track']['box']['length']
        obj_marker.dimensions.y = obj['track']['box']['width']
        obj_marker.dimensions.z = obj['track']['box']['height']
        obj_markers.boxes.append(obj_marker)

        assert len(over_seg_dict[stamp][obj['id']]) > 1, 'error over-seg for gt {} with {} det'.format(obj['id'], len(over_seg_dict[stamp][obj['id']]))
        for det in over_seg_dict[stamp][obj['id']]:
            det_marker = BoundingBox()
            det_marker.header = header

            T_m = transformations.quaternion_matrix(
                np.array([
                    det['track']['rotation']['x'],
                    det['track']['rotation']['y'],
                    det['track']['rotation']['z'],
                    det['track']['rotation']['w']]))
        
            T_m[0, 3] = det['track']['translation']['x']
            T_m[1, 3] = det['track']['translation']['y']
            T_m[2, 3] = det['track']['translation']['z']

            det_marker.label = obj['id']
            det_marker.pose.position.x = T_m[0, 3]
            det_marker.pose.position.y = T_m[1, 3]
            det_marker.pose.position.z = T_m[2, 3]
            det_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
            det_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
            det_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
            det_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
            det_marker.dimensions.x = det['track']['box']['length']
            det_marker.dimensions.y = det['track']['box']['width']
            det_marker.dimensions.z = det['track']['box']['height']
            over_det_markers.boxes.append(det_marker)

    return obj_markers, over_det_markers

def create_occluded_boxes(gt_dict, header):
    obj_markers = BoundingBoxArray()
    obj_markers.header = header

    id_markers = MarkerArray()
    # delete last shadow first 
    marker = Marker()
    marker.id = 0
    marker.ns = 'delete'
    marker.action = Marker.DELETEALL
    id_markers.markers.append(marker)

    for obj in gt_dict:
        occluded_flag = False
        if obj['track'].has_key('tags'):
            for tag in obj['track']['tags']:
                if tag == 'occluded':
                    occluded_flag = True
                    break
        if not occluded_flag:
            continue

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

        T_v = T_m 

        # enlarge box
        scale = 1.2

        obj_marker.label = int(obj['id'])
        obj_marker.pose.position.x = T_v[0, 3]
        obj_marker.pose.position.y = T_v[1, 3]
        obj_marker.pose.position.z = T_v[2, 3]
        obj_marker.pose.orientation.x = quaternion_from_matrix(T_v)[0]
        obj_marker.pose.orientation.y = quaternion_from_matrix(T_v)[1]
        obj_marker.pose.orientation.z = quaternion_from_matrix(T_v)[2]
        obj_marker.pose.orientation.w = quaternion_from_matrix(T_v)[3]
        obj_marker.dimensions.x = obj['track']['box']['length']*scale
        obj_marker.dimensions.y = obj['track']['box']['width']*scale
        obj_marker.dimensions.z = obj['track']['box']['height']

        obj_markers.boxes.append(obj_marker)
        # print(obj_markers.boxes[-1].pose.position.x, obj_markers.boxes[-1].pose.position.y, obj_markers.boxes[-1].pose.position.z)

        id_marker = Marker()
        id_marker.header = header
        id_marker.action = Marker.ADD
        id_marker.ns = 'occluded_gt_id'
        id_marker.type = Marker.TEXT_VIEW_FACING
        id_marker.lifetime = rospy.Duration(1/play_rate)

        id_marker.color.r = 0
        id_marker.color.g = 1
        id_marker.color.b = 0
        id_marker.color.a = 1
        id_marker.scale.z = 1.2
        id_marker.id = int(obj['id'])
        id_marker.text = str((obj['id'])) 
        id_marker.pose.position.x = T_v[0, 3]
        id_marker.pose.position.y = T_v[1, 3]
        id_marker.pose.position.z = T_v[2, 3] + 1
        id_markers.markers.append(id_marker)

    return obj_markers, id_markers

def getStampedSwitch(gt, id_switch):
    switch_gt = {}
    for id, profile in id_switch.items():
        for stamp in profile['history'].keys():
            if not switch_gt.has_key(stamp):
                switch_gt[stamp] = []
            

            switch_gt[stamp].append(gt[stamp])
    return switch_gt

def reconfirue_file(gt, over_seg_dict):
    # {'t1':[{gt1: [det1, det2,..]}, {gt2: [det1, det2,..]}, ...] , 't2': }
    # {'t1':{gt1: [det1, det2,..], gt2: [det1, det2,..], ...} , 't2': }
    stamped_dict = {}
    stamp_list = []

    for k, v in over_seg_dict.items():
        for f in v:
            if not stamped_dict.has_key(f['timestamp']):
                # stamped_dict[str(f['timestamp'])] = []
                stamped_dict[str(f['timestamp'])] = {}
            
            # stamped_dict[str(f['timestamp'])].append({int(k): f['det']})
            stamped_dict[str(f['timestamp'])].update({int(k): f['det']})
    return stamped_dict

def getSwitchDuration(id_list):
    duration_list = {}

    for id, log in id_list.items():
        happen_list = []
        sw_id_time = map(int, log['happened'].keys())
        for f in sorted(sw_id_time):
            if log['happened'][str(f)] == True:
                happen_list.append(f)

        duration_list[id] = {'history': id_list[id]['history'], 'happened': happen_list}
        assert len(happen_list) == log['switch_num'], 'Not match idsw num'

    return duration_list

def interrupt():
    global block
    while not shut_down: 
        # raw input would block and keep waiting(stall) for input
        raw_input('press any key to pause or resume:\n')
        block = not block
        print('pause' if block else 'resume')

    print('shut_down')


def viz_frag(all_gt, all_det, stamp, tp_id_history, frag_id):
    '''
        viz frag line in image
    '''
    global img
    frag_id_list = [int(id) for id in frag_id.keys() if frag_id[id]['frag_num'] > 0]

    trajectories = []
    duration_time = 1
    for idx, obj in enumerate(all_gt[stamp]):
        if  obj['id'] not in frag_id_list:
            continue
        if not (obj['id'] == 5 or obj['id'] == 2):
            continue

        during_frag = False
        for m in frag_id[str(obj['id'])]['happened']:
            if (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) <= int(m)) or \
                (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) > int(m)):
                during_frag = True
                break
        if not during_frag:
            continue

        # draw all gt tras and obj tras during duration_time
        gt_trajectory = []
        det_trajectory = {}
        # iterate frame to get duration times
        for f, gts in all_gt.items():
            if abs(int(f)-int(stamp))/10**9 > duration_time:
                continue

            # search corresponding gt loc
            gt_track = {}
            for gt in gts:
                if int(gt['id']) != int(obj['id']):
                    continue
                else:
                    gt_track = gt
                    break

            if not bool(gt_track):
                continue

            l = gt_track['track']['box']['length']
            w = gt_track['track']['box']['width']
            lengths = [l/2, -l/2, -l/2, l/2]
            widths = [w/2, w/2, -w/2, -w/2]
            corners_img_coor = []

            for idx, pt in enumerate(lengths):
                corner = np.array([lengths[idx], widths[idx], 0, 1])

                T_m = transformations.quaternion_matrix(
                    np.array([
                        gt_track['track']['rotation']['x'],
                        gt_track['track']['rotation']['y'],
                        gt_track['track']['rotation']['z'],
                        gt_track['track']['rotation']['w']]))
            
                T_m[0, 3] = gt_track['track']['translation']['x']
                T_m[1, 3] = gt_track['track']['translation']['y']
                T_m[2, 3] = gt_track['track']['translation']['z']

                ego_pose = T_m.dot(corner)
                # img_pose = [-ego_pose[1]*pixel_scale+(img_size/2), -ego_pose[0]*pixel_scale+(img_size/2)]
                img_pose = [-ego_pose[1]*pixel_scale+(img_size/2), -ego_pose[0]*pixel_scale+(img_size)]
                corners_img_coor.append(img_pose)

            color = (0, 0, 255)
            if int(obj['id']) % 4 == 1:
                color = (0, 255, 255)
            elif int(obj['id']) % 4 == 2:
                color = (255, 0, 0)

            for idx, pt in enumerate(corners_img_coor):
                cv2.line(img, (int(corners_img_coor[idx%len(corners_img_coor)][0]), int(corners_img_coor[idx%len(corners_img_coor)][1])), (int(corners_img_coor[(idx+1)%len(corners_img_coor)][0]), int(corners_img_coor[(idx+1)%len(corners_img_coor)][1])), color, 1)


            # center_img_coord = [
            #     -gt_track['track']['translation']['y']*pixel_scale+(img_size/2),
            #     -gt_track['track']['translation']['x']*pixel_scale+(img_size/2)]
            center_img_coord = [
                -gt_track['track']['translation']['y']*pixel_scale+(img_size/2),
                -gt_track['track']['translation']['x']*pixel_scale+(img_size)]
            
            gt_trajectory.append(center_img_coord)


            # get tp during duration
            tp_det_id = tp_id_history[str(obj['id'])]['id'][str(f)]
            if not np.isnan(tp_det_id):
                if not det_trajectory.has_key(tp_det_id):
                    det_trajectory[tp_det_id] = []
                
                # get tp det loc
                for i, d in enumerate(all_det[str(f)]):
                    if int(d['id']) != tp_det_id:
                        continue
                    else:
                        # center_det_img_coord = [
                        #     -d['track']['translation']['y']*pixel_scale+(img_size/2),
                        #     -d['track']['translation']['x']*pixel_scale+(img_size/2)]  
                        center_det_img_coord = [
                            -d['track']['translation']['y']*pixel_scale+(img_size/2),
                            -d['track']['translation']['x']*pixel_scale+(img_size)]  

                        det_trajectory[tp_det_id].append(center_det_img_coord)
                        break

        
        # drawing gt gt_trajectory during frag duration
        for idx, pt in enumerate(gt_trajectory):
            cv2.circle(img, (int(gt_trajectory[idx%len(gt_trajectory)][0]), int(gt_trajectory[idx%len(gt_trajectory)][1])), 3, (0, 0, 255), -1)
            if idx == len(gt_trajectory) - 1:
                break
            cv2.line(img, (int(gt_trajectory[idx%len(gt_trajectory)][0]), int(gt_trajectory[idx%len(gt_trajectory)][1])), (int(gt_trajectory[(idx+1)%len(gt_trajectory)][0]), int(gt_trajectory[(idx+1)%len(gt_trajectory)][1])), (0, 255, 0), 3)

        # plot fragmented gt as circle where tp_det_id = nan

        # drawing tp det trajectories
        for tp, tras in det_trajectory.items():
            color = (0, 0, 255)
            if tp % 4 == 1:
                color = (0, 255, 255)
            elif tp % 4 == 2:
                color = (255, 0, 0)

            for j, pt in enumerate(tras):
                if j == len(tras) - 1:
                    break
                cv2.line(img, (int(tras[j%len(tras)][0]), int(tras[j%len(tras)][1])), (int(tras[(j+1)%len(tras)][0]), int(tras[(j+1)%len(tras)][1])), color, 3)

    cv2.imshow('Frag', img)
    cv2.waitKey(10)



def frag_diff(gt_dict, objs_dict, frag, tp_id_history, objs_dict_2, frag_2, tp_id_history_2, stamp):
    '''
        Show frag difference from main result(1) and 2 result
        Which 2 has frag but is not fraged at 1 (improve at 1)
    '''
    frag_id_list = [int(id) for id in frag.keys() if frag[id]['frag_num'] > 0]
    frag_id_list_2 = [int(id) for id in frag_2.keys() if frag_2[id]['frag_num'] > 0]
    # print('original: {}'.format(len(frag_id_list_2)))
    # print('modified: {}'.format(len(frag_id_list)))
    improved_id_list = list(set(frag_id_list_2) - set(frag_id_list))
    worsen_id_list = list(set(frag_id_list) - set(frag_id_list_2))
    union_id_list = list(set(frag_id_list_2) | set(frag_id_list))
    interset_id_list = list(set(frag_id_list_2) & set(frag_id_list))
    # print(improved_id_list)
    # print(worsen_id_list)
    # print(interset_id_list)


    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'
    frag_diff_gt_markers = BoundingBoxArray()
    frag_diff_gt_markers.header = header
    frag_det_markers = BoundingBoxArray()
    frag_det_markers.header = header
    frag_det_2_markers = BoundingBoxArray()
    frag_det_2_markers.header = header

    # occluded part shpere
    occluded_marker = Marker()
    occluded_marker.header = header
    occluded_marker.action = Marker.ADD
    occluded_marker.ns = 'occluded_part'
    occluded_marker.type = Marker.POINTS
    occluded_marker.lifetime = rospy.Duration(1/play_rate)
    occluded_marker.color.r = 1
    occluded_marker.color.g = 0
    occluded_marker.color.b = 0
    occluded_marker.color.a = 1
    occluded_marker.scale.x = 0.38
    occluded_marker.scale.y = 0.38
    # occluded_marker.scale.z = 0.5
    occluded_marker.id = 0

    duration_time = 0.3
    for idx, gt in enumerate(gt_dict):

        # Only show id different frag (frag1-frag2) or (frag2-frag1)
        if (gt['id'] not in union_id_list) or (gt['id'] in interset_id_list):
            continue

        # search which result are fragmented
        frag_dict_to_search = frag if gt['id'] in frag_id_list else frag_2
        obj_dict_to_search = objs_dict if gt['id'] in frag_id_list else objs_dict_2 
        tp_dict_to_search = tp_id_history if gt['id'] in frag_id_list else tp_id_history_2 
        markers = frag_det_markers if gt['id'] in frag_id_list else frag_det_2_markers 

        # during_frag = False
        # for m in frag_dict_to_search[str(gt['id'])]['happened']:
        #     if (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) <= int(m)) or \
        #         (abs(int(stamp) - int(m))/1e9 < duration_time and int(stamp) > int(m)):
        #         during_frag = True
        #         # print('happened time: {}'.format(m))
        #         break

        # find duration of frag which is untracked duration (happened is the first frame of untracked)
        sort_t = sorted(tp_dict_to_search[str(gt['id'])]['matched'])
        sort_matched = []
        for t in sort_t:
            sort_matched.append(tp_dict_to_search[str(gt['id'])]['matched'][t])
    
        in_frag = False
        first_track = -1
        last_track = len(sort_matched)
        counted_flag = False
        for idx, tracked in enumerate(sort_matched):
            if tracked:
                first_track = idx
                break
        
        for idx, tracked in reversed(list(enumerate(sort_matched))):
            if tracked:
                last_track = idx
                break
        duration_ts = []
        duration_t = []

        # make sure would be tracked at least 1 frame
        if not(first_track == -1 and last_track == len(sort_matched)):
            for idx, m in enumerate(sort_matched):
                if idx < first_track:
                    continue

                # close duration
                if in_frag and m:
                    duration_ts.append(duration_t)
                    duration_t = []
                    in_frag = False
                # in duration
                elif in_frag and not m:
                    duration_t.append(sort_t[idx])
                # new duration
                if not m and not in_frag:
                    duration_t.append(sort_t[idx])
                    in_frag = True

                if idx > last_track:
                    break
    
        during_frag = False
        for frag_times in duration_ts:
            for m in frag_times:
                if (abs(int(stamp) - int(m))/1e9 < 0.5 and int(stamp) <= int(m)) or \
                (abs(int(stamp) - int(m))/1e9 < 0.2 and int(stamp) > int(m)):
                    during_frag = True
                    break

        if not during_frag:
            continue

        # gt_bbox
        # mark which gt is improved in tracker 1
        if gt['id'] in frag_id_list_2:
            obj_marker = BoundingBox()
            obj_marker.header = header
            T_m = transformations.quaternion_matrix(
                np.array([
                    gt['track']['rotation']['x'],
                    gt['track']['rotation']['y'],
                    gt['track']['rotation']['z'],
                    gt['track']['rotation']['w']]))
        
            T_m[0, 3] = gt['track']['translation']['x']
            T_m[1, 3] = gt['track']['translation']['y']
            T_m[2, 3] = gt['track']['translation']['z']

            obj_marker.label = gt['id']
            obj_marker.pose.position.x = T_m[0, 3]
            obj_marker.pose.position.y = T_m[1, 3]
            if not show_2D:
                obj_marker.pose.position.z = T_m[2, 3]
            else:
                obj_marker.pose.position.z = 0
            obj_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
            obj_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
            obj_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
            obj_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
            obj_marker.dimensions.x = gt['track']['box']['length']*1.5
            obj_marker.dimensions.y = gt['track']['box']['width']*1.5
            if not show_2D:
                obj_marker.dimensions.z = gt['track']['box']['height']
            else:
                obj_marker.dimensions.z = 0.001
            frag_diff_gt_markers.boxes.append(obj_marker)


            # also pub improved occluded part during this frag duration
            tp_det_id = tp_id_history[str(gt['id'])]['id'][str(stamp)]
            for det in objs_dict:
                if det['id'] != tp_det_id:
                    continue

                # pub occluded part
                occlude_part = det['track']['occluded_part'] if det['track'].has_key('occluded_part') else []
                for grid in occlude_part:
                    if show_2D:
                        occluded_marker.points.append(Point( grid['x'],  grid['y'], 0.5))
                    else:
                        occluded_marker.points.append(Point( grid['x'],  grid['y'], grid['z']))
                break



        tp_det_id = tp_dict_to_search[str(gt['id'])]['id'][str(stamp)]
        for det in obj_dict_to_search:
            if det['id'] != tp_det_id:
                continue

            det_marker = BoundingBox()
            det_marker.header = header

            T_m = transformations.quaternion_matrix(
                np.array([
                    det['track']['rotation']['x'],
                    det['track']['rotation']['y'],
                    det['track']['rotation']['z'],
                    det['track']['rotation']['w']]))
        
            T_m[0, 3] = det['track']['translation']['x']
            T_m[1, 3] = det['track']['translation']['y']
            T_m[2, 3] = det['track']['translation']['z']

            # 1000 -> not tracked as blue
            det_marker.label = det['id']
            det_marker.pose.position.x = T_m[0, 3]
            det_marker.pose.position.y = T_m[1, 3]
            det_marker.pose.position.z = T_m[2, 3]
            det_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
            det_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
            det_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
            det_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
            det_marker.dimensions.x = det['track']['box']['length']
            det_marker.dimensions.y = det['track']['box']['width']
            if not show_2D:
                det_marker.dimensions.z = det['track']['box']['height']
            else:
                det_marker.dimensions.z = 0.001
            markers.boxes.append(det_marker)
            break

    return frag_det_markers, frag_det_2_markers, frag_diff_gt_markers, occluded_marker


def sw_diff(gt_dict, stamp, objs_dict, post_id_switch, objs_dict_2, post_id_switch_2):
    header = Header()
    header.stamp.secs = int(int(stamp) / 1e9)
    header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    header.frame_id = 'velodyne'
    sw_diff_gt_markers = BoundingBoxArray()
    sw_diff_gt_markers.header = header
    sw_det_markers = BoundingBoxArray()
    sw_det_markers.header = header
    sw_det_2_markers = BoundingBoxArray()
    sw_det_2_markers.header = header

    sw_id_list_1 = map(int, post_id_switch.keys())
    sw_id_list_2 = map(int, post_id_switch_2.keys())

    union_id_list = list(set(sw_id_list_2) | set(sw_id_list_1))
    interset_id_list = list(set(sw_id_list_2) & set(sw_id_list_1))
    improved_id_list = list(set(sw_id_list_2) - set(sw_id_list_1))
    # improved_id_list = list(set(sw_id_list_1) - set(sw_id_list_2))
    # print(improved_id_list)


    # get happened stamp diff, not id diff
    # {'id': [t1, t2...], ''}
    sw_dict_1 = {}
    sw_dict_2 = {}
    for id, dict in post_id_switch.items():
        if not sw_dict_1.has_key(id):
            sw_dict_1[id] = []
        sw_dict_1[id] += dict['happened']
    for id, dict in post_id_switch_2.items():
        if not sw_dict_2.has_key(id):
            sw_dict_2[id] = []
        sw_dict_2[id] += dict['happened']

    diff_sw_dict_1 = {}
    diff_sw_dict_2 = {}
    for id, times in sw_dict_1.items():
        if sw_dict_2.has_key(id):
            diff_1_2 = list(set(sw_dict_1[id]) - set(sw_dict_2[id]))
            if len(diff_1_2) != 0:
                diff_sw_dict_1[id] = diff_1_2
        else:
            diff_sw_dict_1[id] = times

    for id, times in sw_dict_2.items():
        if sw_dict_1.has_key(id):
            diff_2_1 = list(set(sw_dict_2[id]) - set(sw_dict_1[id]))
            if len(diff_2_1) != 0:
                diff_sw_dict_2[id] = diff_2_1
        else:
            diff_sw_dict_2[id] = times

    # print(diff_sw_dict_1.keys())
    # print(diff_sw_dict_2.keys())

    # if diff_sw_dict_1.has_key(str(89)):
    #     print('1: {}'.format(diff_sw_dict_1[str(89)]))
    # if diff_sw_dict_2.has_key(str(89)):
    #     print('2: {}'.format(diff_sw_dict_2[str(89)]))

    duration_time = 2
    for idx, gt in enumerate(gt_dict):

        # if gt['id'] != 89:
        #     continue
        # if (gt['id'] not in union_id_list) or (gt['id'] in interset_id_list):
        #     continue


        # search which result are fragmented
        # gt_dict_to_search = post_id_switch if gt['id'] in sw_id_list_1 else post_id_switch_2
        # obj_dict_to_search = objs_dict if gt['id'] in sw_id_list_1 else objs_dict_2 
        # markers = sw_det_markers if gt['id'] in sw_id_list_1 else sw_det_2_markers 

        # # just show during duration_num [sec] before and after happening moment
        # during_sw = False
        # time_to_sw = False
        # for m in gt_dict_to_search[str(gt['id'])]['happened']:
        #     if (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) <= m) or \
        #         (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) > m):
        #         during_sw = True

        #         if int(stamp) == m:
        #             time_to_sw = True
        #         break
        # if not during_sw:
        #     continue

        # search timestamp based
        if (str(gt['id']) not in diff_sw_dict_1.keys()) and (str(gt['id']) not in diff_sw_dict_2.keys()):
            continue

        # print('2 Get {}'.format(gt['id']))
        gt_dict_to_search = {}
        obj_dict_to_search = {}
        markers = BoundingBoxArray()

        during_sw = False
        if (str(gt['id']) in diff_sw_dict_1.keys()):
            # just show during duration_num [sec] before and after happening moment
            for m in post_id_switch[str(gt['id'])]['happened']:
                if (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) <= m) or \
                    (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) > m):
                    during_sw = True
                    gt_dict_to_search = post_id_switch
                    obj_dict_to_search = objs_dict
                    markers = sw_det_markers

                    # det
                    try:
                        tp_det_id = gt_dict_to_search[str(gt['id'])]['history'][str(stamp)]
                    except:
                        continue
                    # print('gt: {}; tp: {}'.format(gt['id'],tp_det_id))
                    for det in obj_dict_to_search:
                        if det['id'] != tp_det_id:
                            continue

                        det_marker = BoundingBox()
                        det_marker.header = header


                        T_m = transformations.quaternion_matrix(
                            np.array([
                                det['track']['rotation']['x'],
                                det['track']['rotation']['y'],
                                det['track']['rotation']['z'],
                                det['track']['rotation']['w']]))
                    
                        T_m[0, 3] = det['track']['translation']['x']
                        T_m[1, 3] = det['track']['translation']['y']
                        T_m[2, 3] = det['track']['translation']['z']

                        det_marker.label = det['id']
                        det_marker.pose.position.x = T_m[0, 3]
                        det_marker.pose.position.y = T_m[1, 3]
                        if not show_2D:
                            det_marker.pose.position.z = T_m[2, 3]
                        else:
                            det_marker.pose.position.z = 0
                        det_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
                        det_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
                        det_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
                        det_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
                        det_marker.dimensions.x = det['track']['box']['length']
                        det_marker.dimensions.y = det['track']['box']['width']
                        if not show_2D:
                            det_marker.dimensions.z = det['track']['box']['height']
                        else:
                            det_marker.dimensions.z = 0.001
                        markers.boxes.append(det_marker)
                        break

                    # print('gt: {} in 1'.format(gt['id']))
                    break
        
        if (str(gt['id']) in diff_sw_dict_2.keys()):
            # just show during duration_num [sec] before and after happening moment
            for m in post_id_switch_2[str(gt['id'])]['happened']:
                if (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) <= m) or \
                    (abs(int(stamp) - m)/1e9 < duration_time and int(stamp) > m):
                    during_sw = True
                    gt_dict_to_search = post_id_switch_2
                    obj_dict_to_search = objs_dict_2
                    markers = sw_det_2_markers

                    # det
                    tp_det_id = gt_dict_to_search[str(gt['id'])]['history'][str(stamp)]
                    # print('gt: {}; tp: {}'.format(gt['id'],tp_det_id))
                    for det in obj_dict_to_search:
                        if det['id'] != tp_det_id:
                            continue

                        det_marker = BoundingBox()
                        det_marker.header = header


                        T_m = transformations.quaternion_matrix(
                            np.array([
                                det['track']['rotation']['x'],
                                det['track']['rotation']['y'],
                                det['track']['rotation']['z'],
                                det['track']['rotation']['w']]))
                    
                        T_m[0, 3] = det['track']['translation']['x']
                        T_m[1, 3] = det['track']['translation']['y']
                        T_m[2, 3] = det['track']['translation']['z']

                        det_marker.label = det['id']
                        det_marker.pose.position.x = T_m[0, 3]
                        det_marker.pose.position.y = T_m[1, 3]
                        if not show_2D:
                            det_marker.pose.position.z = T_m[2, 3]
                        else:
                            det_marker.pose.position.z = 0
                        det_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
                        det_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
                        det_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
                        det_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
                        det_marker.dimensions.x = det['track']['box']['length']
                        det_marker.dimensions.y = det['track']['box']['width']
                        if not show_2D:
                            det_marker.dimensions.z = det['track']['box']['height']
                        else:
                            det_marker.dimensions.z = 0.001
                        markers.boxes.append(det_marker)
                        break

                    # print('gt: {} in 2'.format(gt['id']))
                    break

        if not during_sw:
            continue

        # gt
        if (str(gt['id']) in diff_sw_dict_2.keys()) and (str(gt['id']) not in diff_sw_dict_1.keys()):
            gt_marker = BoundingBox()
            gt_marker.header = header
            T_m = transformations.quaternion_matrix(
                np.array([
                    gt['track']['rotation']['x'],
                    gt['track']['rotation']['y'],
                    gt['track']['rotation']['z'],
                    gt['track']['rotation']['w']]))
        
            T_m[0, 3] = gt['track']['translation']['x']
            T_m[1, 3] = gt['track']['translation']['y']
            T_m[2, 3] = gt['track']['translation']['z']

            gt_marker.label = 10000000
            gt_marker.pose.position.x = T_m[0, 3]
            gt_marker.pose.position.y = T_m[1, 3]
            if not show_2D:
                gt_marker.pose.position.z = T_m[2, 3]
            else:
                gt_marker.pose.position.z = 0
            gt_marker.pose.orientation.x = quaternion_from_matrix(T_m)[0]
            gt_marker.pose.orientation.y = quaternion_from_matrix(T_m)[1]
            gt_marker.pose.orientation.z = quaternion_from_matrix(T_m)[2]
            gt_marker.pose.orientation.w = quaternion_from_matrix(T_m)[3]
            gt_marker.dimensions.x = gt['track']['box']['length'] * 1.5
            gt_marker.dimensions.y = gt['track']['box']['width'] * 1.5
            if not show_2D:
                gt_marker.dimensions.z = gt['track']['box']['height']
            else:
                gt_marker.dimensions.z = 0.001
            sw_diff_gt_markers.boxes.append(gt_marker)

    return sw_det_markers, sw_det_2_markers, sw_diff_gt_markers
    
def viz_roadpolys(sub_map, pub):
    global rl_markers
    elevation = -67 # 2020-09-11-17-37-12_4.bag
    elevation = 0 # 2020-09-11-17-31-33_9.bag

    for poly_idx, road_poly in enumerate(sub_map):
        marker = Marker()
        marker.header.frame_id = 'velodyne'
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.ns = 'non_accessible'
        marker.type = Marker.LINE_LIST
        marker.lifetime = rospy.Duration(1/play_rate)

        marker.color.r = 0
        marker.color.g = 1
        marker.color.b = 1
        marker.color.a = 1
        marker.scale.x = 0.1

        marker.points = []
        marker.id = road_poly.get('id', poly_idx)

        # check extend pt
        pt_marker = Marker()
        pt_marker.header.frame_id = 'velodyne'
        pt_marker.header.stamp = rospy.Time.now()
        pt_marker.action = Marker.ADD
        pt_marker.ns = 'non_accessible' + '_pt'
        pt_marker.type = Marker.POINTS
        pt_marker.lifetime = rospy.Duration(1/play_rate)

        pt_marker.color.r = 1
        pt_marker.color.g = 1
        pt_marker.color.b = 0
        pt_marker.color.a = 1
        pt_marker.scale.x = 0.5
        pt_marker.scale.y = 0.5

        pt_marker.points = []
        pt_marker.id = road_poly.get('id', poly_idx)

        id_marker = Marker()
        id_marker.header.frame_id = 'velodyne'
        id_marker.header.stamp = rospy.Time.now()
        id_marker.action = Marker.ADD
        id_marker.ns = 'non_accessible' + '_id'
        id_marker.type = Marker.TEXT_VIEW_FACING
        id_marker.lifetime = rospy.Duration(1/play_rate)

        id_marker.color.r = 1
        id_marker.color.g = 1
        id_marker.color.b = 0
        id_marker.color.a = 1
        id_marker.scale.z = 5

        # id_marker.id = road_poly.get('main_id', poly_idx)
        # id_marker.text = str(road_poly.get('main_id', poly_idx))
        # id_marker.pose.position.x = road_poly['points'][0]['x']
        # id_marker.pose.position.y = road_poly['points'][0]['y']
        # id_marker.pose.position.z = elevation

        extend_marker = Marker()
        extend_marker.header.frame_id = 'velodyne'
        extend_marker.header.stamp = rospy.Time.now()
        extend_marker.action = Marker.ADD
        extend_marker.ns = 'non_accessible' + '_line'
        extend_marker.type = Marker.LINE_LIST
        extend_marker.lifetime = rospy.Duration(1/play_rate)

        extend_marker.color.r = 1
        extend_marker.color.g = 1
        extend_marker.color.b = 0
        extend_marker.color.a = 1
        extend_marker.scale.x = 0.1

        extend_marker.points = []
        extend_marker.id = road_poly.get('id', poly_idx)

        pts = road_poly['points']
        for index, pt in enumerate(pts):
            if pt.has_key('point_id') and pt['point_id'] == -1:
                pt_marker.points.append(Point(pt['x'], pt['y'], elevation))

                if index == 0:
                    extend_marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
                    extend_marker.points.append(Point(pts[(index+1)%len(pts)]['x'], pts[(index+1)%len(pts)]['y'], elevation))
                else:
                    extend_marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
                    extend_marker.points.append(Point(pts[(index-1)%len(pts)]['x'], pts[(index-1)%len(pts)]['y'], elevation))
            
            # don't enclose polygon btw first and last pts
            if index == len(pts)-1:
                break
            marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
            marker.points.append(Point(pts[(index+1)%len(pts)]['x'], pts[(index+1)%len(pts)]['y'], elevation))
        rl_markers.markers.append(marker)
        rl_markers.markers.append(pt_marker)
        if 'non_accessible' == 'non_accessible':
            rl_markers.markers.append(id_marker)
            rl_markers.markers.append(extend_marker)

    pub.publish(rl_markers)


def transformSemanticMap(sub_map, tf_data, pub):
    T_tf = quaternion_matrix(
    np.array([
        tf_data['pose']['rotation']['x'],
        tf_data['pose']['rotation']['y'],
        tf_data['pose']['rotation']['z'],
        tf_data['pose']['rotation']['w']]))

    T_tf[0, 3] = tf_data['pose']['translation']['x']
    T_tf[1, 3] = tf_data['pose']['translation']['y']
    T_tf[2, 3] = tf_data['pose']['translation']['z']

    local_sub_map = []
    for region in sub_map:
        local_sub_map.append(copy.deepcopy(region))
        local_sub_map[-1]['points'] = []
        for pt in region['points']:
            # print('tf: {}, {}, {}'.format(T_tf[0, 3], T_tf[1, 3], T_tf[2, 3]))
            # print('global pt: {}, {}, {}'.format(pt['x'], pt['y'], pt['z']))
            T_global = np.eye(4)
            T_global[0, 3] = pt['x']
            T_global[1, 3] = pt['y']
            T_global[2, 3] = pt['z']

            pt_local = np.linalg.inv(T_tf).dot(T_global)
            local_pt = {}
            local_pt['x'] = pt_local[0, 3]
            local_pt['y'] = pt_local[1, 3]
            local_pt['z'] = pt_local[2, 3]
            local_pt['point_id'] = pt['point_id']
            local_sub_map[-1]['points'].append(local_pt)
            # print('local pt: {}, {}, {}'.format(local_pt['x'], local_pt['y'], local_pt['z']))

    viz_roadpolys(local_sub_map, pub)


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
            det_boxes_msg, det_id_msg = create_boxes_msg(objs_dict, header, 'det')
            gt_boxes_msg, gt_id_msg = create_boxes_msg(gt_dict, header, 'gt')
            fn_boxes_msg, fn_id_msg = create_boxes_msg(fn_dict, header, 'fn')
            fp_boxes_msg, fp_id_msg = create_boxes_msg(fp_dict, header, 'fp')
            pc_msg = create_pc(scans_dict[str(int(stamp))], header)
            # cv2 fragemented ones
            img = np.full((img_size, img_size, 3), 255, np.uint8)
            # viz_frag(gt, det, stamp, tp_id_history, frag)

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

    shut_down = True
    print('Out loop, thread.is_alive: {}, enter any key to stop program thread'.format(thread.is_alive()))
    