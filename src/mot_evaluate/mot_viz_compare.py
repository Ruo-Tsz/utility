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
import cv2

# follow the original result structure
# 'PATH_TO_ROOT_DIR_OF_RESULT/kuang-fu-rd/2020-09-11-17-31-33'
result_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/2022-04-18_01-35-24'
result_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/v2/2022-04-18_04-23-25'
viz_segment = '2020-09-11-17-37-12_4'
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
    