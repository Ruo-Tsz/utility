#! /usr/bin/python2
from __future__ import division
import json
import os
import copy
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, translation_from_matrix, quaternion_from_euler
import shapely
from shapely import geometry, affinity
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

'''
    Find fully occluded groundtruth from evaluation result file by checking point cloud existence
    @param:
        result directory from mot_evaluate.py
        point cloud for that scenes
'''


# scenes = "2020-09-11-17-37-12_4"
# scenes = "2020-09-11-17-31-33_9"
scenes = "2020-09-11-17-37-12_1"
# gt_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735"
# gt_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing"
gt_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1"
gt_data = {}
# check box above close_to_bottom meters for remove ground pt
# close_to_bottom = 0.01
close_to_bottom = 0.0

output_file = "occluded_gt_v2.json"

id_list = []

def getYawFromQuat(obj):
    return euler_from_quaternion([obj['rotation']['x'], obj['rotation']['y'],
            obj['rotation']['z'], obj['rotation']['w']])[2]


class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle, use_radians=True)
        return shapely.affinity.translate(rc, self.cx, self.cy)
        # return rc

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())



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

    return scans_dict

def isBoxEmpty(gt_poly, pc):
    '''
        return True if at least one point in poly
    '''
    import matplotlib.path as mplPath

    # crd = np.array([[0,0], [0,1], [1,1], [1,0]])# poly
    bbPath = mplPath.Path(gt_poly)
    points = []
    for pt in pc:
        points.append([pt[0], pt[1]])

    # pnts = [[0.0, 0.0],[1,1],[0.0,0.5],[0.5,0.0]] # points on edges
    # tolerance for point on edge, if = 0, on edge is not exclude
    r = 0.001 # accuracy
    isIn = [ bbPath.contains_point(pt,radius=r) or bbPath.contains_point(pt,radius=-r) for pt in points]
    
    if np.any(isIn):
        return False
    else:
        return True

def RectToPoly(groundTruth, shift=0, direction=0, retrun_shapely = False):
    halfL = groundTruth['box']['length'] / 2.0
    halfW = groundTruth['box']['width'] / 2.0
    matrixR = quaternion_matrix(
        [groundTruth['rotation']['x'],
        groundTruth['rotation']['y'],
        groundTruth['rotation']['z'],
        groundTruth['rotation']['w']])[:3, :3]

    if direction == 0:
        pt1 = np.dot(matrixR, [halfL + shift, halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt2 = np.dot(matrixR, [- halfL + shift, halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt3 = np.dot(matrixR, [- halfL + shift, - halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt4 = np.dot(matrixR, [halfL + shift, - halfW + shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
    else:
        pt1 = np.dot(matrixR, [halfL + shift, halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt2 = np.dot(matrixR, [- halfL + shift, halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt3 = np.dot(matrixR, [- halfL + shift, - halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]
        pt4 = np.dot(matrixR, [halfL + shift, - halfW - shift, 0.0]) + [groundTruth['translation']['x'],
            groundTruth['translation']['y'], groundTruth['translation']['z']]

    if retrun_shapely:
        return Polygon(
            [(pt1[0], pt1[1]),
            (pt2[0], pt2[1]),
            (pt3[0], pt3[1]),
            (pt4[0], pt4[1])])
    else:
        np_poly = np.vstack((pt1[:2], pt2[:2], pt3[:2], pt4[:2]))
        return np_poly

def checkPlot(gt, pc):
    x = pc[:, 0]
    y = pc[:, 1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.scatter(x, y)

    yaw_gt = getYawFromQuat(gt)
    rect_gt = RotatedRect(gt['translation']['x'], gt['translation']['y'],  gt['box']['length'], gt['box']['width'], yaw_gt)

    # fig, ax = plt.subplots()
    ax.plot(rect_gt.get_contour().boundary.coords.xy[0], rect_gt.get_contour().boundary.coords.xy[1], 'y-')
    ax.set_aspect('equal')
    plt.show()

def filter_box_without_point(gts, scenes_file):
    '''
        Filtering boxes with no point inside
    '''

    # {'id':[t1, t2],...}
    occluded_id_list = {}

    big_scene = scenes_file.split('_')[0]
    # pc_path = os.path.join('/data/itri_output/tracking_output/pointcloud/no_ego_compensated', big_scene, scenes_file)
    # pc_path = os.path.join('/data/itri_output/tracking_output/kuang-fu-rd_livox_public/ego compensation/kuang-fu-rd_v3', big_scene, 'pointcloud', scenes_file)
    pc_path = os.path.join('/data/itri_output/tracking_output/pointcloud/ego_compensated' , big_scene, scenes_file)
    # load point cloud
    all_pc = load_pc(pc_path)

    out_gts_data = copy.deepcopy(gts)

    if len(gts) == 0:
        print("Zero gts, abort")
        return out_gts_data

    # for anno in gts['tracks']:
    total_counter = 0
    original_num = 0
    for scan, tracks in sorted(gts.items()):
        original_num += len(tracks)
        scan_counter = 0
        out_gts_data[scan] = []

        pt_trajectory_list = []
        for idx, gt_data in enumerate(tracks):
            if not gt_data['id'] in id_list:
                id_list.append(gt_data['id'])

            gt = gt_data['track']
            
            # crop pc to box height, N X 4 array
            upper_b = gt['translation']['z']+gt['box']['height']/2+close_to_bottom
            lower_b = gt['translation']['z']-gt['box']['height']/2+close_to_bottom
            roi = 15
            roi = (gt['box']['length']/2 if (gt['box']['length']>gt['box']['width']) else gt['box']['width']/2)
            left_x = gt['translation']['x']-roi*1.5
            right_x = gt['translation']['x']+roi*1.5
            left_y = gt['translation']['y']-roi*1.5
            right_y = gt['translation']['y']+roi*1.5
            current_pc = all_pc[str(scan)]
            # extract pc within box heigh and cropped to 2d (N x 2)
            cropped_pc = current_pc[(current_pc[:, -2] >= lower_b) & (current_pc[:, -2] <= upper_b)][:, :2]
            cropped_center_x = cropped_pc[(cropped_pc[:, 0] >= left_x) & (cropped_pc[:, 0] <= right_x)]
            cropped_center_y = cropped_center_x[(cropped_center_x[:, 1] >= left_y) & (cropped_center_x[:, 1] <= right_y)]

            gt_poly = RectToPoly(gt)

            if isBoxEmpty(gt_poly, cropped_center_y):
                scan_counter += 1
                
                if not gt_data['id'] in occluded_id_list.keys():
                    occluded_id_list.update({gt_data['id']: [scan]})
                else:
                    occluded_id_list[gt_data['id']].append(scan)

                    # checkPlot(gt, cropped_center_y)
                    # checkPlot(gt, current_pc)
                continue            
            
            pt_trajectory_list.append(gt_data)
        
        total_counter += scan_counter
        
        out_gts_data[scan] = pt_trajectory_list

    print('we have {} occluded'.format(total_counter))
    print('we have {}/{} occluded trajectory'.format(len(occluded_id_list), len(id_list)))
    print(occluded_id_list.keys())


    # split into segment of complete occlusion
    id_occlusion_his = findOcclusionSeg(occluded_id_list, all_pc)

    # check average length over largest lenght of each gt
    max_length, ave_length = lengthOcclusion(id_occlusion_his)


    details ={
        "trajectory_num": len(id_list),
        "occluded_trajectory_num": len(occluded_id_list),
        "gt_num": original_num,
        "occluded_num": total_counter,
        "occluded_time": occluded_id_list,
        "occluded_segments": id_occlusion_his,
        "occlusion_length": {"ave_length": ave_length, "max_length": max_length}}
    # output = {"meta:", "details": details}
    # details["occluded_id"].append(occluded_id_list)

    with open (os.path.join(gt_path, output_file), "w") as outFile:
        json.dump(details, outFile, indent = 4)

    return out_gts_data

def findOcclusionSeg(occluded_id_list, all_pc):

    id_occlusion_his_all = {}

    sorted_all_pc = sorted(all_pc)

    for id, ts in occluded_id_list.items():
        sort_t = sorted(ts)

        id_longest = False
        id_occlusion_length = -1
        id_occlusion_his = {}
        
        idx = 0
        while idx != len(sort_t):
            search_idx = idx
            # append interval btw (index-1) and index of existing timestamp
            index = np.searchsorted(np.array(sorted_all_pc), sort_t[idx])
            search_index = index
            assert sort_t[idx] == sorted_all_pc[index], 'ghost timestamp in gt, {}'.format(sort_t[idx])

            # register single shot
            if len(sort_t) < 2:
                id_occlusion_his.update({index: [sort_t[idx]]})
                break

            # continue search, (search_idx - 1) would be last matched occluded timestamp
            while checkNextElement(sort_t[search_idx], sorted_all_pc[search_index]):
                search_idx += 1
                search_index += 1

                if search_idx == len(sort_t) or search_index == len(sorted_all_pc):
                    break

            # register single shot
            id_occlusion_length = search_idx - idx
            if id_occlusion_length == 1:
                id_occlusion_his.update({index: [sort_t[idx:search_idx]]})
                idx = search_idx
                continue
            
            # print('find {} at {}'.format(sort_t[idx], index))
            # print('end at {}'.format(sort_t[search_idx-1]))
            # print('all_scan: ')
            # print(sorted_all_pc[index:search_index])

            # using initial index of initial timestamp in all pointcloud as key_id (not meaningful) 
            id_occlusion_his.update({index: sort_t[idx:search_idx]})
            # print(id_occlusion_his)

            idx = search_idx

        id_occlusion_his_all.update({id: id_occlusion_his})
  
    return id_occlusion_his_all           
                
def checkNextElement(a, b):
    return a == b

def lengthOcclusion(id_occlusion_his):
    '''
        return 
        1. average of MAX length of each occluded gt
        2. Max length among all gt
    '''

    all_max = -1
    ave_length = 0
    id_max_length = {}
    for id, segs in id_occlusion_his.items():
        max_l = -1
        for seg_id, seg_list in segs.items():
            max_l = len(seg_list) if len(seg_list) > max_l else max_l

        id_max_length.update({id: max_l})

    for id, l in id_max_length.items():
        ave_length += l
        all_max = l if l > all_max else all_max
    
    ave_length /= len(id_occlusion_his)
    print(id_max_length)
    print('ave_length: {}'.format(ave_length))
    print('max length among all: {}'.format(all_max))

    return all_max, ave_length


if __name__ == "__main__":
    print("check gt occlusion by points existence....")
    print("scenes: {}".format(scenes))

    with open (os.path.join(gt_path, 'gt.json'), 'r') as inFile:
        gt_data = json.load(inFile)

    remain_gt = filter_box_without_point(gt_data, scenes)

    # check num
    original_num = 0
    for scan, tracks in gt_data.items():
        original_num += len(tracks)
    print("We have total {} gts".format(original_num))

    filtered_num = 0
    for scan, tracks in remain_gt.items():
        filtered_num += len(tracks)
    print("We have remaining {} gts with pt".format(filtered_num))
    
    





