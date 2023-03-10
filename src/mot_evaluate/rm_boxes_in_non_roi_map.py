#! /usr/bin/python2
import json
import os
import yaml
from unicodedata import name
import numpy as np
import copy
from tqdm import tqdm
from tf import transformations
from tf.transformations import euler_from_quaternion, quaternion_from_matrix, quaternion_matrix, translation_from_matrix
import shapely
from shapely import geometry, affinity
from shapely.geometry import Polygon


# Forced output yaml to indent
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def RectToPoly(groundTruth, shift=0, direction=0):
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

    return Polygon(
        [(pt1[0], pt1[1]),
        (pt2[0], pt2[1]),
        (pt3[0], pt3[1]),
        (pt4[0], pt4[1])])


def loadRoiMap(tf_data):
    map_path = '/home/user/catkin_ws_itri/src/imm_tracker_official/src/euclidean_clustering_filter/material/hsinchu_guandfuroad/non_accessible_region_v2.json'
    with open (map_path, mode='r') as f:
        regions = json.load(f)['non_accessible_region']
    
    sub_map_region = []
    traversed_map_id = []
    for t, attribute in tf_data.items():
        ego_pose = np.array([attribute['pose']['translation']['x'], attribute['pose']['translation']['y']])
        for r in regions:
            if r['id'] in traversed_map_id:
                continue
            center = np.array([r['center_point']['x'], r['center_point']['y']])
            if np.sqrt(np.sum((center-ego_pose)**2)) < (r['spanning_length']/2+30):
                sub_map_region.append(r)
                traversed_map_id.append(r['id'])
                print('map id:{}'.format(r['id']))
    
    print('Load {} submap'.format(len(sub_map_region)))
    return sub_map_region


def filter_non_roi_boxes(inBoxes, boxesLen, scenes):
    '''
        filter out trajectory which used to be located in the non accessible areas
        only tra which 80% of lifetime in the non accessible areas would be removed
    '''
    tf_path = os.path.join('/data/itri_output/tracking_output/tf_localization', scenes+'.json')
    with open (tf_path, mode='r') as f:
        tf_data = json.load(f)
    
    sub_map = loadRoiMap(tf_data)    

    # {'id1':[t1, t2...], 'id2':[]}
    rm_box_time = {}
    # remove whole gt trajectory
    rm_id = []
    box_in_non_access = {}
    for region in tqdm(sub_map):
        pointList = [[pt['x'], pt['y']] for pt in region['points']]
        m_poly = Polygon([[p[0], p[1]] for p in pointList])
        center = np.array([region['center_point']['x'], region['center_point']['y']])
        for t, trks in inBoxes.items():
            if not tf_data.has_key(t):
                continue
            T_tf = transformations.quaternion_matrix(
                np.array([
                    tf_data[t]['pose']['rotation']['x'],
                    tf_data[t]['pose']['rotation']['y'],
                    tf_data[t]['pose']['rotation']['z'],
                    tf_data[t]['pose']['rotation']['w']]))
        
            T_tf[0, 3] = tf_data[t]['pose']['translation']['x']
            T_tf[1, 3] = tf_data[t]['pose']['translation']['y']
            T_tf[2, 3] = tf_data[t]['pose']['translation']['z']
            
            for trk in trks:
                # if trk['id'] in rm_id:
                #     continue
                # transform box to global coordinate
                T_local = transformations.quaternion_matrix(
                    np.array([
                        trk['track']['rotation']['x'],
                        trk['track']['rotation']['y'],
                        trk['track']['rotation']['z'],
                        trk['track']['rotation']['w']]))
            
                T_local[0, 3] = trk['track']['translation']['x']
                T_local[1, 3] = trk['track']['translation']['y']
                T_local[2, 3] = trk['track']['translation']['z']

                trk_global_pose = T_tf.dot(T_local)
                if np.sqrt(np.sum((center-trk_global_pose[:2, 3])**2)) > (region['spanning_length']/2+30):
                    continue

                global_track_dict = copy.deepcopy(trk['track'])
                global_track_dict['translation']['x'] = trk_global_pose[0, 3]
                global_track_dict['translation']['y'] = trk_global_pose[1, 3]
                global_track_dict['translation']['z'] = trk_global_pose[2, 3]
                global_track_dict['rotation']['x'] = quaternion_from_matrix(trk_global_pose)[0]
                global_track_dict['rotation']['y'] = quaternion_from_matrix(trk_global_pose)[1]
                global_track_dict['rotation']['z'] = quaternion_from_matrix(trk_global_pose)[2]
                global_track_dict['rotation']['w'] = quaternion_from_matrix(trk_global_pose)[3]
                
                trk_poly = RectToPoly(global_track_dict)
                if m_poly.intersection(trk_poly).area > (0.05*trk_poly.area):
                    if not rm_box_time.has_key(trk['id']):
                        rm_box_time[trk['id']] = []
                        box_in_non_access[trk['id']] = 0
                    rm_box_time[trk['id']].append(t)
                    box_in_non_access[trk['id']] += 1
                    # rm_id.append(trk['id'])
                    # rm_id_candidate.append(trk['id'])

                    # if trk['id'] == 202 or trk['id'] == 351:
                    #     print('poly id: {}'.format(region['id']))
                    #     print('{}: {} at {}'.format(trk['id'], m_poly.intersection(trk_poly).area, t))
                    #     print(global_track_dict)

    # get rm trajectory from overlap candidates
    for id, times in box_in_non_access.items():
        # only remove trajectory which over 80% of lifetime located in non-accessible
        if times/boxesLen[id] > 0.8:
            rm_id.append(id)
            print('{} rm for {:.2f} in non-accessible'.format(id, times/boxesLen[id]))
    
    # output
    outBoxes = {}
    counter = 0
    for t, trks in inBoxes.items():
        outBoxes[t] = []
        for trk in trks:
            if trk['id'] not in rm_id:
                outBoxes[t].append(trk)
            else:
                counter+=1

    print('Remove {} trajectories, {} boxes'.format(len(rm_id), counter))
    return outBoxes, rm_id


def config_data(gts, output_path):
    '''
        gt is configured as trajectory-based method, need frame-by-frame data frame

        return dict of list
        {
            't1':[
                    # {'track':[{'box': }, {'translation': },....], 'id':}, 
                    {'track':{'box': , 'translation': ,....}, 'id': 1}, 
                    {o2, 'id': 2}, ..
                ], 
            't2':[],
        }
    '''
    gt_data = {}
    det_data = {}
    gt_len = {}
    det_len = {}

    if len(gts['tracks']) == 0:
        print("Zero gts, abort")
        return gt_data, det_data

    for anno in gts['tracks']:
        if not anno.has_key('track'):
            print(anno)
            continue

        one_trajectory_list = anno['track']
        gt_len[anno['id']] = len(one_trajectory_list)
        for idx, gt in enumerate(one_trajectory_list):
            sup_dig = len(str(one_trajectory_list[idx]['header']['stamp']['nsecs']))
            t_step = str(one_trajectory_list[idx]['header']['stamp']['secs']) + '0'*(9-sup_dig) + str(gt['header']['stamp']['nsecs'])

            if not gt_data.has_key(t_step) :
                gt_data[t_step] = []
            
            gt_data[t_step].append({
                'track': one_trajectory_list[idx],
                'id': int(anno['id'])})

    print('gt: {}'.format(len(gt_data.keys())))

    with open(os.path.join(output_path, "gt.json"), "w") as outfile:
        json.dump(gt_data, outfile, indent = 4)

    return gt_data, gt_len


def re_congure_data(gts, rm_id, output_path, scenes):
    out_gts = {}
    out_gts = copy.deepcopy(gts)
    out_gts['tracks'] = []

    for trk in gts['tracks']:
        if trk['id'] in rm_id:
            continue
        out_gts['tracks'].append(trk)

    # with open(os.path.join(output_path, '2020-09-11-17-31-33_9_reConfig_remove_non_access_2.yaml'), 'w') as file:
    with open(os.path.join(output_path, scenes + '_reConfig_remove_non_access.yaml'), 'w') as file:
        documents = yaml.dump(out_gts, file, Dumper=MyDumper, default_flow_style=False)
    
    return out_gts



def load_gt(file_path):
    with open(file_path, "r") as gt:
        data = yaml.load(gt)
    return data


if __name__ == '__main__':
    scenes = '2020-09-11-17-37-12_1'
    # gt_path = "/data/annotation/livox_gt/done/2020-09-11-17-31-33_9_reConfig.yaml"
    gt_path = os.path.join('/data/annotation/livox_gt', scenes+'_reConfig.yaml')
    # output_path = "/data/annotation/livox_gt/done"
    output_path = "/data/annotation/livox_gt"
    gts = load_gt(gt_path)
    gt_frame, gt_len = config_data(gts, output_path)
    filter_gts, rm_id = filter_non_roi_boxes(gt_frame, gt_len, scenes)

    with open(os.path.join(output_path, "filter_gt.json"), "w") as outfile:
        json.dump(filter_gts, outfile, indent = 4)

    re_congure_data(gts, rm_id, output_path, scenes)

    