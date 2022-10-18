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
    extract occluded groundtruth from result file and calculate their IDF1 score
'''

# 2020-09-11-17-37-12_4
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/baseline_even_clustering_likelihood_pda/2022-08-13_06-43-43_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/result/occlusion_fun_base9.5_outputall/2022-08-13_07-17-25_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/immortal/2022-08-15_15-59-53_5m"

# cp
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/baseline/2022-08-18_12-04-23_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/occlusion/with ground filter/base9.5/2022-09-02_13-25-38_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_4/frame_2/Minit_4/imm/immortal/2022-08-18_12-43-04_5m"
# occluded_gt_file = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/260_16735/occluded_gt.json"
# --

# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/baseline_even_clustering_likelihood_pda/test_uncertainty_3.5/2022-08-22_02-09-10_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/result/occlusion_fun_base9.5/2022-08-22_02-16-35_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/immortal/2022-08-22_02-10-04_5m"

# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/baseline/2022-08-22_02-07-11_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/occlusion/with groundf/base9.5/2022-09-02_18-29-17_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-31-33_9/frame_2/imm/immortal/2022-08-22_02-08-05_5m"
# occluded_gt_file = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-31-33_9/113_5275_preprocessing/occluded_gt.json"
# --


# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/baseline/2022-09-03_06-24-28_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occlusion/ClipHeight_2m/original_q/9.5/2022-09-15_15-24-49_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/immortal/2022-09-03_06-30-18_5m"

# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/baseline/2022-09-03_05-56-05_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/occlusion/ClipHeight_2m/original_q/9.5/2022-09-15_16-39-56_5m"
# evaluation_path = "/data/itri_output/tracking_output/output/clustering/centerpoint/2020-09-11-17-37-12_1/imm/immortal/2022-09-03_06-01-12_5m"
# occluded_gt_file = "/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_1/occluded_gt.json"


tp_file = "tp_id_history.json"
occluded_gt = {}
tp = {}
lost_trajectory = 0
IDF1_overall = 0
IDF1s = {}
lost_id = []

with open (occluded_gt_file, 'r') as gt_file:
    occluded_gt = json.load(gt_file)

with open (os.path.join(evaluation_path, tp_file), 'r') as inFile:
    tp = json.load(inFile)

# count = 0
for id, times in occluded_gt['occluded_time'].items():
    id_dict = copy.deepcopy(tp[id]['id'])
    print(id)

    # print(id_dict)
    id_history = []
    for key in sorted(id_dict):
        print (key)
        id_history.append(id_dict[key])
    print(id_history)
    # count+=1
    # if count < 5:
    #     continue
    # else:
    #     exit(-1)

    counter = 0
    freq_id = int(id)
    # id_list = self.id_history[gid]
    for det_id in id_history:
        if np.isnan(det_id): 
            continue
        curr_frequency = id_history.count(det_id)
        if(curr_frequency> counter):
            counter = curr_frequency
            freq_id = det_id
    if counter == 0:
        lost_trajectory += 1
        print('Not tracked, skip')
        lost_id.append(id)
        continue

    print("main id {}".format(freq_id))
    main_hypo_id = freq_id
    TP_pos = [det_id==main_hypo_id for det_id in id_history]
    FN_pos = [not e for e in TP_pos]
    FP_pos = [(det_id!=main_hypo_id) and (not np.isnan(det_id)) for det_id in id_history]

    # IDP, IDR, IDF1
    # If gt is not tracking at all, skip otherwise raise error(tp=fp=0)
    # if his.count(True) == 0:
    #     # print('gt {} isn\'t tracked at all'.format(gid))
    #     lost_trajectory += 1
    #     continue

    idtp = TP_pos.count(True)
    idfn = FN_pos.count(True)
    idfp = FP_pos.count(True)
    assert idtp == counter, 'error in idtp'
    assert (idtp + idfn) == len(id_history), 'error in IDF1'

    IDP = (idtp / (idtp+idfp))
    IDR = (idtp / (idtp+idfn))
    print(idtp, idfn, idfp)
    print(IDP, IDR)
    IDF1 = (2*IDP*IDR / (IDP+IDR))
    IDF1s[id] = IDF1
    IDF1_overall += IDF1

IDF1_overall /= occluded_gt['occluded_trajectory_num']

output_dict = {
    "overall IDF1" : IDF1_overall,
    "meta" : IDF1s,
    "occluded_trajectory_num": occluded_gt['occluded_trajectory_num'],
    "trajectory_num" : occluded_gt['trajectory_num'],
    "occluded_num" : occluded_gt['occluded_num'],
    "gt_num" : occluded_gt['gt_num'],
    "lost num" : len(lost_id),
    "lost id" : lost_id}


with open(os.path.join(evaluation_path, "occluded_idf1.json"), "w") as outfile:
    json.dump(output_dict, outfile, indent = 4)