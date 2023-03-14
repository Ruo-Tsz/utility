import numpy as np
import json
import yaml
import os



original_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/test_uncertainty_3.5/occlu_pda_lifetime/occlusion_fun_base6_outputall'
newer_path = '/data/itri_output/tracking_output/output/clustering/merge_detector/v3_map/frame_num_2/2020-09-11-17-37-12_4/270_16607_preprocessing/occlusion_fun_mMinPt0_mOccupiedTh0.3_mAngResol1/without_uncertain_3.5/occlusion_fun_base6_outputall'
gt_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done_local_inter.yaml"
scene_file = '2020-09-11-17-37-12_4_ImmResult.json'

# test = '/data/itri_output/tracking_output/output'



original_file = {}
newer_file = {}
gt = {}


with open (os.path.join(original_path, scene_file), 'r') as inFile:
    original_file = json.load(inFile)['frames']

with open (os.path.join(newer_path, scene_file), 'r') as inFile:
    newer_file = json.load(inFile)['frames']

with open (gt_path, 'r') as inFile:
    gt = yaml.load(inFile)


# test_file= {}
# with open (os.path.join(test, scene_file), 'r') as inFile:
#     test_file = json.load(inFile)['frames']


AllOriStamp = []
original_count = 0
AllOriCount = []
for time, dict in original_file.items():
    AllOriStamp.append(time+'0'*3)
    AllOriCount.append(len(dict['objects']))
    for obj in dict['objects']:
        original_count+=1

AllnewStamp = []
newer_count = 0
AllnewCount = []
for time, dict in newer_file.items():
    AllnewStamp.append(time+'0'*3)
    AllnewCount.append(len(dict['objects']))
    for obj in dict['objects']:
        newer_count+=1

# AlltestStamp = []
# test_count = 0
# AlltestCount = []
# for time, dict in test_file.items():
#     AlltestStamp.append(time+'0'*3)
#     AlltestCount.append(len(dict['objects']))
#     for obj in dict['objects']:
#         test_count+=1


print(AllOriCount)
print(AllnewCount)


print('ori: {}'.format(original_count))
print('new: {}'.format(newer_count))
# print('test: {}'.format(test_count))


# collect gt frame and det frame
AllGtStamp = []
for gts in gt['tracks']:
    one_trajectory_list = gts['track']
    for gt in one_trajectory_list:
        time = str(gt['header']['stamp']['secs']*(10**9) + gt['header']['stamp']['nsecs'])
        AllGtStamp.append(time)

print('GT  frame: {}'.format(len(AllGtStamp)))
print('ori frame: {}'.format(len(AllOriStamp)))
print('new frame: {}'.format(len(AllnewStamp)))

print('GT - older: {}'.format(set(AllGtStamp) - set(AllOriStamp)))
print('GT - newer: {}'.format(set(AllGtStamp) - set(AllnewStamp)))

print('older - GT: {}'.format(set(AllOriStamp) - set(AllGtStamp)))
print('newer - GT: {}'.format(set(AllnewStamp) - set(AllGtStamp)))
# print('test  - GT: {}'.format(set(AlltestStamp) - set(AllGtStamp)))



