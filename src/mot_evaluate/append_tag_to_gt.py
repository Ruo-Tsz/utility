#! /usr/bin/env python2
from __future__ import unicode_literals
import math
import os
import numpy as np
import yaml
import copy
import json
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix, quaternion_from_matrix
from tqdm import tqdm


# requirement: tf data at pc frame (gt at pc frame)
scene = '2020-09-11-17-37-12_4'
tf_path = '/data/itri_output/tracking_output/tf_localization/'+scene+'.json'
pc_path = '/data/itri_output/tracking_output/pointcloud/no_ego_compensated/2020-09-11-17-37-12/' + scene
gt_path = '/data/annotation/livox_gt/done/' + scene + '_reConfig_done_no_inter.yaml'

out_global_gt_path = '/data/annotation/livox_gt/done/' + scene + '_reConfig_done_global.yaml'
out_inter_test_gt_path = '/data/annotation/livox_gt/done/' + scene + '_reConfig_done_global_test.yaml'
out_inter_global_path = '/data/annotation/livox_gt/done/' + scene + '_reConfig_done_global_inter.yaml'
out_inter_local_path = '/data/annotation/livox_gt/done/' + scene + '_reConfig_done_local_inter.yaml'

'''
    Append tag(moving/occluded) to annotaion
    Need to load localization data of velodyne and transform gt to global fram(map) to calculate vel
    
    Local annotation may be absent during occlusion, need to be interpolated in and add a 'occluded' tag
'''

# # Forced output yaml to indent
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


tf_data = {}
with open (tf_path, 'r') as file:
    tf_data = json.load(file)
print(len(tf_data.keys()))

gt_data = {}
with open(gt_path, "r") as gt_file:
    gt_data = yaml.load(gt_file)


# Transform all track to global by tf data
global_gt_data = copy.deepcopy(gt_data)
tracks = gt_data['tracks']
global_tracks = []

for gt in tqdm(tracks):
    global_gt = copy.deepcopy(gt)
    global_tras = []
    for tra in gt['track']:
        global_tra = copy.deepcopy(tra)

        time = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
        tf_localization = {}
        if tf_data.has_key(str(time)):
            tf_localization = tf_data[str(time)]['pose']
        else:
            print('No tf data of gt at time: {}'.format(time))
            continue

        tf_pose = tf.transformations.quaternion_matrix(
            np.array([tf_localization['rotation']['x'], tf_localization['rotation']['y'], tf_localization['rotation']['z'], tf_localization['rotation']['w']]))        
        tf_pose[0, 3] = tf_localization['translation']['x']
        tf_pose[1, 3] = tf_localization['translation']['y']
        tf_pose[2, 3] = tf_localization['translation']['z']
            
        local_pose = tf.transformations.quaternion_matrix(
            np.array([tra['rotation']['x'], tra['rotation']['y'], tra['rotation']['z'], tra['rotation']['w']]))        
        local_pose[0, 3] = tra['translation']['x']
        local_pose[1, 3] = tra['translation']['y']
        local_pose[2, 3] = tra['translation']['z']

        global_pose = tf_pose.dot(local_pose)
        
        global_tra['rotation']['x'] = float(quaternion_from_matrix(global_pose)[0])
        global_tra['rotation']['y'] = float(quaternion_from_matrix(global_pose)[1])
        global_tra['rotation']['z'] = float(quaternion_from_matrix(global_pose)[2])
        global_tra['rotation']['w'] = float(quaternion_from_matrix(global_pose)[3])
        global_tra['translation']['x'] = float(global_pose[0, 3])
        global_tra['translation']['y'] = float(global_pose[1, 3])
        global_tra['translation']['z'] = float(global_pose[2, 3])

        global_tras.append(global_tra)
    
    global_gt['track'] = global_tras
    global_tracks.append(global_gt)

global_gt_data['tracks'] = global_tracks


with open(out_global_gt_path, mode='w') as outFile:
    documents = yaml.dump(global_gt_data, outFile, Dumper=MyDumper, default_flow_style=False)



def interpolateTra(index, tras, times):
    '''
        interpolate of time btw (index-1) and index of tras
        @param:
            INPUT:
                index: int, occluded times is btw (index-1) and index of tras
                tras: list of trajectory
                times: list of timestamps, which is consecutive timestamp needed to be interpolate into tras
            OUPUT:
                dict of interpolated tras

    '''
    assert index > 0, 'wrong position of insertion of trajectory'
    outTras = {}

    p1_dict = tras[index-1]
    t1 = p1_dict['header']['stamp']['secs']* 10**(9) + p1_dict['header']['stamp']['nsecs']
    (roll_1, pitch_1, yaw_1) = euler_from_quaternion ([ p1_dict['rotation']['x'], p1_dict['rotation']['y'], p1_dict['rotation']['z'], p1_dict['rotation']['w'] ])
    p1 = tf.transformations.quaternion_matrix(
        np.array([ p1_dict['rotation']['x'], p1_dict['rotation']['y'], p1_dict['rotation']['z'], p1_dict['rotation']['w'] ]))
    p1[0, 3] = p1_dict['translation']['x']
    p1[1, 3] = p1_dict['translation']['y']
    p1[2, 3] = p1_dict['translation']['z']

    p2_dict = tras[index]
    t2 = p2_dict['header']['stamp']['secs']* 10**(9) + p2_dict['header']['stamp']['nsecs']
    (roll_2, pitch_2, yaw_2) = euler_from_quaternion ([ p2_dict['rotation']['x'], p2_dict['rotation']['y'], p2_dict['rotation']['z'], p2_dict['rotation']['w'] ])
    p2 = tf.transformations.quaternion_matrix(
        np.array([ p2_dict['rotation']['x'], p2_dict['rotation']['y'], p2_dict['rotation']['z'], p2_dict['rotation']['w'] ]))
    p2[0, 3] = p2_dict['translation']['x']
    p2[1, 3] = p2_dict['translation']['y']
    p2[2, 3] = p2_dict['translation']['z']

    # may be opposite orientation wiht pi
    if math.fabs(yaw_1-yaw_2) > math.pi/2:
        if yaw_2 > yaw_1:
            yaw_2 -= math.pi
        else:
            yaw_2 += math.pi
        q_2 = quaternion_from_euler(roll_2, pitch_2, yaw_2)
        p2 = tf.transformations.quaternion_matrix(
            np.array([ q_2[0], q_2[1], q_2[2], q_2[3] ]))

    # v = (p2 - p1) / (float(t2 - t1)/10**(9))
    linear_v = (p2[:3, 3] - p1[:3, 3]) / (float(t2 - t1)/10**(9))
    yaw_v = (yaw_2 - yaw_1) / (float(t2 - t1)/10**(9))

    for t in times:
        outTras[t] = copy.deepcopy(p1_dict)
        outTras[t]['header']['stamp']['secs'] = int(t/10**(9))
        outTras[t]['header']['stamp']['nsecs'] = t%10**(9)
        dt = float(t - t1)/10**(9)
        # p = p1 + v * dt
        p = np.eye(4, dtype=float)
        p[:3, 3] = p1[:3, 3] + linear_v * dt
        yaw_p = yaw_1 + yaw_v * dt
        q = quaternion_from_euler(roll_1, pitch_1, yaw_p)

        outTras[t]['rotation']['x'] = float(q[0])
        outTras[t]['rotation']['y'] = float(q[1])
        outTras[t]['rotation']['z'] = float(q[2])
        outTras[t]['rotation']['w'] = float(q[3])
        outTras[t]['translation']['x'] = float(p[0, 3])
        outTras[t]['translation']['y'] = float(p[1, 3])
        outTras[t]['translation']['z'] = float(p[2, 3])

        # add tags
        outTras[t]['tags'] = ['occluded']
        # print(outTras[t])
    
    return outTras

def sortTime(tra):
    time = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
    return time


# Interpolate data into transformed global annotation
global_gt_data = {}
with open(out_global_gt_path, mode='r') as outFile:
    global_gt_data = yaml.load(outFile)


# import all lidar frame
allLidarFrame = []
allPCList = os.listdir(pc_path)
for file in allPCList:
    allLidarFrame.append(int(file.split('.')[0]))
list.sort(allLidarFrame)

# check if gt is absent by lidar stamp from global annotation
tracks = global_gt_data['tracks']

for gt in tqdm(tracks):
    absent = False
    timestamp = []
    for tra in gt['track']:
        time = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
        timestamp.append(time)

    list.sort(timestamp)
    start_t = sorted(timestamp)[0]
    end_t = sorted(timestamp)[-1]

    s_counter = 0
    for idx, t in enumerate(sorted(allLidarFrame)):
        if t < start_t:
            continue
        else:
            s_counter = idx
            break
    
    e_counter = 0
    for idx, t in enumerate(sorted(allLidarFrame)):
        if t < end_t:
            continue
        else:
            e_counter = idx
            break
    # print(s_counter, e_counter)
    # print(start_t, end_t)
    # print(allLidarFrame[s_counter], allLidarFrame[e_counter])
    track_life = allLidarFrame[s_counter:e_counter+1]
    # print(track_life[0], track_life[-1])
    assert start_t == track_life[0], 'Wrong track life time'
    assert end_t == track_life[-1], 'Wrong track life time'

    absent_num = len(track_life) - len(timestamp)
    # print('absent_num: {}'.format(absent_num))
    assert absent_num >= 0, 'Negative absent num'

    if absent_num == 0:
        continue

    show_up = np.full((len(track_life), ), True)
    for idx, t in enumerate(sorted(track_life)):
        if t not in timestamp:
            show_up[idx] = False
    assert (len(track_life) - np.sum(show_up)) == absent_num, 'Wrong show_up list'


    # find consecutive false stand for occlusion regions 
    lost_interval = []
    lost_time = []
    find_occluded = False
    for idx, shown in enumerate(show_up):
        if not shown:
            if not find_occluded:
                find_occluded = True
            lost_time.append(track_life[idx])

        else:
            # end of occlusion, reset the flag
            if find_occluded:
                find_occluded = False
                lost_interval.append(lost_time)
                lost_time = []

    all_inter_tras = []
    for interval in lost_interval:
        index = np.searchsorted(np.array(timestamp), interval[0]) 
        # append interval btw (index-1) and index of timestamp
        inter_tras = interpolateTra(index, gt['track'], interval)
        all_inter_tras.extend(inter_tras.values())
    # print('Inter {} tras'.format(len(all_inter_tras)))
    # print('Befor inser {} tras'.format(len(gt['track'])))
    gt['track'].extend(all_inter_tras)
    # print('After inser {} tras'.format(len(gt['track'])))

    gt['track'].sort(key=sortTime)


with open(out_inter_test_gt_path, mode='w') as outFile:
    # documents = yaml.dump(global_gt_data, outFile, allow_unicode=True, Dumper=MyDumper, default_flow_style=False)
    # cannot indent 4 but wouldn't have !!python/unicode predent
    documents = yaml.safe_dump(global_gt_data, outFile, default_flow_style=False)


# re-configure
non_indent = {}                                                                                                                                                                                      
with open(out_inter_test_gt_path, "r") as gt_file:
    non_indent = yaml.load(gt_file)
with open(out_inter_global_path, mode='w') as outFile:
    documents = yaml.dump(non_indent, outFile, allow_unicode=True, Dumper=MyDumper, default_flow_style=False)


# transform pose to local velodyne
global_inter_data = {}
with open(out_inter_global_path, "r") as gt_file:
    global_inter_data = yaml.load(gt_file)

local_gt_data = copy.deepcopy(global_inter_data)
tracks = local_gt_data['tracks']
local_tracks = []

for gt in tqdm(tracks):
    local_gt = copy.deepcopy(gt)
    local_tras = []
    for tra in gt['track']:
        local_tra = copy.deepcopy(tra)

        time = tra['header']['stamp']['secs']* 10**(9) + tra['header']['stamp']['nsecs']
        tf_localization = {}
        if tf_data.has_key(str(time)):
            tf_localization = tf_data[str(time)]['pose']
        else:
            print('No tf data of gt at time: {}'.format(time))
            continue

        tf_pose = tf.transformations.quaternion_matrix(
            np.array([tf_localization['rotation']['x'], tf_localization['rotation']['y'], tf_localization['rotation']['z'], tf_localization['rotation']['w']]))        
        tf_pose[0, 3] = tf_localization['translation']['x']
        tf_pose[1, 3] = tf_localization['translation']['y']
        tf_pose[2, 3] = tf_localization['translation']['z']
            
        global_pose = tf.transformations.quaternion_matrix(
            np.array([tra['rotation']['x'], tra['rotation']['y'], tra['rotation']['z'], tra['rotation']['w']]))        
        global_pose[0, 3] = tra['translation']['x']
        global_pose[1, 3] = tra['translation']['y']
        global_pose[2, 3] = tra['translation']['z']

        local_pose = np.linalg.inv(tf_pose).dot(global_pose)
        
        local_tra['rotation']['x'] = float(quaternion_from_matrix(local_pose)[0])
        local_tra['rotation']['y'] = float(quaternion_from_matrix(local_pose)[1])
        local_tra['rotation']['z'] = float(quaternion_from_matrix(local_pose)[2])
        local_tra['rotation']['w'] = float(quaternion_from_matrix(local_pose)[3])
        local_tra['translation']['x'] = float(local_pose[0, 3])
        local_tra['translation']['y'] = float(local_pose[1, 3])
        local_tra['translation']['z'] = float(local_pose[2, 3])

        local_tras.append(local_tra)
    
    local_gt['track'] = local_tras
    local_tracks.append(local_gt)

local_gt_data['tracks'] = local_tracks

with open(out_inter_local_path, mode='w') as outFile:
    documents = yaml.dump(local_gt_data, outFile, Dumper=MyDumper, default_flow_style=False)




# local_gt_data = {}
# with open(out_inter_local_path, mode='r') as file:
#     local_gt_data = yaml.load(file)

# occluded_gts = {}
# for gt in local_gt_data['tracks']:
#     occluded_gts[gt['id']] = []
#     for tra in gt['track']:
#         if tra.has_key('tags'):
#             for tag in tra['tags']:
#                 if tag == 'occluded':
#                     occluded_gts[gt['id']].append(tra['header']['stamp']['secs']*10**(9) + tra['header']['stamp']['nsecs'])

# all_ts = []
# for k, v in occluded_gts.items():
#     if len(v) == 0:
#         continue
#     # print("{}: {}".format(k, v))
#     for t in v:
#         if t not in all_ts:
#             all_ts.append(t)

# print(len(all_ts))

# re_config_stamp_gt = {}
# for k, v in occluded_gts.items():
#     for t in v:
#         if not re_config_stamp_gt.has_key(t):
#             re_config_stamp_gt[t] = []
#         re_config_stamp_gt[t].append(k)


# with open('/data/annotation/livox_gt/done/' + scene + '_reConfig_done_local_inter_occluded.yaml', mode='w') as outFile:
#     documents = yaml.dump(re_config_stamp_gt, outFile, Dumper=MyDumper, default_flow_style=False)

    
    
    


