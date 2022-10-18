#! /usr/bin/env python2
import yaml
import copy
import numpy as np

gt = {}

# 15861
# remove_id_list = [66, 149, 155, 286, 339, 977, 1132, 1129]
remove_id_list = []


#setting dict with {merger: [mergees]} {5: [312], 89: [737, 766], ... }, append mergee at the end of merger
# merge_track_id = {148: [640], 1104: [1269], 383:[743], 677:[811], 862:[1189, 1257], 1198:[1291], 300:[679]}
merge_track_id = {1515: [534],
1519: [1163],
1521: [1099]}

# record mergee track info
merged_track = {}

# Forced output yaml to indent
class MyDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

def load_gt(file_path):
    with open(file_path, "r") as gt_file:
        data = yaml.load(gt_file)
        return data


def getNearestIdx(merger, mergee):
    # keep merger and concatenate mergee to merger with non-overlap timestamp
    start_stamp = mergee[0]['header']['stamp']['secs']*1e9 + mergee[0]['header']['stamp']['nsecs']
    end_stamp = merger[-1]['header']['stamp']['secs']*1e9 + merger[-1]['header']['stamp']['nsecs']
    idx = 1
    while start_stamp <= end_stamp and idx < len(mergee):
        start_stamp = mergee[idx]['header']['stamp']['secs']*1e9 + mergee[idx]['header']['stamp']['nsecs']
        idx+=1
    return (idx-1)

if __name__ == "__main__":

    # file_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done_no_inter.yaml"
    # output_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done_merge.yaml"
    file_path = "/data/annotation/livox_gt/2020-09-11-17-37-12_1_reConfig.yaml"
    output_path = "/data/annotation/livox_gt/2020-09-11-17-37-12_1_reConfig_merge.yaml"
    # file_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_done_local_inter.yaml"
    # output_path = "/data/annotation/livox_gt/done/2020-09-11-17-37-12_4_reConfig_merge.yaml"
    gt = load_gt(file_path)

    # # duplicate static object annotation with few manually labels to whole frame 
    # print('Before duplicate {} tracks'.format(len(gt['tracks'])))
    # # duplicate_ids = [75, 85, 42, 1317, 1318, 1319, 1320, 1321, 1322]
    # duplicate_ids = [1328]

    # for duplicate_id in duplicate_ids:
    #     print('Duplicate {}'.format(duplicate_id))
    #     find = False
    #     track_dict = {}
    #     time_list = []
    #     for track in gt['tracks']:
    #         for instance in track['track']:
    #             time_stamp = instance['header']['stamp']['secs']*1e9 + instance['header']['stamp']['nsecs']
    #             if time_stamp not in time_list:
    #                 time_list.append(time_stamp)
    #         if track['id'] == duplicate_id and (not find):
    #             # record latest pos
    #             # track_dict = track['track'][-1]
    #             track_dict = track['track'][0]
    #             find = True

    #     print(len(time_list))

    #     time_list = sorted(time_list)

    #     track_duplicate = []
    #     for time in time_list:
    #             # header.stamp.secs = int(int(stamp) / 1e9)
    #             # header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    #         instance = copy.deepcopy(track_dict)
    #         instance['header']['stamp']['secs'] = int(int(time) / 1e9)
    #         instance['header']['stamp']['nsecs'] = int(round(int(time) % 1e9 / 1e3) * 1e3)
    #         track_duplicate.append(instance)

    #     for track in gt['tracks']:
    #         if track['id'] == duplicate_id:
    #             track['track'] = track_duplicate
        
    
    # print('After duplicate {} tracks'.format(len(gt['tracks'])))
    # with open(output_path, 'w') as file:
    #     documents = yaml.dump(gt, file, Dumper=MyDumper, default_flow_style=False)    
    # exit(0)


    out_gt = copy.deepcopy(gt)
    out_gt['tracks'] = []

    print('Before remove {} tracks'.format(len(remove_id_list)))
    print(len(gt['tracks']))

    for gt in gt['tracks']:
        if gt['id'] in remove_id_list:
            print('Remove: {}'.format(gt['id']))
            continue
        out_gt['tracks'].append(gt)
    
    print('After remove {} tracks'.format(len(remove_id_list)))
    print(len(out_gt['tracks']))

    # id_count = 0
    # for gt in out_gt['tracks']:
    #     if gt['id'] >= id_count:
    #         id_count = gt['id']
    # print('Final id is {}'.format(id_count))

    merged_id_list = []
    for id_list in merge_track_id.values():
        for id in id_list:
            merged_id_list.append(id)

    non_merged_gt = copy.deepcopy(out_gt)
    non_merged_gt['tracks'] = []
    # record merged_track
    for gt in out_gt['tracks']:
        if gt['id'] in merged_id_list:
            merged_track[gt['id']] = gt['track']
        else:
            non_merged_gt['tracks'].append(gt)

    # print( merged_id_list )
    # print('mergees: ', merged_track.keys())
    print('mergers:' ,merge_track_id.keys())
        
    merged_gt = copy.deepcopy(non_merged_gt)
    # merge
    for idx, gt in enumerate(merged_gt['tracks']):    
        if gt['id'] in merge_track_id.keys():
            ori_time = str(gt['track'][-1]['header']['stamp']['secs']) + str(gt['track'][-1]['header']['stamp']['nsecs'])
            print('\nBefore merge {} last timestamp {}'.format(gt['id'], ori_time))

            for merged_id in merge_track_id[gt['id']]:
                # check merged timestamp
                start_idx = getNearestIdx(gt['track'], merged_track[merged_id])

                if start_idx == (len(merged_track[merged_id])-1):
                    print('skip merge for {}'.format(merged_id))
                    continue
                
                gt['track'] += merged_track[merged_id][start_idx:]

                last_time = str(gt['track'][-1]['header']['stamp']['secs']) + str(gt['track'][-1]['header']['stamp']['nsecs'])
                print('merge {} into {} with last timestamp {}'.format(merged_id, gt['id'], last_time))

    print('After merge {} tracks'.format(len(merged_gt['tracks'])))
    with open(output_path, 'w') as file:
        documents = yaml.dump(merged_gt, file, Dumper=MyDumper, default_flow_style=False)
        
        
    




