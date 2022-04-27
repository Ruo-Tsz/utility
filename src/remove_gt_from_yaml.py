#! /usr/bin/env python2
import yaml
import copy
import numpy as np

gt = {}

remove_id_list = [
224,
375,
432,
464,
403,
553,
533]

# dict with {merge: [mergees]}
merge_track_id = {}

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

    file_path = "/data/annotation/livox_gt/processing/2020-09-11-17-31-33_14_reConfig_official.yaml"
    output_path = "/data/annotation/livox_gt/processing/2020-09-11-17-31-33_14_reConfig_official_merge.yaml"
    gt = load_gt(file_path)

    # id = 1306
    # find = False
    # track_dict = {}
    # time_list = []
    # for track in gt['tracks']:
    #     for instance in track['track']:
    #         time_stamp = instance['header']['stamp']['secs']*1e9 + instance['header']['stamp']['nsecs']
    #         if time_stamp not in time_list:
    #             time_list.append(time_stamp)
    #     if track['id'] == 1306 and (not find):
    #         track_dict = track['track'][0]
    #         find = True

    # # print(time_list)
    # print(len(time_list))

    # time_list = sorted(time_list)

    # track_duplicate = []
    # for time in time_list:
    #         # header.stamp.secs = int(int(stamp) / 1e9)
    #         # header.stamp.nsecs= round(int(stamp) % 1e9 /1e3) * 1e3
    #     instance = copy.deepcopy(track_dict)
    #     instance['header']['stamp']['secs'] = int(int(time) / 1e9)
    #     instance['header']['stamp']['nsecs'] = int(round(int(time) % 1e9 / 1e3) * 1e3)
    #     track_duplicate.append(instance)

    # for track in gt['tracks']:
    #     if track['id'] == 1306:
    #         track['track'] = track_duplicate
    
    

    # with open(output_path, 'w') as file:
    #     documents = yaml.dump(gt, file, Dumper=MyDumper, default_flow_style=False)    
    # exit(0)


    out_gt = copy.deepcopy(gt)
    out_gt['tracks'] = []

    print('Before remove {} tracks'.format(len(remove_id_list)))
    print(len(gt['tracks']))

    for gt in gt['tracks']:
        if gt['id'] in remove_id_list:
            continue
        out_gt['tracks'].append(gt)
    
    print('After remove {} tracks'.format(len(remove_id_list)))
    print(len(out_gt['tracks']))

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

    print( merged_id_list )
    print(merged_track.keys())
    print(merge_track_id.keys())
        
    merged_gt = copy.deepcopy(non_merged_gt)
    # merge
    for idx, gt in enumerate(merged_gt['tracks']):    
        if gt['id'] in merge_track_id.keys():
            ori_time = gt['track'][-1]['header']['stamp']['secs']*1e9 + gt['track'][-1]['header']['stamp']['nsecs']
            print('\nBefore merge {} last timestamp {:d}'.format(gt['id'], int(ori_time)))

            for merged_id in merge_track_id[gt['id']]:
                # check merged timestamp
                start_idx = getNearestIdx(gt['track'], merged_track[merged_id])

                if start_idx == (len(merged_track[merged_id])-1):
                    print('skip merge for {}'.format(merged_id))
                    continue
                
                gt['track'] += merged_track[merged_id][start_idx:]

                last_time = gt['track'][-1]['header']['stamp']['secs']*1e9 + gt['track'][-1]['header']['stamp']['nsecs']
                print('merge {} into {} with last timestamp {:d}'.format(merged_id, gt['id'], int(last_time)))


    with open(output_path, 'w') as file:
        documents = yaml.dump(merged_gt, file, Dumper=MyDumper, default_flow_style=False)
        
        
    




