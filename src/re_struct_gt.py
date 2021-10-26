'''
It's for re-contsruct gt file output by imm to yaml which follow the certain pattern

target gt formal yaml:

tracks:[
    {   
        'id': 1,
        'track': 
        [
            {
                'translation':
                    x: 
                    y: 
                    z: 
                'rotation':
                    x: -3.872533616822136e-10
                    y: -4.613222919675441e-11
                    z: -0.003225702970434119
                    w: 0.9999947974066398
                'header':
                    'stamp':
                        {
                            'secs': t1
                            'nsecs': t1
                        }
                    'frame_id': base_link
                'label': car
                'box':
                    'length': 4.797793198376894
                    'width': 2.263856220990419
                    'height': 1.999564874917269
            },
            {
                'translation':
                    ...
                'rotation':
                    ...
                'header':
                    'stamp':
                        secs: t2
                        nsecs: t2
                    'frame_id': base_link
                'label': car
                'box':
                    ...
            },
            ...
        ]
    },
    {     
        'id': ...
        'track':
        [

        ]
    },
    ...
]           
'''


#! /usr/bin/env python2
import os
import numpy as np
import yaml
import json


# Forced output yaml to indent
class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


input_dir = '/home/ee904/catkin_ws_itri/annotation/livox_gt/unprocessed'
output_dir = '/home/ee904/catkin_ws_itri/annotation/livox_gt/processing'
segment = 14
input_file = '2020-09-11-17-31-33_'+str(segment)+'_ImmResult'
output_file = '2020-09-11-17-31-33_'+str(segment)+'_reConfig'

# with open("/home/ee904/catkin_ws_itri/annotation/2020-09-11-17-15-15_test.yaml", "r") as gt:
#     data = yaml.load(gt)

data_path = os.path.join(input_dir, input_file + '.json')
with open(data_path, 'r') as f:
    gt_data = json.load(f)

# print(gt_data['frames'].keys())

object_dict = {}
frame_id_ = ''

count = 0
for f in gt_data['frames']:
    # print(f)  
    frame_id_ = gt_data['frames'][f]['header']['frame_id']
    for obj in gt_data['frames'][f]['objects']:
        id = obj['tracking_id']

        if object_dict.get(id) == None:
            object_dict[id] = [obj]
        else:
            object_dict[id].append(obj)

    # if count == 10:
    #     # print(object_dict.keys())
    #     print(object_dict['52'])
    #     print(len(object_dict['52']))
    #     exit(0)
    # count = count + 1


# print(object_dict['1252'])
# print(len(object_dict['1252']))

# target output
object_output = {}
object_output['tracks'] = []
object_output['labels'] = ['car', 'bimo', 'truck', 'bus', 'pedestrian']
object_output['tags'] = ['moving', 'standing', 'occluded']
for obj_id, trajectory in object_dict.items():
    obj = {}
    obj['id'] = int(obj_id)
    obj['track'] = []
    # print(trajectory[0])

    for single_tra in trajectory:
        tra = {}
        tra['translation'] = single_tra['translation']
        tra['rotation'] = single_tra['rotation']
        tra['box'] = {}
        tra['box']['length'] = single_tra['size']['l']
        tra['box']['width'] = single_tra['size']['w']
        tra['box']['height'] = single_tra['size']['h']
        tra['label'] = single_tra['tracking_name']
        
        tra['header'] = {'stamp':{'secs': 0, 'nsecs': 0}, 'frame_id': frame_id_}
        tra['header']['stamp']['secs'] = int(int(single_tra['timestamp']) / 1e6)
        tra['header']['stamp']['nsecs'] = int(int(single_tra['timestamp']) % 1e6) * 1000

        obj['track'].append(tra)
    
    object_output['tracks'].append(obj)
    # exit(0) 

print(object_dict.keys())

assert len(object_dict.keys()) == len(object_output['tracks'])

# print(object_output['tracks'][0]['id'])
# print(object_output['tracks'][0]['track'])


output_path = os.path.join(output_dir, output_file + '.yaml')
with open(output_path, 'w') as file:
    documents = yaml.dump(object_output, file, Dumper=MyDumper, default_flow_style=False)

