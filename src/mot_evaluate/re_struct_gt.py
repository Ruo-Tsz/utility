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

id and each track need to be ordered accordingly
'''


#! /usr/bin/env python2
from __future__ import unicode_literals
import os
import numpy as np
import yaml
import json


# Forced output yaml to indent
class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

# representer that handles the unicode to str conversion:
def my_unicode_repr(self, data):
    return self.represent_str(data.encode('utf-8'))


input_dir = '/data/itri_output/tracking_output/output/unprocessed/2020-09-11-17-37-12'
output_dir = '/data/annotation/livox_gt/processing/2020-09-11-17-37-12'
segment = 3
input_file = '2020-09-11-17-37-12_'+str(segment)+'_ImmResult'
output_file = '2020-09-11-17-37-12_'+str(segment)+'_reConfig'


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
        id = int(obj['tracking_id'])

        if object_dict.get(id) == None:
            object_dict[id] = [obj]
        else:
            object_dict[id].append(obj)

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

# print(object_dict.keys())

# order trajectory of each tracker by timestamp
for obj in object_output['tracks']:
    tra_list = obj['track']
    orderedtra = sorted(tra_list, key = lambda i: (i['header']['stamp']['secs']*pow(10, 9)+i['header']['stamp']['nsecs']))
    obj['track'] = orderedtra

assert len(object_dict.keys()) == len(object_output['tracks'])


yaml.representer.Representer.add_representer(unicode, my_unicode_repr)
output_path = os.path.join(output_dir, output_file + '.yaml')
with open(output_path, 'w') as file:
    documents = yaml.dump(object_output, file, Dumper=MyDumper, default_flow_style=False)

