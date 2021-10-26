import yaml

with open("/home/ee904/catkin_ws_itri/annotation/2020-09-11-17-31-33 (copy).yaml", "r") as gt:
    data = yaml.load(gt)

# print(data)
print(data.keys())
print(type(data['tracks']))
print(type(data['labels']))
print(type(data['tags']))
print(data['tags'])
print(type(data['tracks'][0]))
print(data['tracks'][0].keys())
print((data['tracks'][0]['track'][0].keys()))
print(data['tracks'][0]['track'][0]['header'])
exit(0)

# print(len(data['tracks']))

# list of multiple tracks id, each id is dictionary
# [{id1}, {id2}....]
for idx, id in enumerate(data['tracks']):
    # 
    # {'id': int, 
    # 'track':[{anno1}, {anno2}...]}
    # print(id['track'])

    # each gt for single instance id over time
    for anno in id['track']:
        # {anno1} =
        # {'rotation': {'x': float, 'y':float, 'z':float, 'w':float}
        #  'translation': {},
        #  'tags': ['moving', '']2020-09-11-17-15-15_test
        #  'header': {'stamp': {'secs': int, 'nsecs': int}, 'frame_id': 'base_link'}
        print(type(anno['header']['stamp']['nsecs']))
        print(type(anno['box']['width']))
        break

    break