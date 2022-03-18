import yaml
import rospy
from visualization_msgs.msg import Marker, MarkerArray

'''
    It's for loading gt in yaml format and publish it
'''
'''
Format in yaml data
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
        #  'tags': ['moving', '']
        #  'header': {'stamp': {'secs': int, 'nsecs': int}, 'frame_id': 'base_link'}
        print(type(anno['header']['stamp']['nsecs']))
        print(type(anno['box']['width']))
        break

    break
'''


gt = {}
gt_markers = MarkerArray()


def load_gt(file_path):
    with open(file_path, "r") as gt:
        data = yaml.load(gt)
        return data

def createMarkerArray(gt):
    markers = MarkerArray()

    if len(gt['tracks']) == 0:
        return markers

    for anno in gt['tracks']:
        marker = Marker()
        marker.header.frame_id = anno['track'][0]['header']['frame_id']
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.ns = 'env'
        marker.type = Marker.CUBE
        marker.lifetime = rospy.Duration(0.1)
        marker.color.a = 0.5
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0

        marker.id = anno['id']
        marker.pose.position.x = anno['track'][0]['translation']['x']
        marker.pose.position.y = anno['track'][0]['translation']['y']
        marker.pose.position.z = anno['track'][0]['translation']['z']
        marker.pose.orientation.x = anno['track'][0]['rotation']['x']
        marker.pose.orientation.y = anno['track'][0]['rotation']['y']
        marker.pose.orientation.z = anno['track'][0]['rotation']['z']
        marker.pose.orientation.w = anno['track'][0]['rotation']['w']
        marker.scale.x = anno['track'][0]['box']['length']
        marker.scale.y = anno['track'][0]['box']['width']
        marker.scale.z = anno['track'][0]['box']['height']
        markers.markers.append(marker)

    return markers


if __name__ == "__main__":
    rospy.init_node("pub_environment_node", anonymous=True)
    mPubEnv = rospy.Publisher('Env', MarkerArray, queue_size=100)
    
    file_path = "/data/annotation/environment_37.yaml"
    gt = load_gt(file_path)

    print(type(gt['tracks']))
    print(gt['tracks'])

    gt_markers = createMarkerArray(gt)

    mPubEnv.publish(gt_markers)
    print('a', len(gt_markers.markers))

    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        mPubEnv.publish(gt_markers)
        rate.sleep()