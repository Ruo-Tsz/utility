#! /usr/bin/python2
# It's for visualizing the semantic of kuang-fu rd
import rospy
import json
import os
from tf.transformations import quaternion_from_matrix, quaternion_matrix, translation_from_matrix
from tf import transformations
from jsk_recognition_msgs.msg import BoundingBoxArray, BoundingBox
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from std_msgs.msg import Header

# semantic jsons location
map_path = '/data/annotation/map/hsinchu_guandfuroad/new'

# submap viz setting
map_types = ['roadlines', 'pedestrian_crossing', 'curbs', 'no_temperate_parking_zone', 'parking_space', 'roadmarkers', 'non_accessible']
# showing if the pts in each map is enclosed polygons or open segments
map_enclose_types = {
    'roadlines': False, 
    'pedestrian_crossing': True,
    'curbs': False,
    'no_temperate_parking_zone': True,
    'parking_space': True,
    'roadmarkers': True,
    'non_accessible': False}
map_colors = {
    'roadlines': [0.7, 0, 0.7], 
    'pedestrian_crossing': [0, 0, 1],
    'curbs': [0, 1, 0], 
    'no_temperate_parking_zone': [1, 0, 0],
    'parking_space': [1, 1, 0],
    'roadmarkers': [0.7, 0.7, 1],
    'non_accessible': [0, 1, 1]}

map_dict = {}
rl_markers = MarkerArray()


def load_map(result_file):
    global map_dict

    for sub_map_name in map_types:
        try:
            with open (os.path.join(result_file, sub_map_name+'.json'), mode='r') as f:
                map_dict[sub_map_name] = json.load(f)[sub_map_name]
            print('submap: {}, len: {}'.format(sub_map_name, len(map_dict[sub_map_name])))
        except Exception as error: 
            print(error)


def viz_roadpolys(pub, sub_map_name):
    global rl_markers

    for poly_idx, road_poly in enumerate(map_dict[sub_map_name]):
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.ADD
        marker.ns = sub_map_name
        marker.type = Marker.LINE_LIST
        marker.lifetime = rospy.Duration(0.1)

        marker.color.r = map_colors[sub_map_name][0]
        marker.color.g = map_colors[sub_map_name][1]
        marker.color.b = map_colors[sub_map_name][2]
        marker.color.a = 1
        marker.scale.x = 0.1

        marker.points = []
        marker.id = road_poly.get('id', poly_idx) 

        pts = road_poly['points']
        for index, pt in enumerate(pts):
            # don't enclose polygon btw first and last pts
            if (not map_enclose_types[sub_map_name]) and index == len(pts)-1:
                break
            marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], pts[index%len(pts)]['z']))
            marker.points.append(Point(pts[(index+1)%len(pts)]['x'], pts[(index+1)%len(pts)]['y'], pts[(index+1)%len(pts)]['z']))
        rl_markers.markers.append(marker)
            
    # pub.publish(rl_markers)


if __name__ == "__main__":    
    rospy.init_node("pub_map_node", anonymous=True)
    mPubRoadlines = rospy.Publisher('roadlines', MarkerArray, queue_size=100)
    
    load_map(map_path)

    for sub_map_name in map_dict.keys():
        viz_roadpolys(mPubRoadlines, sub_map_name)
    
    mPubRoadlines.publish(rl_markers)
    print('a', len(rl_markers.markers))

    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        mPubRoadlines.publish(rl_markers)
        rate.sleep()