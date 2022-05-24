#! /usr/bin/python2
# It's for visualizing the semantic of kuang-fu rd
from math import floor
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
import copy

# semantic jsons location
map_path = '/data/annotation/map/hsinchu_guandfuroad/new'
# output new non_accessible region flag
output = False

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

    for sub_map_name in map_dict.keys():
        # unit vector to shift all map
        shift_vector = [0.76070793, -0.64909432, 0]
        for poly_idx, road_poly in enumerate(map_dict[sub_map_name]):
            pts = road_poly['points']
            for index, pt in enumerate(pts):
                pt['x'] += shift_vector[0]
                pt['y'] += shift_vector[1]

def viz_roadpolys(pub, sub_map_name):
    global rl_markers
    elevation = -67 # 2020-09-11-17-37-12_4.bag

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

        # check extend pt
        pt_marker = Marker()
        pt_marker.header.frame_id = 'map'
        pt_marker.header.stamp = rospy.Time.now()
        pt_marker.action = Marker.ADD
        pt_marker.ns = sub_map_name + '_pt'
        pt_marker.type = Marker.POINTS
        pt_marker.lifetime = rospy.Duration(0.1)

        pt_marker.color.r = 1
        pt_marker.color.g = 1
        pt_marker.color.b = 0
        pt_marker.color.a = 1
        pt_marker.scale.x = 0.5
        pt_marker.scale.y = 0.5

        pt_marker.points = []
        pt_marker.id = road_poly.get('id', poly_idx) 
        
        id_marker = Marker()
        id_marker.header.frame_id = 'map'
        id_marker.header.stamp = rospy.Time.now()
        id_marker.action = Marker.ADD
        id_marker.ns = sub_map_name + '_id'
        id_marker.type = Marker.TEXT_VIEW_FACING
        id_marker.lifetime = rospy.Duration(0.1) 

        id_marker.color.r = 1
        id_marker.color.g = 1
        id_marker.color.b = 0
        id_marker.color.a = 1
        id_marker.scale.z = 5

        id_marker.id = road_poly.get('main_id', poly_idx)
        id_marker.text = str(road_poly.get('main_id', poly_idx)) 
        id_marker.pose.position.x = road_poly['points'][0]['x']
        id_marker.pose.position.y = road_poly['points'][0]['y']
        id_marker.pose.position.z = elevation

        extend_marker = Marker()
        extend_marker.header.frame_id = 'map'
        extend_marker.header.stamp = rospy.Time.now()
        extend_marker.action = Marker.ADD
        extend_marker.ns = sub_map_name + '_line'
        extend_marker.type = Marker.LINE_LIST
        extend_marker.lifetime = rospy.Duration(0.1)

        extend_marker.color.r = 1
        extend_marker.color.g = 1
        extend_marker.color.b = 0
        extend_marker.color.a = 1
        extend_marker.scale.x = 0.1

        extend_marker.points = []
        extend_marker.id = road_poly.get('id', poly_idx) 
        

        pts = road_poly['points']
        for index, pt in enumerate(pts):
             # check extend pt in non-accessible
            if pt.has_key('point_id') and pt['point_id'] == -1:
                pt_marker.points.append(Point(pt['x'], pt['y'], elevation))

                if index == 0:
                    extend_marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
                    extend_marker.points.append(Point(pts[(index+1)%len(pts)]['x'], pts[(index+1)%len(pts)]['y'], elevation))
                else:
                    extend_marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
                    extend_marker.points.append(Point(pts[(index-1)%len(pts)]['x'], pts[(index-1)%len(pts)]['y'], elevation))        
            
            # don't enclose polygon btw first and last pts
            if (not map_enclose_types[sub_map_name]) and index == len(pts)-1 and (sub_map_name != 'non_accessible'):
                break
            marker.points.append(Point(pts[index%len(pts)]['x'], pts[index%len(pts)]['y'], elevation))
            marker.points.append(Point(pts[(index+1)%len(pts)]['x'], pts[(index+1)%len(pts)]['y'], elevation))
        rl_markers.markers.append(marker)
        rl_markers.markers.append(pt_marker)
        if sub_map_name == 'non_accessible':
            rl_markers.markers.append(id_marker)
            rl_markers.markers.append(extend_marker)

            
    # pub.publish(rl_markers)


def getVector(p_start, p_end):
    return (p_end - p_start) / np.linalg.norm(p_end-p_start)

def getCloestLane(in_pt):
    nearest_pt = in_pt
    nearest_dist = np.inf
    for poly_idx, road_poly in enumerate(map_dict['roadlines']):
        pts = road_poly['points']
        for index, pt in enumerate(pts):
            pt_road = np.array([pt['x'], pt['y'], pt['z']])
            dist = np.linalg.norm(pt_road-in_pt)
            if dist > 30:
                continue

            if dist < nearest_dist:
                nearest_pt = pt_road
                nearest_dist = dist

    return nearest_pt

def mergeSeg():
    '''
    Merge 2 segs with closed end into enclosed shape
    '''
    merge_id_list = []

    for i in range(len(map_dict['non_accessible'])):
        pts_i = map_dict['non_accessible'][i]['points']
        start_i = np.array([pts_i[0]['x'], pts_i[0]['y'], pts_i[0]['z']])
        end_i = np.array([pts_i[-1]['x'], pts_i[-1]['y'], pts_i[-1]['z']])
        end_pt_i = [start_i, end_i]
        if map_dict['non_accessible'][i]['id'] in np.array(merge_id_list).flatten():
            continue

        for j in range(len(map_dict['non_accessible'])):

            if map_dict['non_accessible'][i]['id'] == map_dict['non_accessible'][j]['id']:
                continue

            if map_dict['non_accessible'][j]['id'] in np.array(merge_id_list).flatten():
                continue
            
            pts_j = map_dict['non_accessible'][j]['points']
            start_j = np.array([pts_j[0]['x'], pts_j[0]['y'], pts_j[0]['z']])
            end_j = np.array([pts_j[-1]['x'], pts_j[-1]['y'], pts_j[-1]['z']])
            end_pt_j = [start_j, end_j]

            if np.linalg.norm(start_i-end_j) < 0.5 and np.linalg.norm(end_i-start_j) < 0.5:
                merge_id_list.append([map_dict['non_accessible'][i]['id'], map_dict['non_accessible'][j]['id']])


    print('We have: {} to merge'.format(len(merge_id_list)))

    for pair in merge_id_list:
        print('id_1: {}; id:2 {}'.format(pair[0], pair[1]))

    print('Before merge: {}'.format(len(map_dict['non_accessible'])))
    
    for pair in merge_id_list:
        for i in range(len(map_dict['non_accessible'])):
            if map_dict['non_accessible'][i]['id'] == pair[1]:
                seg_merged_list = copy.deepcopy(map_dict['non_accessible'][i]['points'])
                for j in range(len(map_dict['non_accessible'])):
                    if map_dict['non_accessible'][j]['id'] == pair[0]:
                        map_dict['non_accessible'][j]['points'].extend(seg_merged_list)
                        break
                
                del map_dict['non_accessible'][i]
                break
    
    print('After merge: {}'.format(len(map_dict['non_accessible'])))


def extendSeg():
    mergeSeg()

    for poly_idx, road_poly in enumerate(map_dict['non_accessible']):
        pts = road_poly['points']

        start_pt =  np.array([pts[0]['x'], pts[0]['y'], pts[0]['z']])
        end_pt = np.array([pts[-1]['x'], pts[-1]['y'], pts[-1]['z']])
        # already closed polygon, skip
        if start_pt[0] == end_pt[0] and start_pt[1] == end_pt[1] and start_pt[2] == end_pt[2]:
            continue
        
        # open but close, <10m, just enclose, append first pt to last
        if np.linalg.norm(start_pt-end_pt) < 10:
            pt_node = {}
            pt_node = copy.deepcopy(pts[0])
            pt_node['point_id'] = -2
            pts.append(pt_node)
            continue
        
        # # not enough pt for exterpolate
        # if len(pts) < 4:
        #     continue
        

        # exterpolate pt of seg to enclose a polygon
        second_pt = np.array([pts[1]['x'], pts[1]['y'], pts[1]['z']])
        last_second_pt = np.array([pts[-2]['x'], pts[-2]['y'], pts[-2]['z']])
        start_v = getVector(second_pt, start_pt)
        end_v = getVector(last_second_pt, end_pt)

        # check if v is parallel by dot product, fail correct if vector have > 90 angle but tend to enclose
        same_ori = np.sign(np.dot(start_v, end_v))
        if same_ori <= 0:
            cloest_pt = getCloestLane(start_pt)
            road_v = getVector(start_pt, cloest_pt)
            oppo_v = 'start' if np.dot(start_v, road_v) > 0 else 'end'
            if oppo_v == 'start':
                start_v = end_v
            elif oppo_v == 'end':
                end_v = start_v

        # check if extended pt on the same side of line seg, if not, couldn't enclose a convex hull region


        head_pt = start_pt + 10 * start_v
        last_pt = end_pt + 10 * end_v

        pt_node = {}
        pt_node['point_id'] = -1
        pt_node['x'] = head_pt[0]
        pt_node['y'] = head_pt[1]
        pt_node['z'] = head_pt[2]
        pts.insert(0, pt_node)

        pt_node = {}
        pt_node['point_id'] = -1
        pt_node['x'] = last_pt[0]
        pt_node['y'] = last_pt[1]
        pt_node['z'] = last_pt[2]
        pts.append(pt_node)


def outputRegion():
    non_accessible_region = {'non_accessible_region': copy.deepcopy(map_dict['non_accessible'])}
    for poly_idx, road_poly in enumerate(non_accessible_region['non_accessible_region']):
        pts = road_poly['points']

        start_pt =  np.array([pts[0]['x'], pts[0]['y'], pts[0]['z']])
        end_pt = np.array([pts[-1]['x'], pts[-1]['y'], pts[-1]['z']])
        # already closed polygon, skip
        if start_pt[0] == end_pt[0] and start_pt[1] == end_pt[1] and start_pt[2] == end_pt[2]:
            continue
        
        # enclose extended region, append start to end
        pts.append(pts[0])

    # cal seg spanning of raw data (pt_id != negative)
    for poly_idx, road_poly in enumerate(non_accessible_region['non_accessible_region']):
        pts = road_poly['points']

        # search for ordinary enclose or tend to enclose one, using half as spanning
        raw_enclose_one = False
        # ordinary enclose
        if pts[0]['point_id'] > 0 and pts[-1]['point_id'] > 0:
            raw_enclose_one = True
            print('raw_enclose_one: {}'.format(road_poly['main_id']))
        elif pts[-1]['point_id'] == -2:
            # tend to enclose one
            raw_enclose_one = True
            print('close_one: {}'.format(road_poly['main_id']))

        if raw_enclose_one:
            mid_idx = int(floor(len(pts)/2))
            start_pt =  np.array([pts[0]['x'], pts[0]['y'], pts[0]['z']])
            end_pt = np.array([pts[mid_idx]['x'], pts[mid_idx]['y'], pts[mid_idx]['z']])
            diag = np.linalg.norm(end_pt-start_pt)
            road_poly['spanning_length'] = diag
            center = (end_pt+start_pt)/2
        else:
            # special labeled region
            if road_poly['id'] == 330:
                start_pt =  np.array([pts[-2]['x'], pts[-2]['y'], pts[-2]['z']])
                end_pt = np.array([pts[-3]['x'], pts[-3]['y'], pts[-3]['z']])
                diag = np.linalg.norm(end_pt-start_pt)
                road_poly['spanning_length'] = diag
                center = (end_pt+start_pt)/2
                road_poly['center_point'] = {'x': center[0], 'y': center[1], 'z':center[2]}
                continue
            # for two-ened expened pt (id = -1)
            # for balancing extened region, using one extend and one original pt as diagonal to cal center and spanning length
            # choose longer one
            start_pt =  np.array([pts[0]['x'], pts[0]['y'], pts[0]['z']])
            end_pt = np.array([pts[-2]['x'], pts[-2]['y'], pts[-2]['z']])

            # raw start
            for pt in pts:
                if pt['point_id'] > 0:
                    start_pt = np.array([pt['x'], pt['y'], pt['z']])
                    break

            start_pt_2 =  np.array([pts[0]['x'], pts[0]['y'], pts[0]['z']])
            end_pt_2 = np.array([pts[-1]['x'], pts[-1]['y'], pts[-1]['z']])

            # raw end
            for pt in pts:
                if pt['point_id'] > 0:
                    end_pt_2 = np.array([pt['x'], pt['y'], pt['z']])
                    break

            diag = np.linalg.norm(end_pt-start_pt)
            diag_2 = np.linalg.norm(end_pt_2-start_pt_2)

            # original enclosed ones, get half as span
            if (start_pt[0] == end_pt[0] and start_pt[1] == end_pt[1] and start_pt[2] == end_pt[2]) or road_poly["id"] == 325:
                half_idx = int(len(pts)/2)
                start_pt =  np.array([pts[half_idx]['x'], pts[half_idx]['y'], pts[half_idx]['z']])
                road_poly['spanning_length'] = diag
                center = (end_pt+start_pt)/2
            else:
                road_poly['spanning_length'] = diag if (diag > diag_2) else diag_2
                center = (end_pt+start_pt)/2 if (diag > diag_2) else (end_pt_2+start_pt_2)/2

        road_poly['center_point'] = {'x': center[0], 'y': center[1], 'z':center[2]}
    
    with open(os.path.join(map_path, 'non_accessible_region.json'), 'w') as outFile:
        json.dump(non_accessible_region, outFile, indent=4)
    print('Output to ', os.path.join(map_path, 'non_accessible_region.json'))


if __name__ == "__main__":    
    rospy.init_node("pub_map_node", anonymous=True)
    mPubRoadlines = rospy.Publisher('roadlines', MarkerArray, queue_size=100)
    
    load_map(map_path)

    for sub_map_name in map_dict.keys():
        if sub_map_name == 'non_accessible':
            extendSeg()
        viz_roadpolys(mPubRoadlines, sub_map_name)
    
    mPubRoadlines.publish(rl_markers)
    print('a', len(rl_markers.markers))

    non_accessible_region = {'non_accessible_region': map_dict['non_accessible']}
    if output:
        outputRegion()

    rate = rospy.Rate(10) 
    while not rospy.is_shutdown():
        mPubRoadlines.publish(rl_markers)
        rate.sleep()