'''
It's for extract pointcloud from bag and store in .bin file
Also include test reading and pub stored .bin cloud
'''


#! /usr/bin/env python2
import rospy
import rosbag
import os
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import pypcd
import numpy as np
import argparse

# output_bin_dir
output_file = '/data/itri_output/tracking_output/pointcloud/ego_compensated/2020-09-11-17-37-12/2020-09-11-17-37-12_1'
def msgToBin(msg):
    stamp = msg.header.stamp
    output_bin_name = str(stamp.to_nsec())
    # print( type(stamp.to_nsec()))

    pc = pypcd.PointCloud.from_msg(msg)
    # (N,) array
    x = pc.pc_data['x']
    y = pc.pc_data['y']
    z = pc.pc_data['z']
    intensity = pc.pc_data['intensity']
    # (N*4,) array
    arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
    # array need to in order [x1, y1, z1, i1, x2, y2, z2, i2...]
    # fill arr[0, 4, 8...]
    arr[::4] = x
    # fill arr[1, 5, 9...] 
    arr[1::4] = y
    arr[2::4] = z
    # kitti format intensity [0, 1], not [0, 255]
    arr[3::4] = intensity / 255 
    # print(arr.shape)
    # print(intensity.shape)
    # save it in a binary file using a float32 format
    arr.astype('float32').tofile(os.path.join(output_file, output_bin_name+'.bin'))


def callback(msg):
    print('Get data {}'.format(msg.header.stamp))
    msgToBin(msg)
    

def listener():
    rospy.init_node('pub_bin_node', anonymous=True)
    rospy.Subscriber("compensated_velodyne_points", PointCloud2, callback)
    rospy.spin()

if __name__ == "__main__":
    listener()