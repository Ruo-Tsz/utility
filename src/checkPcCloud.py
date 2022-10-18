from email.header import Header
import rospy
import os, sys
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
import numpy as np
from std_msgs.msg import Header
from threading import Thread

shut_down = False
block = False


def load_pc(pc_path):
    files_list = sorted(os.listdir(pc_path))
    lidar_scans_list = [lidar_scan.split('.')[0] for lidar_scan in files_list]
    # print(lidar_scans_list)

    scans_dict = {}
    for index, f in enumerate(files_list):
        scanPath = os.path.join(pc_path, f)
        raw_scan = np.fromfile(scanPath, dtype=np.float32)
        # (N, 4)
        scan = raw_scan.reshape(-1, 4)
        scans_dict[lidar_scans_list[index]] = scan

    return scans_dict

def create_pc(pc, header):
    fields = []
    # fill sensor_msg with density
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1)]

    pc_msg = pcl2.create_cloud(header, fields, pc)
    return pc_msg

def interrupt():
    global block
    while not shut_down: 
        # raw input would block and keep waiting(stall) for input
        raw_input('press any key to pause or resume:\n')
        block = not block
        print('pause' if block else 'resume')

    print('shut_down')


if __name__ == '__main__':
    rospy.init_node("visualize_pc_node", anonymous=True)                                                                   

    scans_dict_2 = load_pc(os.path.join('/data/itri_output/tracking_output/output/livox_gt_annotate_velodyne_raw/ego_compensation/2020-09-11-17-31-33_sync_frame(v3)/pointcloud_deprecated/2020-09-11-17-31-33_13'))
    scans_dict = load_pc(os.path.join('/data/itri_output/tracking_output/kuang-fu-rd_livox_public/without ego compensation/kuang-fu-rd_v2/2020-09-11-17-31-33/pointcloud/2020-09-11-17-31-33_13'))
    scans_dict_3 = load_pc(os.path.join('/data/itri_output/tracking_output/output/livox_gt_annotate_velodyne_raw/ego_compensation/2020-09-11-17-31-33_sync_frame(v3)/pointcloud/2020-09-11-17-31-33_13'))

    mPubScans = rospy.Publisher('velodyne_points', PointCloud2, queue_size=100)
    mPubScansCompensated = rospy.Publisher('old_com_velodyne_points', PointCloud2, queue_size=100)
    mPubScansCompensatedC = rospy.Publisher('com_velodyne_points', PointCloud2, queue_size=100)

    stamps=sorted(scans_dict.keys())
    header = Header()
    header.frame_id = 'velodyne'

    # thread = Thread(target = interrupt, args = [])
    # thread.start()

    while not rospy.is_shutdown():
        rate = rospy.Rate(1)
        for stamp in stamps:
            pc_msg = create_pc(scans_dict[str(int(stamp))], header)
            pc_msg_2 = create_pc(scans_dict_2[str(int(stamp))], header)
            pc_msg_3 = create_pc(scans_dict_3[str(int(stamp))], header)

            while block:
                pass

            mPubScans.publish(pc_msg)
            mPubScansCompensated.publish(pc_msg_2)
            mPubScansCompensatedC.publish(pc_msg_3)

            rate.sleep()
        break

    shut_down = True
