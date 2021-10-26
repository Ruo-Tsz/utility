#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>

ros::Publisher pub_cloud;
std::string filename;
// /home/ee904/catkin_ws_itri/annotation/map_data_31_10sec/1599816705192072000.pcd


bool readPcdFile(const std::string& filename, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
    if (pcl::io::loadPCDFile<pcl::PointXYZI> (filename, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return false;
    }
    std::cout << "Loaded "
        << cloud->points.size()
        << " data points from " << filename << std::endl;
    return true;
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "read_pcd_node");
    ros::NodeHandle private_nh("~");
    ros::NodeHandle nh;

    pub_cloud = private_nh.advertise<sensor_msgs::PointCloud2>("/velodyne_points", 1);
    nh.param<std::string>("filename", filename, "/home/ee904/catkin_ws_itri/annotation/map_data_31_10sec/1599816705192072000.pcd");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 out_cloud_msg;
    
    if(!readPcdFile(filename, cloud))
    {
        std::cout << "Failed to open pcd file, exit\n";
        return -1;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::VoxelGrid<pcl::PointXYZI> sor;
    sor.setInputCloud (cloud);
    sor.setLeafSize (0.2f, 0.2f, 0.2f);
    sor.filter (*filtered_cloud);

    pcl::toROSMsg(*filtered_cloud, out_cloud_msg);
    out_cloud_msg.header.frame_id = "/base_link";
    out_cloud_msg.header.stamp = ros::Time::now();

    std::cout << "Pub at: " << out_cloud_msg.header.frame_id << std::endl;

    ros::Rate loop(10);
    while (ros::ok())
    {
        pub_cloud.publish(out_cloud_msg);
        loop.sleep();
    }
    
    return (0);
}