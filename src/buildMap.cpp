#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <signal.h>


ros::Subscriber sub_map;
ros::Publisher pub_map;
tf::TransformListener *mListener;
pcl::PointCloud<pcl::PointXYZI>::Ptr mMap (new pcl::PointCloud<pcl::PointXYZI>);
std_msgs::Header mLastHeader;


bool getTransform(const std_msgs::Header header, tf::StampedTransform& outTransform)
{
    try
    {
        mListener->waitForTransform("/map", header.frame_id, header.stamp, ros::Duration(1.0));
        mListener->lookupTransform("/map", header.frame_id,  header.stamp, outTransform);
        std::cout << "origin: " << outTransform.getOrigin().x() << ", " << outTransform.getOrigin().y() << std::endl;
        return true;
    }
    catch (tf::TransformException ex)
    {
        ROS_WARN("%s",ex.what());
        return false;
    }
}


void CloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    std::cout<<"Get cloud: " << msg->header.stamp << ",at " <<  msg->header.frame_id << std::endl;
    std::cout<<"Points: " << msg->width * msg->height << std::endl;
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_scan (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_map (new pcl::PointCloud<pcl::PointXYZI>);
    sensor_msgs::PointCloud2 local = *msg;
    sensor_msgs::PointCloud2 transformCloud;

    tf::StampedTransform transform;
    if(getTransform(msg->header, transform))
    {
        sensor_msgs::PointCloud2Ptr transformCloud(new sensor_msgs::PointCloud2);
        sensor_msgs::PointCloud2Ptr mapCloud(new sensor_msgs::PointCloud2);
        pcl_ros::transformPointCloud("/map", transform, local, *transformCloud);
        pcl::fromROSMsg(*transformCloud, *local_scan);
        *mMap += *local_scan;
        mLastHeader = transformCloud->header;

        pcl::VoxelGrid<pcl::PointXYZI> sor;
        sor.setInputCloud (mMap);
        sor.setLeafSize (0.2f, 0.2f, 0.2f);
        sor.filter (*filtered_map);

        pcl::toROSMsg(*filtered_map, *mapCloud);
        mapCloud->header.frame_id = "/map";
        pub_map.publish(*mapCloud);
        std::cout << "Map points: " << filtered_map->points.size() << std::endl;
        std::cout << "-----------\n";
    }
    else
    {
        ROS_WARN("cannot get tf to current frame");
    }
}

template <typename T>
std::string to_string_with_precision(const T a_value, const int n) 
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


void closeHandler(int sig)
{
    mMap->header.frame_id = mLastHeader.frame_id;
    mMap->header.stamp = mLastHeader.stamp.toNSec();
    std::cout << "---Close Node---\n";
    std::cout << "Save map to pcd\n";
    std::cout << "points: " << mMap->points.size() << std::endl;
    std::cout << "frame: " << mMap->header.frame_id << std::endl;
    std::cout << "stamp: " << mMap->header.stamp << std::endl;
    std::cout << "first pt: " << mMap->points.front().x << ", " << mMap->points.front().y << std::endl;
    std::string filename = to_string_with_precision(mLastHeader.stamp.toNSec(), 0);
    pcl::io::savePCDFile (filename + ".pcd", *mMap);
    ros::shutdown();   
}




int main(int argc, char** argv)
{
    ros::init(argc, argv, "build_map_node", ros::init_options::NoSigintHandler);
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    sub_map = nh.subscribe("/no_ground", 10, &CloudCallback);
    pub_map = private_nh.advertise<sensor_msgs::PointCloud2>("map_cloud", 10);
    
    mListener = new tf::TransformListener;

    signal(SIGINT, closeHandler);

    ros::Rate loop(10);
    while (ros::ok())
    {
        ros::spinOnce();
        loop.sleep();
    }
    
}