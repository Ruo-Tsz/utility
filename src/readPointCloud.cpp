#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <iostream>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::sort, std::stable_sort
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

//sensor_msgs::PointCloud2Iterator<float> iter_x, iter_y, iter_z, iter_intensity, iter_time;
//sensor_msgs::PointCloud2Iterator<uint16_t> iter_ring;

/*
velodyne HDL-32E rotate in clockwise manner, azimuth is with respect to y-axis 
NOTE: Due to rotation delay, one revolution is NOT EXACT equal to 360 degrees.
The start azimuth of each scan may drift.
*/

ros::Subscriber sub;
ros::Publisher pubEarliestBlock;
std::string currentFrame;


void publishEarliestBlock(const float& startAngle)
{
	visualization_msgs::Marker BlockPt;
	BlockPt.header.frame_id = currentFrame;
	BlockPt.header.stamp = ros::Time::now();
	BlockPt.type = visualization_msgs::Marker::LINE_STRIP;
	BlockPt.action = visualization_msgs::Marker::ADD;
	BlockPt.color.g = 1.0f;
	BlockPt.color.a = 1.0f;
	BlockPt.scale.x = 0.03f;

	geometry_msgs::Point pt;
	for(size_t i = 0; i < 10; i++)
	{
		pt.x = i * std::sin(startAngle * M_PI / 180.0);
		pt.y = i * std::cos(startAngle * M_PI / 180.0);
		pt.z = 0;
		BlockPt.points.push_back(pt);
	}
	pubEarliestBlock.publish(BlockPt);
}

void sortAngleByTime(
	const std::vector<float>& angles,
	sensor_msgs::PointCloud2Iterator<float> iter_time)
{
	std::vector<float> times;
	for(; iter_time != iter_time.end(); ++iter_time)
	{
		times.push_back(*iter_time);
	} 
	
	// initialize original index locations
  	std::vector<int> idx(times.size());
  	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	// using std::stable_sort instead of std::sort
	// to avoid unnecessary index re-orderings
	// when v contains elements of equal values 
  	stable_sort(idx.begin(), idx.end(),
       [&times](int i1, int i2) {return times[i1] < times[i2];});

	std::vector<float> sortedAngles;
	for(size_t i = 0; i < idx.size(); i++)
	{
		sortedAngles.push_back(angles[idx[i]]);
	}
	for(size_t i = 0; i < sortedAngles.size(); i++)
	{
		//std::cout << sortedAngles[i] << ": " << times[idx[i]] << std::endl;
	}
	std::cout << std::endl;
	std::cout << "Times size: " << times.size() << ", angles size: " << sortedAngles.size() << std::endl;
	std::cout << "The FOV in single scan: \033[1;32m" << (sortedAngles.back() - sortedAngles.front()) + 360 << "\033[0m degs.\n";
	std::cout << "Start angle w.r.t y-axis is: " << sortedAngles[0] << " degs."<< std::endl;

	publishEarliestBlock(sortedAngles[0]);
}

void getFieldData(sensor_msgs::PointCloud2& inCloud)
{
    sensor_msgs::PointCloud2Iterator<float> iter_x (inCloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y (inCloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z (inCloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_intensity (inCloud, "intensity");
    sensor_msgs::PointCloud2Iterator<uint16_t> iter_ring(inCloud, "ring");
    sensor_msgs::PointCloud2Iterator<float> iter_time(inCloud, "time");
	/*
	iter_x = sensor_msgs::PointCloud2Iterator<float>(inCloud, "x");
    iter_y = sensor_msgs::PointCloud2Iterator<float>(inCloud, "y");
    iter_z = sensor_msgs::PointCloud2Iterator<float>(inCloud, "z");
    iter_intensity = sensor_msgs::PointCloud2Iterator<float>(inCloud, "intensity");
    iter_ring = sensor_msgs::PointCloud2Iterator<uint16_t >(inCloud, "ring");
    iter_time = sensor_msgs::PointCloud2Iterator<float >(inCloud, "time");
	*/

	std::vector<float> angles;
	float maxAngle = std::numeric_limits<float>::min();
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
		float azimuth = std::atan2(*iter_x, *iter_y) * 180.0 / M_PI;
        if (azimuth < 0.0)
            azimuth += 360.0;
		if (azimuth > maxAngle)
			maxAngle = azimuth;
	//	std::cout << "angle: " << azimuth << ", ring: " << *iter_ring << ", time: " << *iter_time << std::endl;
		
		angles.push_back(azimuth);
    }
	/*
	for(size_t i = 0; i < angles.size(); i++)
	{
		std::cout << angles[i] << " ";
	}
	*/
	sortAngleByTime(angles, iter_time);

	std::cout << "Max azimuth in this scan: \033[1;35m" << maxAngle << "\033[0m" << std::endl;

}


void callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
	std::cout << "Get lidar at: " << msg->header.stamp << std::endl;
	currentFrame = msg->header.frame_id;
	sensor_msgs::PointCloud2 msg_copy = *msg;
	
	getFieldData(msg_copy);
	
	std::cout << "Done\n";
	std::cout << "---------------------------\n";
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "readPointCloud");
	ros::NodeHandle nh;
	
	
    sub = nh.subscribe("velodyne_points", 1000, &callback);
	pubEarliestBlock = nh.advertise<visualization_msgs::Marker>("earliest_block", 1000);

	ros::Rate loop(10);	
	while(ros::ok())
	{
		ros::spin();
		loop.sleep();
	}
	return 0;
}
