#include <yaml-cpp/yaml.h>
#include <visualization_msgs/MarkerArray.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/tf.h>

ros::Subscriber sub_cloud_;
ros::Publisher pub_markers_;

string file_dir = "/home/ee904/catkin_ws_itri/annotation/2020-09-11-17-31-33.yaml";

using namespace YAML;
using namespace std;

struct TrackInstance
{
    std_msgs::Header header;
    string label;
    vector<string> tags;
    tf::StampedTransform center;
    tf::Vector3 box_size;
};


void CallbackCloud(const sensor_msgs::PointCloud2ConstPtr & msg)
{
    
}

bool LoadGt(const string& file)
{
  Node node;
  try
  {
    node = LoadFile(file);
  }
  catch (Exception const& e)
  {
    stringstream stream;
    stream << "Failed to open " << file << ": " << e.msg;
    ROS_DEBUG_STREAM(stream.str());
    return false;
  }

}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "viz_gt");
    ros::NodeHandle nh("~");

    sub_cloud_ = nh.subscribe("/velodyne_points", 1, &CallbackCloud);

    if(LoadGt(file_dir))
        cout << "Successful load file " << file_dir.c_str() << std::endl;
    else
        return -1;


    ros::Rate loop(10);
    while(ros::ok())
    {
        loop.sleep();
        ros::spinOnce;
    }
    

    return 0;
}