#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <itri_msgs/DetectedObjectArray.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>


struct colors
{
    float r;
    float g;
    float b;
};
std::map<int, colors> colorMap_;
int current_id = 0;


ros::Subscriber sub;
ros::Publisher pub;
std_msgs::Header mheader;
tf::TransformListener *tf_listener_;
tf::StampedTransform transform;
bool initialize = false;
std::vector<std::vector<geometry_msgs::Point>> trajectory_list;

int searchLargestId(const std::vector<itri_msgs::DetectedObject>& objs)
{
    std::cout << "searchLargestId\n"; 
    size_t id = std::numeric_limits<uint32_t>::min();
    for(const auto& obj : objs)
    {
        std::cout << obj.id <<std::endl;
        if(obj.id > id)
            id = obj.id;
        std::cout << id <<std::endl;
    }
    return id;
}


void createTraList(const std::vector<itri_msgs::DetectedObject>& objs)
{
    // try
    // {
    //     tf_listener_->waitForTransform(
    //         "/map", "/base_link", mheader.stamp, ros::Duration(1.0));
    //     tf_listener_->lookupTransform(
    //         "/map", "/base_link", mheader.stamp, transform);
    // }
    // catch (tf::TransformException ex)
    // {
    //     ROS_WARN("%s",ex.what());
    // }

    int objs_num = searchLargestId(objs);
    std::cout << "Get max_id: " << objs_num << std::endl;
    std::vector<std::vector<geometry_msgs::Point>> tra_list(objs_num+1);
    std::cout << "tra_list size: " << tra_list.size() << std::endl;
    for (size_t i = 0; i < objs.size(); i++)
    {
        float hullz = (objs[i].convex_hull.polygon.points[0].z +
        objs[i].convex_hull.polygon.points.back().z) / 2.0f;
        tf::Point pt(
            0.0,
            0.0,
            hullz);
        tf::Point ptTransForm = transform * pt;
        
        int id = objs[i].id;
        std::cout << "obj id: " << id << std::endl;
        tra_list[id] = objs[i].history_path;

        for(size_t j = 0; j < tra_list[id].size(); j++)
        {
            tra_list[id][j].z = ptTransForm.z();
        }
    }
    trajectory_list = tra_list;
    std::cout << "trajectory_list size: " << trajectory_list.size() << std::endl;
    std::cout << "--------------\n";
}

void appendHistory(const std::vector<itri_msgs::DetectedObject>& objs)
{
    createTraList(objs);
    for(size_t i = 0; i < objs.size(); i++)
    {
        int id = objs[i].id;
        if(id <= trajectory_list.size()-1)
            trajectory_list[id] = objs[i].history_path;
        else
        {
            trajectory_list.push_back(objs[i].history_path);
        }
    }
}

void visualize_tra()
{
    visualization_msgs::MarkerArray tra_markers;
    for(size_t i = 0; i < trajectory_list.size(); i++)
    {
        if(trajectory_list[i].size() == 0)
            continue;
        
        visualization_msgs::Marker trajectoryPath, trajectoryPt;
        trajectoryPath.header.frame_id = "/map";
        trajectoryPath.header.stamp = ros::Time::now();
        trajectoryPath.ns = "trajectory_path";
        trajectoryPath.action = visualization_msgs::Marker::ADD;
        trajectoryPath.type = visualization_msgs::Marker::LINE_STRIP;
        trajectoryPath.lifetime = ros::Duration(0.1);
        trajectoryPath.pose.orientation.w = 1.0;
        trajectoryPath.id = i;
        trajectoryPath.scale.x = 0.2;
        trajectoryPath.color.r = colorMap_[i].r/255;
        trajectoryPath.color.g = colorMap_[i].g/255;
        trajectoryPath.color.b = colorMap_[i].b/255;
        trajectoryPath.color.a = 1;

        trajectoryPt.header.frame_id = "/map";
        trajectoryPt.header.stamp = ros::Time::now();
        trajectoryPt.ns = "trajectory_pt";
        trajectoryPt.action = visualization_msgs::Marker::ADD;
        trajectoryPt.type = visualization_msgs::Marker::POINTS;
        trajectoryPt.lifetime = ros::Duration(0.1);
        trajectoryPt.id = i;
        trajectoryPt.scale.x = 0.2f;
        trajectoryPt.scale.y = 0.2f;
        trajectoryPt.color.r = 1.0f;
        trajectoryPt.color.a = 1;

        for(const auto& pt : trajectory_list[i])
        {
            trajectoryPath.points.push_back(pt);
            trajectoryPt.points.push_back(pt);
        }

        tra_markers.markers.push_back(trajectoryPath);
        tra_markers.markers.push_back(trajectoryPt);
    }
    pub.publish(tra_markers);
}

void generateColors(const itri_msgs::DetectedObjectArray& input)
{
    // search for the max id
    int max_id = std::numeric_limits<uint32_t>::min();
    for(size_t i = 0; i < input.objects.size(); i++)
    {
        max_id = (input.objects[i].id > max_id ? input.objects[i].id : max_id);
    }
    std::map<int, colors>::iterator it = colorMap_.find(max_id);
    if(it != colorMap_.end())
        return;

    // create colorMap till max_id
    float r_count = 0;
    float g_count = 0;
    float b_count = 0;
    if(current_id == 0)
    {
        r_count = 0;
        g_count = 0;
        b_count = 0;
    }
    else
    {
        r_count = colorMap_[current_id-1].r;
        g_count = colorMap_[current_id-1].g;
        b_count = colorMap_[current_id-1].b;
    }

    for(int i = current_id; i < max_id+1; i++)
    {
        r_count = (r_count > 255 ? 0 : r_count);
        g_count = (g_count > 255 ? 0 : g_count);
        b_count = (b_count > 255 ? 0 : b_count);

        colors c;
        c.r = r_count;
        c.g = g_count;
        c.b = b_count;
        if(i % 3 == 0)
        {
            c.r = static_cast<float>(r_count/255);
            r_count+=5;
        }
        else if(i % 3 == 1)
        {
            c.g = static_cast<float>(g_count/255);
            g_count+=10;
        }
        else
        {
            c.b = static_cast<float>(b_count/255);
            b_count+=15;
        }
        colorMap_[i] = c;
    }
    current_id = max_id+1;
}


void visualize_hull(const std::vector<itri_msgs::DetectedObject>& objs)
{
    for(size_t i = 0; i < objs.size(); i++)
    {
        trackedHull.header.frame_id = "/base_link";
        trackedHull.header.stamp = ros::Time::now();
        trackedHull.ns = "hull";
        trackedHull.action = visualization_msgs::Marker::ADD;
        trackedHull.id = objs[i].id;
        trackedHull.type = visualization_msgs::Marker::LINE_STRIP;
        trackedHull.scale.x = 0.1;
        trackedHull.color.r = 0.0;
        trackedHull.color.g = 0.65;
        trackedHull.color.b = 0.65;
        trackedHull.color.a = 1.0;
        trackedHull.lifetime = ros::Duration(0.1);

        // if(tracker.trajectory.size() < 4)
        // {
        //     return;
        // }

        std::string label = tracker.label;
        // std::cout << label << " " << tracker.score << std::endl;
        if( strlen(label.c_str())==0 || label.compare("lost"))
        {
            // std::cout << strlen(label.c_str()) << "," << label.compare("lost") << std::endl;
            // std::cout << "Get label: " << label << std::endl;
            trackedHull.color.r = label_map[label][0];
            trackedHull.color.g = label_map[label][1];
            trackedHull.color.b = label_map[label][2];
            trackedHull.color.a = tracker.score;
        }

        geometry_msgs::Point markerPt;
        for (size_t j = 0; j < tracker.convex_hull.polygon.points.size(); j++)
        {
            tf::Point pt(
                tracker.convex_hull.polygon.points[j].x,
                tracker.convex_hull.polygon.points[j].y,
                tracker.convex_hull.polygon.points[j].z);
            tf::Point ptTransForm = transform * pt;

            markerPt.x = ptTransForm.x();
            markerPt.y = ptTransForm.y();
            markerPt.z = ptTransForm.z();
            trackedHull.points.push_back(markerPt);
        }
    }
}

void callback(const itri_msgs::DetectedObjectArrayConstPtr& msg)
{
    itri_msgs::DetectedObjectArray array = *msg;
    std::vector<itri_msgs::DetectedObject> objs = array.objects;

    mheader = array.header;

    std::cout << "Get objs " << mheader << std::endl;

    if(array.objects.size() == 0)
        return;
    
    generateColors(array);

    try
    {
        tf_listener_->waitForTransform(
            "/map", "/base_link", mheader.stamp, ros::Duration(1.0));
        tf_listener_->lookupTransform(
            "/map", "/base_link", mheader.stamp, transform);
    }
    catch (tf::TransformException ex)
    {
        ROS_WARN("%s",ex.what());
    }

    createTraList(objs);
    initialize = true;

    visualize_tra();

    visualize_hull(objs);


}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "visTrajectory");
    ros::NodeHandle nh("~");

    sub = nh.subscribe("/detected_objects", 1000, &callback);
    pub = nh.advertise<visualization_msgs::MarkerArray>("trajectory", 1000);

    tf_listener_ = (new tf::TransformListener);
    
    ros::Rate loop(10);
    while (ros::ok())
    {
        ros::spin();
        loop.sleep();
    }
}