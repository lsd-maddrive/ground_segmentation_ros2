#include <memory>
#include <string>
#include <unordered_map>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "ground_segmentation/ground_segmentation.h"

using rclcpp::QoS;
using rclcpp::KeepLast;
using rclcpp::ReliabilityPolicy;
using rclcpp::DurabilityPolicy;
using rclcpp::HistoryPolicy;

namespace {

ReliabilityPolicy parse_reliability(const std::string &s) {
  static const std::unordered_map<std::string, ReliabilityPolicy> map = {
      {"reliable", ReliabilityPolicy::Reliable},
      {"best_effort", ReliabilityPolicy::BestEffort}};
  auto it = map.find(s);
  if (it != map.end())
    return it->second;
  RCLCPP_WARN(rclcpp::get_logger("ground_segmentation"),
              "Unknown reliability '%s', falling back to Reliable", s.c_str());
  return ReliabilityPolicy::Reliable;
}

DurabilityPolicy parse_durability(const std::string &s) {
  static const std::unordered_map<std::string, DurabilityPolicy> map = {
      {"volatile", DurabilityPolicy::Volatile},
      {"transient_local", DurabilityPolicy::TransientLocal}};
  auto it = map.find(s);
  if (it != map.end())
    return it->second;
  RCLCPP_WARN(rclcpp::get_logger("ground_segmentation"),
              "Unknown durability '%s', falling back to Volatile", s.c_str());
  return DurabilityPolicy::Volatile;
}

HistoryPolicy parse_history(const std::string &s) {
  static const std::unordered_map<std::string, HistoryPolicy> map = {
      {"keep_last", HistoryPolicy::KeepLast},
      {"keep_all", HistoryPolicy::KeepAll}};
  auto it = map.find(s);
  if (it != map.end())
    return it->second;
  RCLCPP_WARN(rclcpp::get_logger("ground_segmentation"),
              "Unknown history '%s', falling back to KeepLast", s.c_str());
  return HistoryPolicy::KeepLast;
}

QoS make_qos(const std::string &history_str, int depth,
             const std::string &reliability_str,
             const std::string &durability_str) {
  HistoryPolicy hist = parse_history(history_str);
  ReliabilityPolicy rel = parse_reliability(reliability_str);
  DurabilityPolicy dur = parse_durability(durability_str);

  QoS qos = hist == HistoryPolicy::KeepAll ? QoS(rclcpp::KeepAll()) : QoS(KeepLast(depth));
  qos.reliability(rel).durability(dur);
  return qos;
}

}  // anonymous namespace

class SegmentationNode : public rclcpp::Node {
 public:
  explicit SegmentationNode(const rclcpp::NodeOptions &opts)
      : Node("ground_segmentation", opts) {
    /* ------------------- Parameters ------------------- */
    std::string ground_topic =
        this->declare_parameter("ground_output_topic", "ground_cloud");
    std::string obstacle_topic =
        this->declare_parameter("obstacle_output_topic", "obstacle_cloud");
    std::string input_topic =
        this->declare_parameter("input_topic", "input_cloud");

    /* QoS parameters */
    // Subscriber QoS
    std::string sub_history =
        this->declare_parameter("sub_qos_history", "keep_last");
    int sub_depth = this->declare_parameter("sub_qos_depth", 10);
    std::string sub_reliability =
        this->declare_parameter("sub_qos_reliability", "reliable");
    std::string sub_durability =
        this->declare_parameter("sub_qos_durability", "volatile");

    // Publisher QoS
    std::string pub_history =
        this->declare_parameter("pub_qos_history", "keep_last");
    int pub_depth = this->declare_parameter("pub_qos_depth", 10);
    std::string pub_reliability =
        this->declare_parameter("pub_qos_reliability", "reliable");
    std::string pub_durability =
        this->declare_parameter("pub_qos_durability", "volatile");

    QoS sub_qos = make_qos(sub_history, sub_depth, sub_reliability, sub_durability);
    QoS pub_qos = make_qos(pub_history, pub_depth, pub_reliability, pub_durability);

    /* Algorithm parameters */
    gravity_aligned_frame_ =
        this->declare_parameter("gravity_aligned_frame", "gravity_aligned");

    params_.visualize       = this->declare_parameter("visualize", params_.visualize);
    params_.n_bins          = this->declare_parameter("n_bins", params_.n_bins);
    params_.n_segments      = this->declare_parameter("n_segments", params_.n_segments);
    params_.max_dist_to_line = this->declare_parameter("max_dist_to_line", params_.max_dist_to_line);
    params_.max_slope       = this->declare_parameter("max_slope", params_.max_slope);
    params_.min_slope       = this->declare_parameter("min_slope", params_.min_slope);
    params_.long_threshold  = this->declare_parameter("long_threshold", params_.long_threshold);
    params_.max_long_height = this->declare_parameter("max_long_height", params_.max_long_height);
    params_.sensor_height   = this->declare_parameter("sensor_height", params_.sensor_height);
    params_.line_search_angle = this->declare_parameter("line_search_angle", params_.line_search_angle);
    params_.n_threads       = this->declare_parameter("n_threads", params_.n_threads);

    /* Publishers / Subscribers */
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        input_topic, sub_qos,
        std::bind(&SegmentationNode::scanCallback, this, std::placeholders::_1));

    ground_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        ground_topic, pub_qos);
    obstacle_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
        obstacle_topic, pub_qos);

    tf_buffer_   = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    segmenter_ = std::make_shared<GroundSegmentation>(params_);

    RCLCPP_INFO(this->get_logger(),
                "Ground segmentation node initialized with custom QoS.");
  }

 private:
  void scanCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(*msg, cloud);

    bool is_original_pc = true;
    pcl::PointCloud<pcl::PointXYZ> cloud_transformed;

    if (!gravity_aligned_frame_.empty()) {
      try {
        geometry_msgs::msg::TransformStamped transform =
            tf_buffer_->lookupTransform(gravity_aligned_frame_, msg->header.frame_id, rclcpp::Time(0));
        Eigen::Matrix4f eigen_transform =
            pcl::getTransformation(transform.transform.translation.x,
                                   transform.transform.translation.y,
                                   transform.transform.translation.z,
                                   0, 0, 0).matrix().cast<float>();
        pcl::transformPointCloud(cloud, cloud_transformed, eigen_transform);
        is_original_pc = false;
      } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
      }
    }

    const pcl::PointCloud<pcl::PointXYZ> &cloud_proc = is_original_pc ? cloud : cloud_transformed;
    std::vector<int> labels;
    segmenter_->segment(cloud_proc, &labels);

    pcl::PointCloud<pcl::PointXYZ> ground_cloud, obstacle_cloud;
    ground_cloud.reserve(cloud_proc.size());
    obstacle_cloud.reserve(cloud_proc.size());

    for (size_t i = 0; i < cloud_proc.size(); ++i) {
      if (labels[i] == 1)
        ground_cloud.push_back(cloud_proc[i]);
      else
        obstacle_cloud.push_back(cloud_proc[i]);
    }

    sensor_msgs::msg::PointCloud2 ground_msg;
    sensor_msgs::msg::PointCloud2 obstacle_msg;
    pcl::toROSMsg(ground_cloud, ground_msg);
    pcl::toROSMsg(obstacle_cloud, obstacle_msg);
    ground_msg.header   = msg->header;
    obstacle_msg.header = msg->header;

    ground_pub_->publish(ground_msg);
    obstacle_pub_->publish(obstacle_msg);
  }

  /* Members */
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacle_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;

  GroundSegmentationParams params_;
  std::shared_ptr<GroundSegmentation> segmenter_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::string gravity_aligned_frame_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<SegmentationNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}