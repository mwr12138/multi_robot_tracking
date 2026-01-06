#include <std_msgs/Bool.h>
#include <Eigen/Geometry>
#include <unordered_set>
//cv image
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//ros
#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>

//phd filter class
#include <multi_robot_tracking/PhdFilter.h>

//jpdaf filter class
#include <multi_robot_tracking/JpdafFilter.h>

//Simple Kalman filter class
#include <multi_robot_tracking/SimpleKalman.h>

//export and store csv
#include <iostream>
#include <fstream>

#include <chrono>

#define HOST

using namespace std;

bool want_export_toCSV = false;


class multi_robot_tracking_Nodelet : public nodelet::Nodelet
{
public:
    multi_robot_tracking_Nodelet() {}
    ~multi_robot_tracking_Nodelet();

    void onInit(); //default init of nodelet

    

    //callback functions
    void detection_Callback(const geometry_msgs::PoseArray& in_PoseArray); //bbox to track
#ifdef HOST    
    void image_Callback(const sensor_msgs::ImageConstPtr &img_msg); //rgb raw
#endif
    void imu_Callback(const sensor_msgs::ImuConstPtr &imu_msg); //rgb raw

    Eigen::MatrixXf get_B_ang_vel_matrix(float x, float y); //return B matrix for each measurement


    //extra helper functions
    void draw_image();
    void init_matrices();
    void publish_tracks();
    void associate_consensus();
    Eigen::MatrixXf project_2d_to_3d(Eigen::MatrixXf position);
    void consensus_sort();

    std::string filter_to_use_; //choose between phd or jpdaf
    std::string input_bbox_topic; //choose between /hummingbird1/track/bounding_box or /image_processor/objects_center
    std::string input_img_topic; //choose between /hummingbird1/track/bounding_box or /image_processor/objects_center
    std::string input_imu_topic;
    int num_drones;

    float init_pos_x_left =0; //init position to read from rosparam for consensus
    float init_pos_y_left =0;
    float init_pos_x_right = 0;
    float init_pos_y_right = 0;
    float init_pos_self_x = 0;
    float init_pos_self_y = 0;
    int id_left = 0;
    int id_right = 0;


    bool consensus_sort_complete = true;

    ros::Time img_timestamp; //当前图像消息的时间戳
    ros::Time prev_img_timestamp; //上一帧图像的时间戳
    ros::Time bbox_timestamp; //目标检测结果（bounding box）的时间戳
    ros::Time imu_timestamp; //IMU消息的时间戳
    double previous_timestamp = 0; //t-1  上一次检测回调的时间（用于计算时间间隔）
    double current_timestamp = 0;  //t    当前检测回调的时间
    double delta_timestamp = 0;    //dt   当前与上一次检测之间的时间差（`current_timestamp - previous_timestamp`）
    double imu_time = 0;
    int detection_seq = 0;

    bool first_track_flag = false; //flag after 1st track update, use to update asynchronous prediction

    std::vector<sensor_msgs::ImageConstPtr> image_buffer_; //存储未处理的图像消息（`sensor_msgs::ImageConstPtr`），用于与检测结果的时间同步
    std::vector<sensor_msgs::Imu> sensor_imu_buffer_;  //存储IMU消息（代码中声明但未使用）
    std::ofstream tracking_csv_; //////
    int frame_count_ = 0; ///////

    //filters initialize
    PhdFilter phd_filter_;
    JpdafFilter jpdaf_filter_;
    KalmanFilter kalman_filter_;

    //sub and pub
    image_transport::Publisher image_pub_;
    ros::Publisher pose_glass2drone_pub_;
    ros::Publisher pose_glass2drone_proj_pub_;
    ros::Publisher tracked_pose_pub_;
    ros::Publisher tracked_velocity_pub_;

    ros::Subscriber detection_sub_;
    ros::Subscriber image_sub_;
    ros::Subscriber imu_sub_;
    ros::Subscriber groundtruth_sub_;

    //output RGB data, pose data
#ifdef HOST
    cv::Mat input_image;
    cv::Mat previous_image;
    sensor_msgs::ImagePtr image_msg;
#endif
    sensor_msgs::Imu imu_;


    //3D matrices for transform
    Eigen::MatrixXf rotm_world2cam;
    Eigen::MatrixXf Hmatfiller1x4;
    Eigen::MatrixXf k_matrix3x3;
    Eigen::MatrixXf k_matrix3x3_inv;

    //init world coordinates for consensus
    Eigen::MatrixXf positions_world_coordinate;
    Eigen::MatrixXf positions_cam_coordinate;
    Eigen::MatrixXf projected_2d_initial_coord;

    Eigen::MatrixXi id_consensus;
    Eigen::MatrixXi id_array_init;


    //B matrix constants for ang velocity
    float cx, cy, f;

    float filter_dt;

    bool enable_async_pdf;

    //Detection image frame 
    int detection_height, detection_width;
    int detection_offset_x, detection_offset_y;

    //Camera frame size
    int image_height, image_width;

    //phd_filter_parameters
    float q_pos, q_vel, r_meas;
    float p_pos_init, p_vel_init;
    float phd_prune_weight_threshold;
    float phd_prune_mahalanobis_dist_threshold;
    float phd_extract_weight_threshold;

    //output csv file
    ofstream outputFile;
private:
    // ID关联相关成员变量
    std::vector<int> previous_ids_;              // 上一帧的目标ID
    std::vector<int> current_ids;              // 当前帧的目标ID
    std::vector<Eigen::Vector2f> previous_positions_; // 上一帧的目标位置
    int next_id_=0;                                // 下一个可用的ID
    float max_association_distance_=50.0f;             // 关联的最大距离阈值
    float boundary_threshold_ = 20.0f; //边界阈值
    void associate_ids();
    // 新增：存储有遮挡关系的ID组（每组为一个vector<int>）
    std::vector<std::vector<int>> occlusion_id_groups; 
    // 新增：记录当前使用的ID，用于限制范围
    std::unordered_set<int> used_ids; 


    // ID状态矩阵：4 x NUM_DRONES
    // 第0行：ID编号 (0, 1, 2, ..., NUM_DRONES-1)
    // 第1行：状态 (0=正常, 1=离开视野)
    // 第2行：离开时的x坐标
    // 第3行：离开时的y坐标
    Eigen::MatrixXf id_status_matrix_;
    
    // 参数
    float return_distance_threshold_ = 80.0f;  // 回归距离阈值(像素)
    int min_visible_frames_ = 5;               // 最小可见帧数才记录离开位置
    
    // 每个ID的可见帧数计数器
    std::vector<int> id_visible_frames_;
    
    // 新增函数：判断是被遮挡还是离开视野
    int determine_occlusion_or_leave(int target_id, int& occluder_id);
    int find_returning_target(float x, float y);
    void update_id_status_matrix();
    int find_available_id(float x, float y);
    void process_new_detections();
    void update_id_consensus_from_status();
};

multi_robot_tracking_Nodelet::~multi_robot_tracking_Nodelet()
{
    if (tracking_csv_.is_open()) {
        tracking_csv_.close();
        ROS_INFO("Tracking CSV file closed");
    }
    
    // 如果有其他资源需要释放，也放在这里
    if (outputFile.is_open()) {  // 原CSV文件
        outputFile.close();
        ROS_INFO("Ground truth CSV file closed");
    }
}


int multi_robot_tracking_Nodelet::determine_occlusion_or_leave(int target_id, int& occluder_id) {
    float weight = phd_filter_.wk_bar_display(target_id);
    float current_x = phd_filter_.X_k(0, target_id);
    float current_y = phd_filter_.X_k(2, target_id);
    
    // 情况1：目标在图像边界附近，且权重很低 -> 可能是离开视野
    bool near_boundary = (current_x < boundary_threshold_ || 
                         current_x > detection_width - boundary_threshold_ ||
                         current_y < boundary_threshold_ || 
                         current_y > detection_height - boundary_threshold_);
    
    if (near_boundary && weight < 0.2f) {
        ROS_DEBUG("目标 %d 在边界附近且权重低，判断为离开视野", target_id);
        return 1; // 离开视野
    }
    
    // 情况2：目标在图像中央区域，但权重突然降低 -> 可能是被遮挡
    bool in_central_area = (current_x > boundary_threshold_ * 2 && 
                           current_x < detection_width - boundary_threshold_ * 2 &&
                           current_y > boundary_threshold_ * 2 && 
                           current_y < detection_height - boundary_threshold_ * 2);
    
    if (weight < 0.3f && in_central_area) {
        // 进一步检查是否有其他目标在附近（检测框重叠）
        float target_x = phd_filter_.Detections(0, target_id);
        float target_y = phd_filter_.Detections(1, target_id);
        float target_w = phd_filter_.Detections(2, target_id);
        float target_h = phd_filter_.Detections(3, target_id);
        
        float left1 = target_x - target_w/2;
        float right1 = target_x + target_w/2;
        float top1 = target_y - target_h/2;
        float bottom1 = target_y + target_h/2;
        
        float max_overlap = 0.0f;
        int best_occluder = -1;
        
        for (int i = 0; i < num_drones; i++) {
            if (i == target_id) continue;
            
            float other_x = phd_filter_.Detections(0, i);
            float other_y = phd_filter_.Detections(1, i);
            float other_w = phd_filter_.Detections(2, i);
            float other_h = phd_filter_.Detections(3, i);
            
            float left2 = other_x - other_w/2;
            float right2 = other_x + other_w/2;
            float top2 = other_y - other_h/2;
            float bottom2 = other_y + other_h/2;
            
            // 计算重叠区域
            float overlap_x = std::max(0.0f, std::min(right1, right2) - std::max(left1, left2));
            float overlap_y = std::max(0.0f, std::min(bottom1, bottom2) - std::max(top1, top2));
            float overlap_area = overlap_x * overlap_y;
            float target_area = target_w * target_h;
            
            // 计算重叠比例
            float overlap_ratio = overlap_area / target_area;
            
            if (overlap_ratio > max_overlap) {
                max_overlap = overlap_ratio;
                best_occluder = i;
            }
        }
        
        // 如果最大重叠比例超过阈值，认为是被遮挡
        if (max_overlap > 0.3f) {
            occluder_id = best_occluder;
            ROS_DEBUG("目标 %d 与目标 %d 重叠比例 %.2f，判断为被遮挡", 
                     target_id, best_occluder, max_overlap);
            return 2; // 被遮挡
        }
    }
    
    // 情况3：权重逐渐降低，且向边界移动 -> 离开视野
    // 这里需要检查历史位置来判断移动趋势
    // 暂时简化处理：如果不在中央区域且权重低，认为是离开视野
    
    ROS_DEBUG("目标 %d 默认判断为离开视野", target_id);
    return 1; // 默认认为是离开视野
}


// void multi_robot_tracking_Nodelet::update_id_status_matrix() {   //更新ID状态矩阵的核心逻辑
//     // 确保矩阵已初始化
//     if (id_status_matrix_.cols() == 0) return;
    
//     for (int i = 0; i < num_drones; i++) {
//         float weight = phd_filter_.wk_bar_display(i);
//         float current_x = phd_filter_.X_k(0, i);
//         float current_y = phd_filter_.X_k(2, i);
        
//         if (weight > 0.3f) {  // 目标可见
//             id_visible_frames_[i]++;
            
//             // 如果目标之前是离开状态，检查是否是回归
//             if (id_status_matrix_(1, i) == 1) {  // 之前是离开状态
//                 float leave_x = id_status_matrix_(2, i);
//                 float leave_y = id_status_matrix_(3, i);
                
//                 // 计算与离开位置的距离
//                 float distance = std::sqrt(std::pow(current_x - leave_x, 2) + 
//                                          std::pow(current_y - leave_y, 2));
                
//                 if (distance < return_distance_threshold_) {
//                     // 目标回归！保持原有ID
//                     id_status_matrix_(1, i) = 0;  // 状态恢复正常
//                     id_status_matrix_(2, i) = -1; // 清除离开坐标
//                     id_status_matrix_(3, i) = -1;
//                     ROS_INFO("目标 %d 从离开位置回归，距离: %.1f", i, distance);
//                 } else {
//                     // 可能是新目标，在远处出现，暂时不处理
//                 }
//             } else {
//                 // 目标正常可见，确保状态为正常
//                 id_status_matrix_(1, i) = 0;
//             }
//         } else {  // 目标不可见
//             // 如果目标之前可见且达到最小可见帧数，记录离开位置
//             if (id_status_matrix_(1, i) == 0 && id_visible_frames_[i] >= min_visible_frames_) {
//                 id_status_matrix_(1, i) = 1;  // 标记为离开状态
//                 id_status_matrix_(2, i) = phd_filter_.X_k(0, i);  // 记录离开x坐标
//                 id_status_matrix_(3, i) = phd_filter_.X_k(2, i);  // 记录离开y坐标
//                 id_visible_frames_[i] = 0;  // 重置可见帧数
//                 ROS_INFO("目标 %d 离开视野，位置: (%.1f, %.1f)", 
//                         i, id_status_matrix_(2, i), id_status_matrix_(3, i));
//             } else if (id_status_matrix_(1, i) == 0) {
//                 // 可见帧数不足，不记录离开位置，但重置计数器
//                 id_visible_frames_[i] = 0;
//             }
//         }
//     }
    
// }


void multi_robot_tracking_Nodelet::update_id_status_matrix() {
    // 确保矩阵已初始化
    if (id_status_matrix_.cols() == 0) {
        id_status_matrix_ = Eigen::MatrixXf::Zero(5, num_drones);
        id_visible_frames_.resize(num_drones, 0);
        
        for (int i = 0; i < num_drones; i++) {
            id_status_matrix_(0, i) = i;  // ID编号
            id_status_matrix_(1, i) = 0;  // 初始状态为正常
            id_status_matrix_(2, i) = -1; // 离开/遮挡x坐标
            id_status_matrix_(3, i) = -1; // 离开/遮挡y坐标
            id_status_matrix_(4, i) = -1; // 遮挡者ID
        }
        return;
    }
    
    // 确保不越界
    int min_cols = std::min(num_drones, static_cast<int>(phd_filter_.X_k.cols()));
    min_cols = std::min(min_cols, static_cast<int>(phd_filter_.wk_bar_display.cols()));
    
    for (int i = 0; i < min_cols; i++) {
        float weight = phd_filter_.wk_bar_display(i);
        float current_x = phd_filter_.X_k(0, i);
        float current_y = phd_filter_.X_k(2, i);
        
        if (weight > 0.3f) {  // 目标可见
            id_visible_frames_[i]++;
            
            // 检查是否从离开/遮挡状态回归
            int current_state = id_status_matrix_(1, i);
            
            if (current_state == 1) {  // 之前是离开状态
                float leave_x = id_status_matrix_(2, i);
                float leave_y = id_status_matrix_(3, i);
                
                if (leave_x >= 0 && leave_y >= 0) {
                    float distance = std::sqrt(std::pow(current_x - leave_x, 2) + 
                                             std::pow(current_y - leave_y, 2));
                    
                    if (distance < return_distance_threshold_) {
                        id_status_matrix_(1, i) = 0;  // 恢复为正常
                        id_status_matrix_(2, i) = -1;
                        id_status_matrix_(3, i) = -1;
                        ROS_INFO("目标 %d 从离开位置回归，距离: %.1f", i, distance);
                    }
                }
            } 
            else if (current_state == 2) {  // 之前是被遮挡状态
                // 检查遮挡是否解除（与遮挡者的重叠是否消失）
                int occluder_id = id_status_matrix_(4, i);
                if (occluder_id >= 0 && occluder_id < num_drones) {
                    // 检查当前是否还与遮挡者有显著重叠
                    float target_x = phd_filter_.Detections(0, i);
                    float target_y = phd_filter_.Detections(1, i);
                    float target_w = phd_filter_.Detections(2, i);
                    float target_h = phd_filter_.Detections(3, i);
                    
                    float occluder_x = phd_filter_.Detections(0, occluder_id);
                    float occluder_y = phd_filter_.Detections(1, occluder_id);
                    float occluder_w = phd_filter_.Detections(2, occluder_id);
                    float occluder_h = phd_filter_.Detections(3, occluder_id);
                    
                    // 计算当前重叠
                    float left1 = target_x - target_w/2;
                    float right1 = target_x + target_w/2;
                    float top1 = target_y - target_h/2;
                    float bottom1 = target_y + target_h/2;
                    
                    float left2 = occluder_x - occluder_w/2;
                    float right2 = occluder_x + occluder_w/2;
                    float top2 = occluder_y - occluder_h/2;
                    float bottom2 = occluder_y + occluder_h/2;
                    
                    float overlap_x = std::max(0.0f, std::min(right1, right2) - std::max(left1, left2));
                    float overlap_y = std::max(0.0f, std::min(bottom1, bottom2) - std::max(top1, top2));
                    float overlap_area = overlap_x * overlap_y;
                    float target_area = target_w * target_h;
                    float overlap_ratio = overlap_area / target_area;
                    
                    if (overlap_ratio < 0.2f) {  // 重叠很小，遮挡解除
                        id_status_matrix_(1, i) = 0;  // 恢复为正常
                        id_status_matrix_(4, i) = -1; // 清除遮挡者ID
                        ROS_INFO("目标 %d 遮挡解除", i);
                    }
                } else {
                    // 遮挡者ID无效，直接恢复
                    id_status_matrix_(1, i) = 0;
                    id_status_matrix_(4, i) = -1;
                }
            }
        } 
        else {  // 目标不可见或权重低
            int occluder_id = -1;
            int new_state = determine_occlusion_or_leave(i, occluder_id);
                
            if (new_state != 0 && id_status_matrix_(1, i) == 0 && 
                id_visible_frames_[i] >= min_visible_frames_) {
                
                id_status_matrix_(1, i) = new_state;  // 更新状态
                id_status_matrix_(2, i) = current_x;  // 记录位置
                id_status_matrix_(3, i) = current_y;
                
                if (new_state == 2) {  // 被遮挡
                    id_status_matrix_(4, i) = occluder_id;  // 记录遮挡者
                    ROS_INFO("目标 %d 被目标 %d 遮挡", i, occluder_id);
                } else {  // 离开视野
                    id_status_matrix_(4, i) = -1;  // 清除遮挡者ID
                    ROS_INFO("目标 %d 离开视野，位置: (%.1f, %.1f)", i, current_x, current_y);
                }
                
                id_visible_frames_[i] = 0;
            } 
            else if (id_status_matrix_(1, i) == 0) {
                id_visible_frames_[i] = 0;
            }
        }
    }
}


int multi_robot_tracking_Nodelet::find_available_id(float x, float y) {  //为新目标寻找可用ID
    // 第一步：寻找完全空闲的ID（状态正常但当前不可见）
    for (int i = 0; i < num_drones; i++) {
        if (id_status_matrix_(1, i) == 0 && phd_filter_.wk_bar_display(i) < 0.1f) {
            return i;
        }
    }
    
    // 第二步：寻找离开状态但位置较远的ID
    for (int i = 0; i < num_drones; i++) {
        if (id_status_matrix_(1, i) == 1) {  // 离开状态
            float leave_x = id_status_matrix_(2, i);
            float leave_y = id_status_matrix_(3, i);
            
            // 计算与离开位置的距离
            float distance = std::sqrt(std::pow(x - leave_x, 2) + 
                                     std::pow(y - leave_y, 2));
            
            // 如果距离足够远，可以复用这个ID
            if (distance > return_distance_threshold_ * 2) {
                ROS_WARN("复用离开状态的ID %d (新位置距离离开位置 %.1f)", i, distance);
                id_status_matrix_(1, i) = 0;  // 状态恢复正常
                id_status_matrix_(2, i) = -1; // 清除离开坐标
                id_status_matrix_(3, i) = -1;
                return i;
            }
        }
    }
    
    // 第三步：如果没有可用ID，返回-1（理论上不应该发生）
    ROS_ERROR("没有可用的ID！");
    return -1;
}


void multi_robot_tracking_Nodelet::associate_ids()
{
    // 获取当前帧的目标位置 (x, y)
    int current_num_targets = phd_filter_.X_k.cols();
    Eigen::MatrixXf current_positions(2, current_num_targets);
    for (int i = 0; i < current_num_targets; i++) {
        current_positions(0, i) = phd_filter_.X_k(0, i); // x坐标
        current_positions(1, i) = phd_filter_.X_k(2, i); // y坐标
    }

    // 初始化当前帧ID向量
    current_ids.resize(current_num_targets, -1);
    
    // 遮挡检测相关变量（局部变量，在函数内初始化）
    std::vector<bool> occlusion_flags(current_num_targets, false);
    std::vector<std::pair<int, int>> occlusion_pairs;
    
    // 清空遮挡ID组和已使用ID集合
    occlusion_id_groups.clear();
    used_ids.clear();

    // ==== 1. 修正遮挡检测（基于中心坐标计算目标框边界）====
    for (int i = 0; i < current_num_targets; ++i) {
        // 目标i的边界框：中心(x1,y1)，宽w1，高h1 → 左上角(x1-w1/2, y1-h1/2)，右下角(x1+w1/2, y1+h1/2)
        float x1 = phd_filter_.Detections(0, i);
        float y1 = phd_filter_.Detections(1, i);
        float w1 = phd_filter_.Detections(2, i);
        float h1 = phd_filter_.Detections(3, i);
        float left1 = x1 - w1 / 2.0f;
        float right1 = x1 + w1 / 2.0f;
        float top1 = y1 - h1 / 2.0f;
        float bottom1 = y1 + h1 / 2.0f;

        for (int j = i + 1; j < current_num_targets; ++j) {
            // 目标j的边界框
            float x2 = phd_filter_.Detections(0, j);
            float y2 = phd_filter_.Detections(1, j);
            float w2 = phd_filter_.Detections(2, j);
            float h2 = phd_filter_.Detections(3, j);
            float left2 = x2 - w2 / 2.0f;
            float right2 = x2 + w2 / 2.0f;
            float top2 = y2 - h2 / 2.0f;
            float bottom2 = y2 + h2 / 2.0f;

            // 轴对齐矩形交集判断
            bool collision = (left1 < right2) && (right1 > left2) &&
                             (top1 < bottom2) && (bottom1 > top2);

            if (collision) {
                occlusion_flags[i] = true;
                occlusion_flags[j] = true;
                occlusion_pairs.emplace_back(i, j);
            }
        }
    }

    // 打印遮挡信息（调试用）
    ROS_INFO("Occlusion Flags: ");
    for (int i = 0; i < current_num_targets; ++i) {
        ROS_INFO("Target %d: %s", i, occlusion_flags[i] ? "true" : "false");
    }
    
    ROS_INFO("Occlusion Pairs: ");
    for (const auto& pair : occlusion_pairs) {
        ROS_INFO("(%d, %d)", pair.first, pair.second);
    }

    // ==== 2. ID关联（优先处理非遮挡目标，再处理遮挡目标）====
    // 2.1 处理非遮挡目标：基于最近邻关联
    for (int i = 0; i < current_num_targets; ++i) {
        if (occlusion_flags[i]) continue; // 遮挡目标稍后处理

        float min_dist = std::numeric_limits<float>::max();
        int matched_idx = -1;
        for (size_t j = 0; j < previous_positions_.size(); ++j) {
            float dist = (current_positions.col(i) - previous_positions_[j]).norm();
            if (dist < min_dist && dist < max_association_distance_) {
                min_dist = dist;
                matched_idx = j;
            }
        }

        if (matched_idx != -1) {
            current_ids[i] = previous_ids_[matched_idx]; // 关联到上一帧ID
        } else {
            // 分配新ID：从0~num_drones-1中选一个未使用的
            for (int id = 0; id < num_drones; ++id) {
                if (used_ids.find(id) == used_ids.end()) {
                    current_ids[i] = id;
                    used_ids.insert(id);
                    break;
                }
            }
        }
        used_ids.insert(current_ids[i]); // 标记为已使用
    }

    // 2.2 处理遮挡目标：基于上一帧ID和位置预测关联
    for (const auto& pair : occlusion_pairs) {
        int idx1 = pair.first;
        int idx2 = pair.second;

        // 情况1：其中一个已关联，另一个尝试关联到最可能的ID
        if (current_ids[idx1] != -1 && current_ids[idx2] == -1) {
            // 从可用ID中为idx2分配新ID
            for (int id = 0; id < num_drones; ++id) {
                if (used_ids.find(id) == used_ids.end()) {
                    current_ids[idx2] = id;
                    used_ids.insert(id);
                    break;
                }
            }
        } else if (current_ids[idx2] != -1 && current_ids[idx1] == -1) {
            // 同理为idx1分配新ID
            for (int id = 0; id < num_drones; ++id) {
                if (used_ids.find(id) == used_ids.end()) {
                    current_ids[idx1] = id;
                    used_ids.insert(id);
                    break;
                }
            }
        } 
        // 情况2：两者都未关联，分配未使用的ID
        else if (current_ids[idx1] == -1 && current_ids[idx2] == -1) {
            // 为idx1分配ID
            for (int id = 0; id < num_drones; ++id) {
                if (used_ids.find(id) == used_ids.end()) {
                    current_ids[idx1] = id;
                    used_ids.insert(id);
                    break;
                }
            }
            // 为idx2分配不同的ID
            for (int id = 0; id < num_drones; ++id) {
                if (used_ids.find(id) == used_ids.end()) {
                    current_ids[idx2] = id;
                    used_ids.insert(id);
                    break;
                }
            }
        }
    }

    // ==== 3. 生成遮挡ID组（将索引对转换为ID对）====
    for (const auto& pair : occlusion_pairs) {
        int id1 = current_ids[pair.first];
        int id2 = current_ids[pair.second];
        occlusion_id_groups.push_back({id1, id2}); // 存储有遮挡关系的ID
    }

    // 调试输出：打印遮挡ID组
    ROS_INFO("Occlusion ID Groups:");
    for (const auto& group : occlusion_id_groups) {
        ROS_INFO("IDs: %d, %d", group[0], group[1]);
    }

    // ==== 4. 更新共识矩阵和历史状态====
    id_consensus.resize(1, current_ids.size());
    for (size_t i = 0; i < current_ids.size(); ++i) {
        id_consensus(i) = current_ids[i];
    }

    previous_ids_ = current_ids;
    previous_positions_.clear();
    for (int i = 0; i < current_num_targets; ++i) {
        previous_positions_.push_back(current_positions.col(i));
    }
}


void multi_robot_tracking_Nodelet::init_matrices()
{
    ROS_INFO("init matrix for drone num: %d",num_drones);
    ROS_WARN("nodelet start init matrix... verify cam K matrix for simulation or snapdragon pro!");
    k_matrix3x3 = Eigen::MatrixXf(3,3);
    k_matrix3x3_inv = Eigen::MatrixXf(3,3);

    positions_world_coordinate =  Eigen::MatrixXf(3,num_drones);
    positions_cam_coordinate =  Eigen::MatrixXf(3,num_drones);
    projected_2d_initial_coord = Eigen::MatrixXf(2,num_drones);


    id_consensus = Eigen::MatrixXi(1,num_drones);
    id_array_init = Eigen::MatrixXi(1,num_drones);
    for(int i=0; i<num_drones; i++)
    {
        id_consensus(i) = i;
        id_array_init(i) = i;
    }
    // id_consensus << id_left, id_right ;

    

    // id_array_init(0) = id_left;
    // id_array_init(1) = id_right;


    rotm_world2cam = Eigen::MatrixXf(3,3);

    // cx = 329; //from HW
    // cy = 243;
    // f = 431;

    k_matrix3x3(0,0) = f; k_matrix3x3(0,1) = 0;   k_matrix3x3(0,2) = cx;
    k_matrix3x3(1,0) = 0;   k_matrix3x3(1,1) = f; k_matrix3x3(1,2) = cy;
    k_matrix3x3(2,0) = 0;   k_matrix3x3(2,1) = 0;   k_matrix3x3(2,2) = 1;

    k_matrix3x3_inv = k_matrix3x3.inverse();

    /* [0 -1  0]
   * [0  0 -1]
   * [1  0  0]
   */
    rotm_world2cam(0,0) = 0;   rotm_world2cam(0,1) =-1;   rotm_world2cam(0,2) =  0;
    rotm_world2cam(1,0) = 0;   rotm_world2cam(1,1) = 0;   rotm_world2cam(1,2) = -1;
    rotm_world2cam(2,0) = 1;   rotm_world2cam(2,1) = 0;   rotm_world2cam(2,2) =  0;


    // 初始化ID状态矩阵
    id_status_matrix_ = Eigen::MatrixXf::Zero(5, num_drones);
    id_visible_frames_.resize(num_drones, 0);
    
    // 设置ID编号
    for (int i = 0; i < num_drones; i++) {
        id_status_matrix_(0, i) = i;  // ID编号
        id_status_matrix_(1, i) = 0;  // 初始状态为正常
        id_status_matrix_(2, i) = -1; // 离开x坐标，-1表示无效
        id_status_matrix_(3, i) = -1; // 离开y坐标，-1表示无效
        id_status_matrix_(4, i) = -1; // 被遮挡
    }
}

// /* use tracking data to draw onto 2D image
//  * input: N/A
//  * output: 2D image with tracking ID
//  */
// void multi_robot_tracking_Nodelet::draw_image()
// {
// #ifdef HOST
//     if(filter_to_use_.compare("jpdaf") == 0)
//     {
//         //          ROS_INFO("drawing jpdaf estimation");
//         float scaleX = input_image.cols / (float)detection_width;
//         float scaleY = input_image.rows / (float)detection_height;
//         for(int k=0; k < jpdaf_filter_.tracks_.size(); k++)
//         {
//             Eigen::Vector2f temp_center;
//             temp_center = jpdaf_filter_.tracks_[k].get_z();
//             int scaledX = floor((temp_center[0] + detection_offset_x) * scaleX);
//             int scaledY = floor((temp_center[1] + detection_offset_y) * scaleY);
//             temp_center[0] = scaledX;
//             temp_center[1] = scaledY;
//             cv::Point2f target_center(temp_center(0), temp_center(1));
//             cv::Point2f id_pos(temp_center(0),temp_center(1)+30);
//             cv::circle(input_image,target_center,4, cv::Scalar(0, 210, 255), 2);
//             putText(input_image, to_string(k), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 2.0, cvScalar(0, 0, 255), 2, cv::LINE_AA);//size 1.5 --> 0.5

//             //draw cross
//             cv::Point2f det_cross_a(temp_center(0)-5, temp_center(1)-5);
//             cv::Point2f det_cross_b(temp_center(0)+5, temp_center(1)-5);
//             cv::Point2f det_cross_c(temp_center(0)-5, temp_center(1)+5);
//             cv::Point2f det_cross_d(temp_center(0)+5, temp_center(1)+5);
//             line(input_image, det_cross_a, det_cross_d, cv::Scalar(255, 20, 150), 1, 1 );
//             line(input_image, det_cross_b, det_cross_c, cv::Scalar(255, 20, 150), 1, 1 );
//         }


//     }

//     else if(filter_to_use_.compare("phd") == 0) {

//         //scale 224x224 to 640x480

//         float scaleX = input_image.cols / (float)detection_width;
//         float scaleY = input_image.rows / (float)detection_height;
// //        ROS_INFO("drawing phd estimation");
        
// //measured input
//         for (int k=0; k < phd_filter_.Detections.cols(); k++)
//         {

//             int scaledX = floor((phd_filter_.Detections(0,k) + detection_offset_x) * scaleX);
//             int scaledY = floor((phd_filter_.Detections(1,k) + detection_offset_y) * scaleY);
//             float scaledW = phd_filter_.Detections(2,k) * scaleX;
//             float scaledH = phd_filter_.Detections(3,k) * scaleY;


//             cv::Point2f measured_center(scaledX, scaledY);
//             //cv::Point2f id_pos(phd_filter_.Z_k(0,k),phd_filter_.Z_k(1,k)+10);
//             //cv::circle(input_image,measured_center,4, cv::Scalar(255, 0, 0), 1);
//             //              putText(previous_image, to_string(k), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(0, 255, 0), 2, cv::LINE_AA);//size 1.5 --> 0.5
//             cv::Point2f top_left(scaledX - scaledW/2, scaledY - scaledH/2);
//             cv::Point2f bottom_right(scaledX + scaledW/2, scaledY + scaledH/2);
//             cv::rectangle(input_image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);

//         }
// // ROS_ERROR_STREAM("画图X_k is:\n" << phd_filter_.X_k << "\n");
//         for(int k=0; k < phd_filter_.X_k.cols(); k++)
//         {
//             int scaledX = floor((phd_filter_.X_k(0,k) + detection_offset_x) * scaleX);
//             int scaledY = floor((phd_filter_.X_k(2,k) + detection_offset_y) * scaleY);
//             if(scaledX > 0 && scaledX < input_image.cols && scaledY > 0 && scaledY < input_image.rows)
//             {
//             cout << "object[" << k << "] - "
//              << "(" << scaledX << ", " << scaledY << ")"
//              << " ID: " << id_consensus(k) << endl;
//             cv::Point2f target_center(scaledX,scaledY);
//             cv::Point2f id_pos(scaledX,scaledY+20);
//             cv::circle(input_image,target_center,2, cv::Scalar(0, 210, 255), 1);
//             putText(input_image, to_string(int(id_consensus(k))), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.4, cvScalar(0, 0, 255), 2.5, cv::LINE_AA);//size 1.5 --> 0.5
//         cout<<""<<endl;
//             }
//         }

        
//         //cout<<"draw down"<<endl;
//     }

//     else if(filter_to_use_.compare("kalman") == 0) {

//         //scale 224x224 to 640x480

//         float scaleX = input_image.cols / (float)detection_width;
//         float scaleY = input_image.rows / (float)detection_height;
// //        ROS_INFO("drawing phd estimation");
//         for(int k=0; k < num_drones; k++)
//         {
//             int scaledX = floor((kalman_filter_.X_k(0,k) + detection_offset_x) * scaleX);
//             int scaledY = floor((kalman_filter_.X_k(2,k) + detection_offset_y) * scaleY);

//             cv::Point2f target_center(scaledX,scaledY);
//             cv::Point2f id_pos(scaledX,scaledY+10);
//             cv::circle(input_image,target_center,6, cv::Scalar(0, 210, 255), 3);
//             putText(input_image, to_string(int(id_consensus(k))), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(0, 255, 0), 2, cv::LINE_AA);//size 1.5 --> 0.5
//         }

//         //measured input
//         for (int k=0; k < num_drones; k++)
//         {

//             int scaledX = floor((kalman_filter_.Detections(0,k) + detection_offset_x) * scaleX);
//             int scaledY = floor((kalman_filter_.Detections(1,k) + detection_offset_y) * scaleY);
//             float scaledW = kalman_filter_.Detections(2,k) * scaleX;
//             float scaledH = kalman_filter_.Detections(3,k) * scaleY;


//             cv::Point2f measured_center(scaledX, scaledY);
//             //cv::Point2f id_pos(phd_filter_.Z_k(0,k),phd_filter_.Z_k(1,k)+10);
//             cv::circle(input_image,measured_center,4, cv::Scalar(255, 0, 0), 2);
//             //              putText(previous_image, to_string(k), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(0, 255, 0), 2, cv::LINE_AA);//size 1.5 --> 0.5
//             cv::Point2f top_left(scaledX - scaledW/2, scaledY - scaledH/2);
//             cv::Point2f bottom_right(scaledX + scaledW/2, scaledY + scaledH/2);
//             cv::rectangle(input_image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);

//         }
//     }

//     //  ROS_INFO("drawing ground truth");
//     //  for(int k=0; k < vicon_projected_2DposeArray.cols(); k++)
//     //  {
//     //    cv::Point2f target_center(vicon_projected_2DposeArray(0,k),vicon_projected_2DposeArray(1,k));
//     //    cv::Point2f id_pos(vicon_projected_2DposeArray(0,k),vicon_projected_2DposeArray(1,k)+10);
//     //    cv::circle(input_image,target_center,4, cv::Scalar(0, 255, 0), 2);
//     //    putText(input_image, to_string(k), id_pos, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cvScalar(0, 255, 0), 2, CV_AA);//size 1.5 --> 0.5
//     //  }
//     // cv::imwrite("/home/greend/Desktop/0.png", input_image);
//     image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", input_image).toImageMsg();
//     image_msg->header.stamp = img_timestamp;
//     image_pub_.publish(image_msg);

//     //  ROS_WARN("img time: %f",prev_img_timestamp.toSec());
//     //  ROS_WARN("bbox time: %f",bbox_timestamp.toSec());

// #endif
//     return;

// }

void multi_robot_tracking_Nodelet::draw_image() {
#ifdef HOST
    if(filter_to_use_.compare("phd") == 0) {
        // 尺度缩放
        float scaleX = input_image.cols / (float)detection_width;
        float scaleY = input_image.rows / (float)detection_height;

        // 绘制检测框
        for (int k=0; k < phd_filter_.Detections.cols(); k++)
        {
            int scaledX = floor((phd_filter_.Detections(0,k) + detection_offset_x) * scaleX);
            int scaledY = floor((phd_filter_.Detections(1,k) + detection_offset_y) * scaleY);
            float scaledW = phd_filter_.Detections(2,k) * scaleX;
            float scaledH = phd_filter_.Detections(3,k) * scaleY;

            cv::Point2f top_left(scaledX - scaledW/2, scaledY - scaledH/2);
            cv::Point2f bottom_right(scaledX + scaledW/2, scaledY + scaledH/2);
            cv::rectangle(input_image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);
        }

        // 绘制跟踪目标和速度箭头
        for(int k=0; k < phd_filter_.X_k.cols(); k++)
        {
            int scaledX = floor((phd_filter_.X_k(0,k) + detection_offset_x) * scaleX);
            int scaledY = floor((phd_filter_.X_k(2,k) + detection_offset_y) * scaleY);
            
            if(scaledX > 0 && scaledX < input_image.cols && scaledY > 0 && scaledY < input_image.rows)
            {
                cv::Point2f target_center(scaledX, scaledY);
                cv::Point2f id_pos(scaledX, scaledY+20);
                
                // 绘制目标中心点
                cv::circle(input_image, target_center, 3, cv::Scalar(0, 210, 255), 2);
                
                // 绘制ID文本
                putText(input_image, to_string(int(id_consensus(k))), id_pos, 
                       cv::FONT_HERSHEY_COMPLEX_SMALL, 1.1, cvScalar(0, 0, 255), 1.1, cv::LINE_AA);
                
                // === 简化版速度箭头（使用OpenCV内置函数）===
                float vx = phd_filter_.X_k(1,k);
                float vy = phd_filter_.X_k(3,k);
                float speed = std::sqrt(vx*vx + vy*vy);
                
                if (speed > 0.1f) {
                    // 计算箭头终点
                    float arrow_scale = 1.0f;
                    cv::Point2f arrow_end(
                        scaledX + vx * arrow_scale,
                        scaledY + vy * arrow_scale
                    );
                    
                    // 使用OpenCV内置函数绘制箭头（黄色）
                    cv::arrowedLine(input_image, target_center, arrow_end, 
                                  cv::Scalar(0, 255, 255), 1, cv::LINE_AA, 0, 0.3);
                }
            }
        }
    }
    
    // 其他滤波器类型的绘制代码...
    
    // 发布图像
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", input_image).toImageMsg();
    image_msg->header.stamp = img_timestamp;
    image_pub_.publish(image_msg);

#endif
    return;
}




/* callback for 2D image to store before publishing
 * input: RGB Image
 * output: N/A
 */
#ifdef HOST
void multi_robot_tracking_Nodelet::image_Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    auto start_time1 = std::chrono::high_resolution_clock::now();

    //image timestamp is faster than detection result timestamp by approx 1 second. store image in buffer
    image_buffer_.push_back(img_msg);  //将接收到的图像消息存入`image_buffer_`
    //cout<<"lalala"<<endl;
    // ROS_INFO("img buff size: %lu",image_buffer_.size());

    //look up first image with timestamp smaller than
    
    for(int i = 0; i < image_buffer_.size(); i++)
    {
        //cout<<"image_buffer_[i]->header.stamp.toSec()="<<image_buffer_[i]->header.stamp.toSec()<<endl;
        //cout<<"current_timestamp="<<current_timestamp<<endl;
        //cout<<"differ"<<image_buffer_[i]->header.stamp.toSec() - current_timestamp<<endl;
        //cout<<"lalala"<<endl;
        //cout<<"进入for循环"<<endl;
        // ROS_INFO("looking for timestamp less than: %f, image buff[i].stamp: %f",current_timestamp, image_buffer_[i]->header.stamp.toSec() );
        //if found image with matching detection sequence
        //current_timestamp=0;
        if(image_buffer_[i]->header.stamp.toSec() >= current_timestamp)
        {
            // ROS_INFO("detection timestamp: %f, image buff[i].stamp: %f", current_timestamp, image_buffer_[i]->header.stamp.toSec() );
            //ROS_INFO("FOUND IMG matchlalala");
            cout << "matched image buff index" << i << " Buffer Size: " << image_buffer_.size() <<endl;
            auto sync_image_ptr = image_buffer_[i];
            //store img pointer
            img_timestamp = image_buffer_[i]->header.stamp;
            cv_bridge::CvImageConstPtr im_ptr_ = cv_bridge::toCvShare(sync_image_ptr, "rgb8");
            input_image = im_ptr_->image;

            //draw image with matched image
            //draw_image();

            //remove from img buffer
            image_buffer_.erase(image_buffer_.begin() + i);

            break;
        }

    }
        auto end_time1 = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1 - start_time1);
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time1 - start_time1);
            std::cout << "image_callback() 运行时间：\n";
            std::cout << duration_ms.count() << " 毫秒\n"; 
            std::cout << duration_us.count() << " 微秒\n";
}
#endif
/* callback for imu to store for faster motion prediction
 * input: IMU Image
 * output: N/A
 */
void multi_robot_tracking_Nodelet::imu_Callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    //use imu buffer
    
    ros::Duration timeDifIMU = imu_msg->header.stamp - imu_timestamp;
    phd_filter_.dt_imu = timeDifIMU.toSec();  //IMU消息之间的时间间隔，用于PHD滤波器的异步预测
    ROS_WARN("imu time: %f", phd_filter_.dt_imu);
    imu_.angular_velocity = imu_msg->angular_velocity;
    imu_.angular_velocity_covariance = imu_msg->angular_velocity_covariance;
    imu_.header = imu_msg->header;
    imu_.linear_acceleration = imu_msg->linear_acceleration;
    imu_.linear_acceleration_covariance = imu_msg->linear_acceleration_covariance;
    imu_.orientation = imu_msg->orientation;
    imu_.orientation_covariance = imu_msg->orientation_covariance;
    if(phd_filter_.first_callback == false)
    {
        phd_filter_.ang_vel_k(0) = imu_msg->angular_velocity.x;
        phd_filter_.ang_vel_k(1) = imu_msg->angular_velocity.y;
        phd_filter_.ang_vel_k(2) = imu_msg->angular_velocity.z;

        //apply rotation from imu2cam frame
        phd_filter_.ang_vel_k.block<3,1>(0,0) = rotm_world2cam *  phd_filter_.ang_vel_k;

        //asynchronous motion prediction
        if(first_track_flag)
        {
            phd_filter_.asynchronous_predict_existing();  //phd的预测步骤
            publish_tracks();
        }
    }
    // if(filter_to_use_.compare("kalman") == 0)
    // {
    //     kalman_filter_.ang_vel_k(0) = imu_msg->angular_velocity.x;
    //     kalman_filter_.ang_vel_k(1) = imu_msg->angular_velocity.y;
    //     kalman_filter_.ang_vel_k(2) = imu_msg->angular_velocity.z;

    //     //apply rotation from imu2cam frame
    //     kalman_filter_.ang_vel_k.block<3,1>(0,0) = rotm_world2cam *  phd_filter_.ang_vel_k;

    //     //asynchronous motion prediction
    //     if(first_track_flag)
    //     {
    //         kalman_filter_.kalmanPredict();
    //         publish_tracks();
    //     }
    // }
    imu_timestamp = imu_msg->header.stamp;
}

Eigen::MatrixXf multi_robot_tracking_Nodelet::get_B_ang_vel_matrix(float x, float y)
{
    Eigen::MatrixXf temp_B_matrix;
    temp_B_matrix = Eigen::MatrixXf::Zero(4,3);

    temp_B_matrix(0,0) = (x-cx)*(y-cy)/f;       temp_B_matrix(0,1) = -(pow((x-cx),2)/f)-f;  temp_B_matrix(0,2) = (y-cy);
    temp_B_matrix(1,0) = 0;                     temp_B_matrix(1,1) = 0;                     temp_B_matrix(1,2) = 0;
    temp_B_matrix(2,0) = f+(pow((y-cy),2))/f;   temp_B_matrix(2,1) = (x-cx)*(y-cy)/f;       temp_B_matrix(2,2) = -x+cx;
    temp_B_matrix(3,0) = 0;                     temp_B_matrix(3,1) = 0;                     temp_B_matrix(3,2) = 0;

    temp_B_matrix = temp_B_matrix * phd_filter_.dt_imu;
    return temp_B_matrix;

}


/* callback for 2D image to call phd track when using flightmare rosbag data
 * input: PoseArray
 * output: N/A
 */
void multi_robot_tracking_Nodelet::detection_Callback(const geometry_msgs::PoseArray& in_PoseArray)
{   
    auto start_time = std::chrono::high_resolution_clock::now();
    static int callback_count = 0; // 静态变量，用于记录调用次数
    ros::Time current_time = ros::Time::now(); // 获取当前时间
    //std::cout << "detection_Callback called " << ++callback_count << " times at " << current_time.toSec() << std::endl;
    if(in_PoseArray.poses.size() > num_drones)
    {
        ROS_ERROR("MORE DETECTIONS THAN NO OF DRONES !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return;
    }

    //get time of detection
    bbox_timestamp = in_PoseArray.header.stamp;
    current_timestamp = bbox_timestamp.toSec();
    //cout<<"bbox_timestamp.toSec()========"<<bbox_timestamp.toSec()<<endl;



    ROS_WARN("bbox time: %f, dt: %f, imu time: %f",current_timestamp, delta_timestamp, imu_time);

    // ROS_INFO("detected size: %lu ", in_PoseArray.poses.size() );
    jpdaf_filter_.last_timestamp_synchronized = in_PoseArray.header.stamp.toSec(); //用于JPDAF滤波器记录上一次同步的时间

    //store Z

    //========= use jpdaf filter ===========
    if(filter_to_use_.compare("jpdaf") == 0)
    {

        jpdaf_filter_.detected_size_k = in_PoseArray.poses.size();

        //store Z
        jpdaf_filter_.flightmare_bounding_boxes_msgs_buffer_.push_back(in_PoseArray);
        //store imu
        jpdaf_filter_.imu_buffer_.push_back(imu_);

        //    jpdaf_filter_.Z_k = Eigen::MatrixXf::Zero(4,jpdaf_filter_.detected_size_k);

        //    for(int i =0; i < jpdaf_filter_.detected_size_k; i++)
        //    {
        //      jpdaf_filter_.Z_k(0,i) = in_PoseArray.poses[i].position.x;
        //      jpdaf_filter_.Z_k(1,i) = in_PoseArray.poses[i].position.y;
        //    }

        jpdaf_filter_.track(true);
    }

    //========= use phd filter ===========
    else if(filter_to_use_.compare("phd") == 0)
    {

        if(phd_filter_.first_callback)
        {
            phd_filter_.set_num_drones(num_drones);
            phd_filter_.initialize_matrix(cx, cy, f, filter_dt);
        }

        phd_filter_.Detections.setZero();  //每次都将检测到的目标位置和方向初始化为0
        phd_filter_.detected_size_k = in_PoseArray.poses.size();

        for(int i =0; i < phd_filter_.detected_size_k; i++)
        {
            //store Z
            // x, y, w, h
            phd_filter_.Z_k(0,i) = in_PoseArray.poses[i].position.x;
            phd_filter_.Z_k(1,i) = in_PoseArray.poses[i].position.y;

            phd_filter_.Detections(0,i) = in_PoseArray.poses[i].position.x;
            phd_filter_.Detections(1,i) = in_PoseArray.poses[i].position.y;
            phd_filter_.Detections(2,i) = in_PoseArray.poses[i].orientation.x;
            phd_filter_.Detections(3,i) = in_PoseArray.poses[i].orientation.y;
        }
        ROS_INFO_STREAM("Num Meas: " << phd_filter_.detected_size_k << "\n");
        ROS_INFO_STREAM("Z_k_CB: " << endl << phd_filter_.Z_k << "\n");
        ROS_INFO_STREAM("WK-1: " << phd_filter_.wk << "\n");
        if(phd_filter_.first_callback)
        {
            delta_timestamp =0.033;//0.043; //0.025;//0.143; //0.225   0.125
            phd_filter_.dt_cam = delta_timestamp;  //摄像头检测之间的时间间隔，用于PHD滤波器的主更新

            phd_filter_.initialize(q_pos, q_vel, r_meas, p_pos_init, p_vel_init,
                                phd_prune_weight_threshold,
                                phd_prune_mahalanobis_dist_threshold,
                                phd_extract_weight_threshold);
            phd_filter_.first_callback = false;

            previous_timestamp = current_timestamp;
        }

        else 
        {


            delta_timestamp =0.025;//0.043; //0.025; //hard-coded for 4.5 Hz TO DO FIX   0.143
            //      delta_timestamp = current_timestamp - previous_timestamp;
            //check for data with no timestamp and thus dt = 0

            phd_filter_.dt_cam = delta_timestamp;
            previous_timestamp = current_timestamp;
            for(int i =0; i < phd_filter_.X_k.cols(); i++)
            {
                phd_filter_.B.block<4,3>(0,3*i) = get_B_ang_vel_matrix(phd_filter_.X_k(0,i),phd_filter_.X_k(2,i));
            }

            phd_filter_.phd_track();   //运行gmphd    执行更新步骤
            //id_consensus = phd_filter_.id_consensus;
            // 更新ID状态矩阵 上边的注释掉了
            update_id_status_matrix();
            
            // 处理新检测的目标
            process_new_detections();
            
            // 更新id_consensus
            update_id_consensus_from_status();
            draw_image();
            ROS_ERROR_STREAM("id_status_matrix_: \n" << id_status_matrix_);
            // auto end_time = std::chrono::high_resolution_clock::now();
            // auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            // auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            // std::cout << "phd_track() 运行时间：\n";
            // std::cout << duration_ms.count() << " 毫秒\n"; 
            // std::cout << duration_us.count() << " 微秒\n";
            
            first_track_flag = true;
            consensus_sort();   //################################################################################################################################
            //associate_ids();
            //after tracking, store previous Z value to update velocity
            for(int i =0; i < phd_filter_.detected_size_k; i++)
            {
                //store Z
                phd_filter_.Z_k_previous(0,i) = in_PoseArray.poses[i].position.x;
                phd_filter_.Z_k_previous(1,i) = in_PoseArray.poses[i].position.y;

            }


            phd_filter_.B = Eigen::MatrixXf::Zero(4,3*num_drones);

            //update for B ang vel matrix
            //store B matrix for ang velocity
            

           // imu_timestamp = in_PoseArray.header.stamp;
        }
    }

    else if(filter_to_use_.compare("kalman") == 0)
    {

        if(kalman_filter_.first_callback)
        {
            kalman_filter_.setNumDrones(num_drones);
            kalman_filter_.initializeMatrix(cx, cy, f, filter_dt);
        }


        kalman_filter_.detected_size_k = in_PoseArray.poses.size();
        

        for(int i =0; i < kalman_filter_.detected_size_k; i++)
        {
            //store Z
            // x, y, w, h
            kalman_filter_.Z_k(0,i) = in_PoseArray.poses[i].position.x;
            kalman_filter_.Z_k(1,i) = in_PoseArray.poses[i].position.y;

            kalman_filter_.Detections(0,i) = in_PoseArray.poses[i].position.x;
            kalman_filter_.Detections(1,i) = in_PoseArray.poses[i].position.y;
            kalman_filter_.Detections(2,i) = in_PoseArray.poses[i].orientation.x;
            kalman_filter_.Detections(3,i) = in_PoseArray.poses[i].orientation.y;
        }
        ROS_INFO_STREAM("Num Meas: " << kalman_filter_.detected_size_k << "\n");
        ROS_INFO_STREAM("Z_k_CB: " << endl << kalman_filter_.Z_k << "\n");
        ROS_INFO_STREAM("WK-1: " << kalman_filter_.wk << "\n");
        if(kalman_filter_.first_callback)
        {
            delta_timestamp = 0.125;//0.143; //0.225
            kalman_filter_.dt_cam = delta_timestamp;

            kalman_filter_.initialize(q_pos, q_vel, r_meas, p_pos_init, p_vel_init,
                                phd_prune_weight_threshold,
                                phd_prune_mahalanobis_dist_threshold,
                                phd_extract_weight_threshold);
            kalman_filter_.first_callback = false;

            previous_timestamp = current_timestamp;
        }

        else
        {


            delta_timestamp = 0.143; //hard-coded for 4.5 Hz TO DO FIX
            //      delta_timestamp = current_timestamp - previous_timestamp;
            //check for data with no timestamp and thus dt = 0

            kalman_filter_.dt_cam = delta_timestamp;
            previous_timestamp = current_timestamp;
            for(int i =0; i < phd_filter_.X_k.cols(); i++)
            {
                kalman_filter_.B.block<4,3>(0,3*i) = get_B_ang_vel_matrix(phd_filter_.X_k(0,i),phd_filter_.X_k(2,i));
            }

            kalman_filter_.kalmanTrack();
            ROS_INFO_STREAM("Finished track");
            first_track_flag = true;

            //after tracking, store previous Z value to update velocity
            // for(int i =0; i < phd_filter_.detected_size_k; i++)
            // {
            //     //store Z
            //     kalman_filter_.Z_k_previous(0,i) = in_PoseArray.poses[i].position.x;
            //     kalman_filter_.Z_k_previous(1,i) = in_PoseArray.poses[i].position.y;

            // }


            kalman_filter_.B = Eigen::MatrixXf::Zero(4,3*num_drones);

            //update for B ang vel matrix
            //store B matrix for ang velocity
            

            // consensus_sort();

            // imu_timestamp = in_PoseArray.header.stamp;
        }
    }

    ROS_INFO_STREAM("Pub track");
    publish_tracks();

// ===== 新增：生成跟踪CSV =====
if (filter_to_use_.compare("phd") == 0 && tracking_csv_.is_open()) {
    // 确定目标数量
    int num_targets = phd_filter_.X_k.cols();
    
    // 定义缓冲区存储上一次的宽度和高度信息
    static std::vector<float> prev_widths;
    static std::vector<float> prev_heights;
    if (prev_widths.size() < num_targets) {
        prev_widths.resize(num_targets, 50);
        prev_heights.resize(num_targets, 50);
    }

    for (int i = 0; i < num_targets; i++) {
        // 获取目标ID
        int target_id = (i < id_consensus.size()) ? id_consensus(i) : i;

        // 获取跟踪估计的中心位置
        float track_center_x = phd_filter_.X_k(0, i);
        float track_center_y = phd_filter_.X_k(2, i);

        // 获取边界框信息
        float center_x = 0, center_y = 0, width = 50, height = 50;
        bool valid_detection = false;

        // 动态计算距离阈值
        float distance_threshold = 0.0f;

        // 优先使用当前帧检测，并进行筛选
        if (i < in_PoseArray.poses.size()) {
            center_x = in_PoseArray.poses[i].position.x;
            center_y = in_PoseArray.poses[i].position.y;
            width = in_PoseArray.poses[i].orientation.x;
            height = in_PoseArray.poses[i].orientation.y;

            // 计算动态距离阈值
            distance_threshold = (width + height) / 2.0f;

            // 计算检测框中心与跟踪估计中心的距离
            float distance = std::hypot(center_x - track_center_x, center_y - track_center_y);
            if (distance < distance_threshold) {
                valid_detection = true;
                // 更新缓冲区
                prev_widths[i] = width;
                prev_heights[i] = height;
            }
        }

        // 若当前帧检测无效，使用缓冲区中的上一次宽度和高度
        if (!valid_detection) {
            center_x = track_center_x;
            center_y = track_center_y;
            width = prev_widths[i];
            height = prev_heights[i];
        }

        // 将浮点数值四舍五入为整数
        int bb_left = static_cast<int>(std::round(center_x - width/2.0f));
        int bb_top = static_cast<int>(std::round(center_y - height/2.0f));
        int bb_width = static_cast<int>(std::round(width));
        int bb_height = static_cast<int>(std::round(height));
        
        // 检查是否有任何数值为负数
        bool has_negative = (bb_left < 0) || (bb_top < 0) || 
                            (bb_width < 0) || (bb_height < 0) ||
                            (target_id < 0) || (frame_count_ + 1 < 0);
        bool exceeds_height = (bb_top + bb_height / 2.0f) > detection_height;
        bool exceeds_width = (bb_left + bb_width / 2.0f) > detection_width;
        if (!has_negative && !exceeds_height && !exceeds_width) {
            // 写入CSV行（全部使用整数）
            tracking_csv_ << frame_count_+1 << ","
                          << target_id+1 << ","
                          << bb_left << "," << bb_top << ","
                          << bb_width << "," << bb_height << ","
                          << 1 << ","  // 置信度
                          << 1 << ","    // 类别
                          << 1 << "\n"; // 可见度
        }

    }

    frame_count_++;
}

auto end_time = std::chrono::high_resolution_clock::now();
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            std::cout << "detection_callback() 运行时间：\n";
            std::cout << duration_ms.count() << " 毫秒\n"; 
            std::cout << duration_us.count() << " 微秒\n";
}


void multi_robot_tracking_Nodelet::process_new_detections() {
    // 这里可以添加逻辑来处理新检测到的目标
    // 比如当检测到新目标时，调用find_available_id来分配ID
}

void multi_robot_tracking_Nodelet::update_id_consensus_from_status() {
    // 确保id_consensus是正确类型（整数矩阵）
    if (id_consensus.cols() != num_drones) {
        id_consensus = Eigen::MatrixXi(1, num_drones);
    }
    
    // 安全地赋值
    for (int i = 0; i < num_drones && i < id_consensus.cols(); i++) {
        id_consensus(i) = i;
    }
    
    ROS_DEBUG("更新id_consensus完成，大小: %d", id_consensus.cols());
}



/* given known initial projected coordinated, and Left/Right ID
 * sort the id_consensus according to the estimated target
 */
void multi_robot_tracking_Nodelet::consensus_sort()
{
    if(!consensus_sort_complete)
    {
        float euclidian_distance = 0;
        float min_distance = 1000000;
        int min_index = 0;
        //get init proj

        //get measurement

        //compare euclidian distance for each target with init projection
        for (int z = 0; z < phd_filter_.X_k.cols(); z++)
        {
            min_index = 0;
            min_distance = 100000;

            for(int index = 0; index < num_drones; index++)
            {
                cout << "X_k: " << phd_filter_.X_k(0,z) << "projX: " << projected_2d_initial_coord(0,index) << endl;
                euclidian_distance = abs(phd_filter_.X_k(0,z) - projected_2d_initial_coord(0,index)) + abs(phd_filter_.X_k(1,z) - projected_2d_initial_coord(1,index)) ;
                //store min index
                if(euclidian_distance < min_distance)
                {
                    min_distance = euclidian_distance;
                    min_index = index;
                }
            }
            id_consensus(z) = int(id_array_init(min_index));

        }

        //sort ID accordingly
        consensus_sort_complete = true;
        cout << " **************** consensus ID: " << id_consensus  << "**************** " << endl;
    }

}


/*
 *
 */
void multi_robot_tracking_Nodelet::associate_consensus()
{
    //ROS_WARN("inside consensus func");

    //get delta init positions in world coordinate
    Eigen::MatrixXf delta_left_to_self, delta_right_to_self;
    Eigen::MatrixXf projected_2d_padding;
    delta_left_to_self = Eigen::MatrixXf::Zero(3,1);
    delta_right_to_self = Eigen::MatrixXf::Zero(3,1);
    projected_2d_padding = Eigen::MatrixXf::Zero(3,num_drones);

    //delta_left_to_self(0) = init_pos_x_left - init_pos_self_x;
    //delta_left_to_self(1) = init_pos_y_left - init_pos_self_y;
    delta_left_to_self(0) = 3.4641;
    delta_left_to_self(1) = 2;
    delta_left_to_self(2) = 0;

    //    delta_right_to_self(0) = init_pos_x_right - init_pos_self_x;
    //    delta_right_to_self(1) = init_pos_y_right - init_pos_self_y;
    delta_right_to_self(0) = 3.4641;
    delta_right_to_self(1) = -2;
    delta_right_to_self(2) = 0;

    cout << "deltaL: " << endl << delta_left_to_self << endl;
    cout << "deltaR: " << endl << delta_right_to_self << endl;

    positions_world_coordinate.block<3,1>(0,0) = delta_left_to_self;
    positions_world_coordinate.block<3,1>(0,1) = delta_right_to_self;

    //get delta init positions in camera coordinate
    positions_cam_coordinate.block<3,1>(0,0) = rotm_world2cam * positions_world_coordinate.block<3,1>(0,0) ;
    positions_cam_coordinate.block<3,1>(0,1) = rotm_world2cam * positions_world_coordinate.block<3,1>(0,1) ;

    cout << "positions_cam_coordinate: " << endl << positions_cam_coordinate << endl;


    //project into 2D space
    for(int i =0; i < num_drones;i++)
    {
        projected_2d_padding.block<3,1>(0,i) = k_matrix3x3 * positions_cam_coordinate.block<3,1>(0,i);
        projected_2d_padding.block<3,1>(0,i) = projected_2d_padding.block<3,1>(0,i) / projected_2d_padding(2,i);
        projected_2d_initial_coord.block<2,1>(0,i) = projected_2d_padding.block<2,1>(0,i);
    }

    cout << "projected_2d_initial_coord: " << endl << projected_2d_initial_coord << endl;


}


void multi_robot_tracking_Nodelet::publish_tracks()
{
//    ROS_INFO("publish tracks");

    geometry_msgs::PoseArray tracked_output_pose, tracked_velocity_pose;
    geometry_msgs::Pose temp_pose, temp_velocity;

    if(filter_to_use_.compare("jpdaf") == 0)
    {
        //TO DO fill in publishing for jpdaf as well
    }

    else
    {

        //store estimated PHD X_k into tracked output
        // for(int i =0; i < phd_filter_.X_k.cols(); i++) {

        //     temp_pose.position.x = phd_filter_.X_k(0,i);
        //     temp_pose.position.y = phd_filter_.X_k(1,i);
        //     tracked_output_pose.poses.push_back(temp_pose);

        //     temp_velocity.position.x = phd_filter_.X_k(2,i);
        //     temp_velocity.position.y = phd_filter_.X_k(3,i);
        //     tracked_velocity_pose.poses.push_back(temp_velocity);
        // }
        for(int i =0; i < phd_filter_.X_k.cols(); i++) {

            temp_pose.position.x = phd_filter_.X_k(0,i);
            temp_pose.position.y = phd_filter_.X_k(2,i);
            tracked_output_pose.poses.push_back(temp_pose);

            temp_velocity.position.x = phd_filter_.X_k(1,i);
            temp_velocity.position.y = phd_filter_.X_k(3,i);
            temp_velocity.orientation.w = sqrt (phd_filter_.X_k(1,i)*phd_filter_.X_k(1,i) + phd_filter_.X_k(3,i)*phd_filter_.X_k(3,i));   //模长
            temp_velocity.orientation.z = atan2(phd_filter_.X_k(3,i),phd_filter_.X_k(1,i)); //弧度
            tracked_velocity_pose.poses.push_back(temp_velocity);
        }
    }

    //publish output (stored as either jpdaf or phd)
    tracked_output_pose.header.stamp = bbox_timestamp;
    tracked_velocity_pose.header.stamp = bbox_timestamp;
    tracked_pose_pub_.publish(tracked_output_pose);
    tracked_velocity_pub_.publish(tracked_velocity_pose);

    //empty tracked output
    while(!tracked_output_pose.poses.empty()) {
        tracked_output_pose.poses.pop_back();
    }

    while(!tracked_velocity_pose.poses.empty()) {
        tracked_velocity_pose.poses.pop_back();
    }

//    ROS_INFO("end publish tracks");

}

/* Nodelet init function to handle subscribe/publish
 * input: N/A
 * output: N/A
 */
void multi_robot_tracking_Nodelet::onInit(void)
{
    ros::NodeHandle nh = getNodeHandle();
    ros::NodeHandle priv_nh(getPrivateNodeHandle());
    
    if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Error) ) {
        ros::console::notifyLoggerLevelsChanged();
    }

    image_transport::ImageTransport it(nh);

    priv_nh.param<std::string>("filter",filter_to_use_,"phd"); //store which filter to use
    priv_nh.param<std::string>("input_bbox_topic",input_bbox_topic,"/DragonPro1/snpe_ros/detections"); //input bbox topic
    priv_nh.param<std::string>("input_img_topic",input_img_topic,"/usb_cam/image_raw"); //input img topic
    priv_nh.param<std::string>("input_imu_topic",input_imu_topic,"/DragonPro1/imu"); //input imu topic

    priv_nh.param<int>("num_drones",num_drones,2);

    //consensus - init coordinates read in from launch file
    priv_nh.param<float>("max_association_distance", max_association_distance_, 50.0f);
    priv_nh.param<float>("init_pos_self_x",init_pos_self_x,0);
    priv_nh.param<float>("init_pos_self_y",init_pos_self_y,0);
    priv_nh.param<float>("init_pos_x_left",init_pos_x_left,0);
    priv_nh.param<float>("init_pos_y_left",init_pos_y_left,0);
    priv_nh.param<float>("init_pos_x_right",init_pos_x_right,0);
    priv_nh.param<float>("init_pos_y_right",init_pos_y_right,0);

    priv_nh.param<int>("id_left",id_left,0);
    priv_nh.param<int>("id_right",id_right,0);

    priv_nh.param<float>("camera_cx", cx, 0);
    priv_nh.param<float>("camera_cy", cy, 0);
    priv_nh.param<float>("camera_f", f, 0);
    priv_nh.param<float>("dt", filter_dt, 0.225);

    priv_nh.param<int>("viz_detection_height", detection_height, 168);
    priv_nh.param<int>("viz_detection_width", detection_width, 224);
    priv_nh.param<int>("viz_detection_offset_x", detection_offset_x, 0);
    priv_nh.param<int>("viz_detection_offset_y", detection_offset_y, 28);
    priv_nh.param<bool>("use_generated_id", consensus_sort_complete,0);

    priv_nh.param<float>("phd/q_pos", q_pos, 6.25);
    priv_nh.param<float>("phd/q_vel", q_vel, 12.5);
    priv_nh.param<float>("phd/p_pos_init", p_pos_init, 5.0);
    priv_nh.param<float>("phd/p_vel_init", p_vel_init, 2.0);
    priv_nh.param<float>("phd/r_meas", r_meas, 45);
    priv_nh.param<float>("phd/prune_weight_threshold", phd_prune_weight_threshold, 1e-1);
    priv_nh.param<float>("phd/prune_mahalanobis_threshold_", phd_prune_mahalanobis_dist_threshold, 4.0);
    priv_nh.param<float>("phd/extract_weight_threshold", phd_extract_weight_threshold, 5e-1);

    priv_nh.param<float>("jpdaf/q_pos", q_pos, 6.25);
    priv_nh.param<float>("jpdaf/q_vel", q_vel, 12.5);
    priv_nh.param<float>("jpdaf/p_pos_init", p_pos_init, 5.0);
    priv_nh.param<float>("jpdaf/p_vel_init", p_vel_init, 2.0);
    priv_nh.param<float>("jpdaf/r_meas", r_meas, 45);
    priv_nh.param<float>("jpdaf/alpha_0_threshold", jpdaf_filter_.params.alpha_0_threshold); 
    priv_nh.param<float>("jpdaf/alpha_cam", jpdaf_filter_.params.alpha_cam);
    priv_nh.param<float>("jpdaf/associaiton_cost", jpdaf_filter_.params.assoc_cost);
    priv_nh.param<float>("jpdaf/beta_0_threshold", jpdaf_filter_.params.beta_0_threshold);
    priv_nh.param<float>("jpdaf/false_measurements_density", jpdaf_filter_.params.false_measurements_density);
    priv_nh.param<float>("jpdaf/gamma", jpdaf_filter_.params.gamma);
    priv_nh.param<float>("jpdaf/max_missed_rate",jpdaf_filter_.params.max_missed_rate);
    priv_nh.param<float>("jpdaf/min_acceptance_rate", jpdaf_filter_.params.min_acceptance_rate);
    priv_nh.param<float>("jpdaf/probability_detection", jpdaf_filter_.params.pd);

    ROS_INFO_STREAM("Consensus sort during init " << consensus_sort_complete);

    if(filter_to_use_.compare("phd") == 0) //using phd
    {
        ROS_WARN("will be using: %s", filter_to_use_.c_str());
        init_matrices(); //initialize matrix for storing 3D pose
        //associate_consensus(); //determine 2d position from known init positions

    }

    else if(filter_to_use_.compare("kalman") == 0) //using kalman
    {
        ROS_WARN("will be using: %s", filter_to_use_.c_str());
        init_matrices(); //initialize matrix for storing 3D pose
        //associate_consensus(); //determine 2d position from known init positions

    }

    else if (filter_to_use_.compare("jpdaf") == 0) 
    {
        jpdaf_filter_.params.focal_length = f;
        jpdaf_filter_.params.nb_drones = num_drones;
        jpdaf_filter_.params.P_0 << p_pos_init, 0, 0, 0,
                                    0, p_vel_init, 0, 0,
                                    0, 0, p_pos_init, 0,
                                    0, 0, 0, p_vel_init;
        jpdaf_filter_.params.principal_point << cx, cy;
        jpdaf_filter_.params.R <<   r_meas, 0,
                                    0, r_meas;
        ROS_WARN("will be using: %s", filter_to_use_.c_str());
    }

    else 
    {
        ROS_ERROR("wrong filter param input");
        return;
    }


    //bbox subscription of PoseArray Type
    detection_sub_ = priv_nh.subscribe(input_bbox_topic, 10, &multi_robot_tracking_Nodelet::detection_Callback, this);
    //img subscription
#ifdef HOST
    image_sub_ = priv_nh.subscribe(input_img_topic, 10, &multi_robot_tracking_Nodelet::image_Callback, this);
#endif
    //imu subscription
    imu_sub_ = priv_nh.subscribe(input_imu_topic, 10, &multi_robot_tracking_Nodelet::imu_Callback, this);
    //groundtruth bbox subscription


    image_pub_ = it.advertise("tracked_image",1);
    tracked_pose_pub_ = nh.advertise<geometry_msgs::PoseArray>("tracked_pose_output",1);
    tracked_velocity_pub_ = nh.advertise<geometry_msgs::PoseArray>("tracked_velocity_output",1);

    //init export csv file
    outputFile.open("/home/mwr/tracking/src/multi_robot_tracking/out.csv");
    //outputFile << "Time" << "," << "GND_truth_X1" << "," << "GND_truth_Y1" << "," << "GND_truth_X2" << "," << "GND_truth_Y2" << ","
    //           << "est_X1" << "," << "est_Y1" << ","  << "est_X2" << "," << "est_Y2" << "," <<  std::endl;



    std::string csv_path = "/home/mwr/tracking/src/multi_robot_tracking/tracking_results.csv";
    tracking_csv_.open(csv_path);
    if (tracking_csv_.is_open()) {
        //tracking_csv_ << "frame,id,bb_left,bb_top,bb_width,bb_height,conf,class,visibility\n";
        ROS_INFO("Tracking CSV opened: %s", csv_path.c_str());
    } else {
        ROS_ERROR("Failed to open tracking CSV: %s", csv_path.c_str());
    }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(multi_robot_tracking_Nodelet, nodelet::Nodelet);