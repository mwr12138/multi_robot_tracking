#include <Eigen/Geometry>
#include <Eigen/Dense>
//#include <eigen_conversions/eigen_msg.h>
#include <math.h>

//ros
#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

//detection bbox
//#include <darknet_ros_msgs/BoundingBoxes.h>
//#include <darknet_ros_msgs/BoundingBox.h>

//CV
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <deque> 


#define PI 3.14159
//#define NUM_DRONES 3

// ===== Tracknew definition =====
struct Tracknew {
    int id;                      // 唯一 ID
    Eigen::VectorXf x;           // 状态
    Eigen::MatrixXf P;           // 协方差
    float confidence;            // 可信度（暂时用权重）
    int missed_count;            // 连续未匹配帧数
    bool active;                 // 是否仍然有效
    int match_type;              // 匹配类型    
    // 0: Miss (没匹配到/遮挡)
    // 1: Match (正常匹配)
    // 2: Revive (借尸还魂/复活)
    // 3: Birth (新生/投胎)

    std::deque<Eigen::Vector2f> position_history;  // 位置历史
    std::deque<Eigen::Vector2f> velocity_history;  // 速度历史
    Eigen::Vector2f velocity; //当前帧的卡尔曼/预测计算
};

//从PHD到track的候选结构体
struct Candidate {
    Eigen::VectorXf x;
    Eigen::MatrixXf P;
    float w;
};


class PhdFilter
{
 public:
  PhdFilter();

  void initialize_matrix(float cam_cu, float cam_cv, float cam_f, float meas_dt=0.225);
  void initialize(float q_pos, float q_vel, float r_meas, float p_pos_init, float p_vel_init,
                float prune_weight_threshold, float prune_mahalanobis_threshold, float extract_weight_threshold);
  void phd_track();
  void phd_predict_existing();
  void phd_construct();
  void phd_update();
  void phd_prune();
  void phd_state_extract();
  void asynchronous_predict_existing();
  void removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove);
  void removeColumnf(Eigen::MatrixXf& matrix, unsigned int colToRemove);
  void removeColumni(Eigen::MatrixXi& matrix, unsigned int colToRemove);

  void set_num_drones(int num_drones);
  float clutter_intensity(const float X, const float Y);
  Eigen::MatrixXf left_divide(const Eigen::MatrixXf);
  void update_F_matrix(float input_dt);
  void update_A_matrix(float input_dt);


  ros::Time startTime,endTime,processTime;

//新增、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\、0104新增
  void updateTracks(const std::vector<Candidate>& candidates);
  void update_track_data(Tracknew& tr, const Candidate& cand);
  void create_new_track(const Candidate& cand, int id);//创建新轨迹
  Eigen::Vector2f predict_position(const Tracknew& tr) const;//新 基于历史位置预测当前位置
  float calculate_velocity_consistency(const Tracknew& tr, const Eigen::Vector2f& candidate_velocity) const; //新 计算速度连续性得分
  void update_track_data_directional(Tracknew& tr, const Candidate& cand);
  // 在 PhdFilter.h 或初始化函数中
  void initTracks();

//新增、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、、\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\0104新增


// 添加速度历史记录
  std::vector<std::deque<Eigen::Vector2f>> velocity_history;  // 每个目标的速度历史
  std::vector<std::deque<Eigen::Vector2f>> position_history;  // 每个目标的位置历史
  const int HISTORY_SIZE = 5;  // 保留最近5帧的历史
  //void update_velocity_history(); //更新速度历史
  void initialize_velocity_history();
  //float calculate_velocity_consistency(int target_id, const Eigen::Vector2f& candidate_velocity); //计算速度连续性得分===========================旧的=========
  //float calculate_position_consistency(int target_id, const Eigen::Vector2f& candidate_position); //计算位置连续性得分
  //Eigen::Vector2f predict_position(int target_id);//基于历史位置预测当前位置================================旧的=================
  //int find_target_using_id(int target_id, const Eigen::MatrixXi& newIndex, 
  //                         const std::vector<std::pair<float, int>>& weighted_targets, 
  //                         const std::vector<bool>& id_used);// 查找当前使用指定ID的目标索引
  //void assign_target_to_id(int source_idx, int target_id, 
  //                     const Eigen::MatrixXf& wk_bar_fixed_k,
  //                     const Eigen::MatrixXf& mk_bar_fixed_k,
  //                     const Eigen::MatrixXf& Pk_bar_fixed_k);
  //void cleanup_memory(); // 清理内存
  int memory_cleanup_counter;
  static const int MEMORY_CLEANUP_INTERVAL = 100;
  Eigen::MatrixXi id_consensus;


  geometry_msgs::PoseArray Z_current_k;
  int NUM_DRONES;
  int detected_size_k;

  float last_timestamp_synchronized;
  double last_timestamp_from_rostime;
  bool first_callback = true;
  int numTargets_Jk_k_minus_1;
  int numTargets_Jk_minus_1;
  int L = 0;

  int k_iteration = 0;
  bool flag_asynch_start = false;
  bool enable_async = true;


  //kalman filter variables
  Eigen::MatrixXf F;
  Eigen::MatrixXf A;
  Eigen::MatrixXf H;
  Eigen::MatrixXf Q;
  Eigen::MatrixXf R;
  Eigen::MatrixXf K;

  //phd variables

  float prob_survival = 1;  //目标存在的概率   这里为什么是1.0？？？
  float prob_detection = 0.9;//1.0;

  float dt_cam = 0.033; //0.125; //8hz
  float dt_imu = 0.01;  //100hz

  Eigen::MatrixXf mk_minus_1;
  Eigen::MatrixXf wk_minus_1;
  Eigen::MatrixXf Pk_minus_1;
  Eigen::MatrixXf mk_k_minus_1;
  Eigen::MatrixXf wk_k_minus_1;
  Eigen::MatrixXf Pk_k_minus_1;
  Eigen::MatrixXf P_k_k;

  Eigen::MatrixXf S;

  Eigen::MatrixXf mk;
  Eigen::MatrixXf wk;
  Eigen::MatrixXf Pk;

  Eigen::MatrixXf mk_bar;
  Eigen::MatrixXf wk_bar;
  Eigen::MatrixXf Pk_bar;

  Eigen::MatrixXf mk_bar_fixed;
  Eigen::MatrixXf wk_bar_fixed;
  Eigen::MatrixXf Pk_bar_fixed;

  Eigen::MatrixXf mk_bar_display;
  Eigen::MatrixXf wk_bar_display;
  Eigen::MatrixXf Pk_bar_display;

  Eigen::MatrixXf mk_k_minus_1_beforePrediction;

  Eigen::MatrixXf Z_k;
  Eigen::MatrixXf Z_k_previous;
  Eigen::MatrixXf ang_vel_k;
  Eigen::MatrixXf X_k;
  Eigen::MatrixXf X_k_previous;

  Eigen::MatrixXf B;

  Eigen::MatrixXf Detections;
  Eigen::MatrixXf mahalDistance;

  sensor_msgs::ImagePtr image_msg;

  float cu, cv, f;
  float dt;

  const uint8_t n_state = 4;
  const uint8_t n_meas = 2;
  const uint8_t n_input = 3;

  std::vector<int> occluded_frame_count;  // 记录每列连续低权重帧数（需作为类成员变量）
  const int MIN_OCCLUDED_FRAMES = 3;  // 至少连续3帧权重低才视为空列
  bool is_occlusion_counter_init = false;  // 标记是否已初始化
  std::vector<Tracknew> tracks_; // 滤波器维护的所有轨迹的容器（包含活跃和非活跃的）
  std::vector<Candidate> candidates_;// 存储当前帧的候选跟踪
  std::vector<Candidate> candidates_for_matching;// 存储当前帧的候选跟踪，用于匹配
  private:

  //新增：存储所有有效跟踪
  
  
  int next_track_id_ = 0; // 下一个可用的跟踪ID
  // 辅助函数
  void apply_CA_AKS_Update(Tracknew& tr, const Candidate& cand);
  std::vector<std::pair<int, int>> associate_candidates_greedy(
      const std::vector<int>& track_indices,
      const std::vector<int>& cand_indices,
      float gate_dist,
      bool strict_direction
  );

  int occlusion_counter = 0;//连续遮挡帧数计数器
  const int OCCLUSION_THRESHOLD = 20; // 触发清零的连续遮挡帧数阈值
  std::vector<int> empty_columns; //声明为空列列表
};

