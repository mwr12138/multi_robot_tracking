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



// 添加速度历史记录
    std::vector<std::deque<Eigen::Vector2f>> velocity_history;  // 每个目标的速度历史
    std::vector<std::deque<Eigen::Vector2f>> position_history;  // 每个目标的位置历史
    const int HISTORY_SIZE = 5;  // 保留最近5帧的历史
    void update_velocity_history(); //更新速度历史
    void initialize_velocity_history();
    float calculate_velocity_consistency(int target_id, const Eigen::Vector2f& candidate_velocity); //计算速度连续性得分
    float calculate_position_consistency(int target_id, const Eigen::Vector2f& candidate_position); //计算位置连续性得分
    Eigen::Vector2f predict_position(int target_id);//基于历史位置预测当前位置
    int find_target_using_id(int target_id, const Eigen::MatrixXi& newIndex, 
                            const std::vector<std::pair<float, int>>& weighted_targets, 
                            const std::vector<bool>& id_used);// 查找当前使用指定ID的目标索引
    void assign_target_to_id(int source_idx, int target_id, 
                        const Eigen::MatrixXf& wk_bar_fixed_k,
                        const Eigen::MatrixXf& mk_bar_fixed_k,
                        const Eigen::MatrixXf& Pk_bar_fixed_k);
    void cleanup_memory(); // 清理内存
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

  float dt_cam = 0.125; //8hz
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
 private:
  int occlusion_counter = 0;//连续遮挡帧数计数器
  const int OCCLUSION_THRESHOLD = 5; // 触发清零的连续遮挡帧数阈值
  std::vector<int> empty_columns; //声明为空列列表
};

