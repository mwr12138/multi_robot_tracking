#include <ros/console.h>
#include <tf/transform_datatypes.h>
#include "multi_robot_tracking/PhdFilter.h"
#include <chrono>
#include <deque>

using namespace std;
using namespace std::chrono;

// #define TIMING

PhdFilter::PhdFilter()

{
    
}

void PhdFilter::phd_track()
{
#ifdef TIMING
    auto time_start = high_resolution_clock::now();
#endif TIMING
    startTime = ros::Time::now();
    k_iteration = k_iteration + 1;
    ROS_INFO("iter: %d",k_iteration);

    //kalmanPredict(); // ToDo: Can be asynch ???
#ifdef TIMING
    auto time_endP = high_resolution_clock::now();
    auto durationP = duration_cast<std::chrono::microseconds>(time_endP - time_start);
    ROS_ERROR_STREAM("Time taken by function predict: " << durationP.count() << " microseconds" << endl);
#endif

    phd_construct(); //滤波器构建
#ifdef TIMING
    auto time_endI = high_resolution_clock::now();
    auto durationI = duration_cast<std::chrono::microseconds>(time_endI - time_endP);
    ROS_ERROR_STREAM("Time taken by function Issue: " << durationI.count() << " microseconds" << endl);
#endif

    phd_update(); //更新
#ifdef TIMING
    auto time_endA = high_resolution_clock::now();
    auto durationA = duration_cast<std::chrono::microseconds>(time_endA - time_endI);
    ROS_ERROR_STREAM("Time taken by function Associate: " << durationA.count() << " microseconds" << endl);
#endif

    phd_prune(); //剪枝
#ifdef TIMING
    auto time_endU = high_resolution_clock::now();
    auto durationU = duration_cast<std::chrono::microseconds>(time_endU - time_endA);
    ROS_ERROR_STREAM("Time taken by function Update: " << durationU.count() << " microseconds" << endl);
#endif

    phd_state_extract(); //状态提取
    cleanup_memory();//清理内存
#ifdef TIMING
    auto time_endE = high_resolution_clock::now();
    auto durationE = duration_cast<std::chrono::microseconds>(time_endE - time_endU);
    ROS_ERROR_STREAM("Time taken by function Extract: " << durationE.count() << " microseconds" << endl);

    auto time_end = high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(time_end - time_start);
    ROS_ERROR_STREAM("Time taken by function: " << duration.count() << " microseconds" << endl);
#endif

    startTime = ros::Time::now();
    k_iteration = k_iteration + 1;
    ROS_INFO("iter: %d",k_iteration);
    endTime = ros::Time::now();
    ROS_WARN("end of track iteration");

}



// 初始化历史记录
void PhdFilter::initialize_velocity_history() 
{
    velocity_history.resize(NUM_DRONES);
    position_history.resize(NUM_DRONES);
}

// 更新速度历史
void PhdFilter::update_velocity_history() 
{
    for(int i = 0; i < NUM_DRONES; i++) {
        if(wk_bar_display(i) > 0.3f) {  // 只有权重足够高的目标才记录历史
            Eigen::Vector2f current_vel(mk_bar_display(1,i), mk_bar_display(3,i));
            Eigen::Vector2f current_pos(mk_bar_display(0,i), mk_bar_display(2,i));
            
            velocity_history[i].push_back(current_vel);
            position_history[i].push_back(current_pos);
            // 保持历史长度
            if(velocity_history[i].size() > HISTORY_SIZE) {
                velocity_history[i].pop_front();
            }
            if(position_history[i].size() > HISTORY_SIZE) {
                position_history[i].pop_front();
            }
        } else {
            // 权重低的目标清空历史
            velocity_history[i].clear();
            position_history[i].clear();
        }
    }
    ROS_ERROR_STREAM("velocity_history_ " << velocity_history.size() << endl);
}

// 计算速度连续性得分
float PhdFilter::calculate_velocity_consistency(int target_id, const Eigen::Vector2f& candidate_velocity) 
{
    if(velocity_history[target_id].size() < 2) {
        return 1.0f;  // 历史不足，返回中性得分
    }
    
    float angle_consistency = 0.0f;
    float magnitude_consistency = 0.0f;
    int count = 0;
    
    // 计算与历史速度的相似性
    for(int i = 0; i < velocity_history[target_id].size(); i++) {
        const Eigen::Vector2f& hist_vel = velocity_history[target_id][i];
        
        if(hist_vel.norm() > 0.1f && candidate_velocity.norm() > 0.1f) {
            // 1. 速度方向一致性（夹角余弦）
            float cos_angle = hist_vel.dot(candidate_velocity) / 
                                (hist_vel.norm() * candidate_velocity.norm());
            cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle));  // 确保在[-1,1]范围内
            angle_consistency += (cos_angle + 1.0f) / 2.0f;  // 转换为[0,1]范围
            
            // 2. 速度大小一致性
            float ratio = std::min(hist_vel.norm(), candidate_velocity.norm()) / 
                            std::max(hist_vel.norm(), candidate_velocity.norm());
            magnitude_consistency += ratio;
            
            count++;
        }
    }
    
    if(count == 0) return 0.5f;  // 没有有效历史
    
    angle_consistency /= count;
    magnitude_consistency /= count;
    
    // 综合得分：方向一致性权重更高
    return 0.7f * angle_consistency + 0.3f * magnitude_consistency;
}

// 计算位置连续性得分
float PhdFilter::calculate_position_consistency(int target_id, const Eigen::Vector2f& candidate_position) 
{
    if(position_history[target_id].size() < 2) {
        return 1.0f;  // 历史不足，返回中性得分
    }
    
    // 基于历史位置预测当前位置
    Eigen::Vector2f predicted_position = predict_position(target_id);
    
    // 计算预测位置与候选位置的差异
    float distance = (predicted_position - candidate_position).norm();
    
    // 转换为得分：距离越小，得分越高
    float max_expected_distance = 50.0f;  // 最大预期距离（像素）
    return std::max(0.0f, 1.0f - distance / max_expected_distance);
}

// 基于历史位置预测当前位置
Eigen::Vector2f PhdFilter::predict_position(int target_id) 
{
    if(position_history[target_id].size() < 2) {
        return position_history[target_id].back();  // 无法预测，返回最后位置
    }
    
    // 简单线性预测：使用最后两个位置
    const Eigen::Vector2f& last_pos = position_history[target_id].back();
    const Eigen::Vector2f& second_last_pos = position_history[target_id][position_history[target_id].size()-2];
    
    Eigen::Vector2f velocity = last_pos - second_last_pos;
    return last_pos + velocity;  // 假设匀速运动
}

// 查找当前使用指定ID的目标索引
int PhdFilter::find_target_using_id(int target_id, const Eigen::MatrixXi& newIndex, 
                        const std::vector<std::pair<float, int>>& weighted_targets, 
                        const std::vector<bool>& id_used) 
{
    for(int i = 0; i < weighted_targets.size(); i++) {
        int source_idx = weighted_targets[i].second;
        if(newIndex(source_idx) % NUM_DRONES == target_id) {
            return source_idx;
        }
    }
    return -1;
}

void PhdFilter::assign_target_to_id(int source_idx, int target_id, 
                                const Eigen::MatrixXf& wk_bar_fixed_k,
                                const Eigen::MatrixXf& mk_bar_fixed_k,
                                const Eigen::MatrixXf& Pk_bar_fixed_k) 
{
    wk_bar_fixed.block(0, target_id, 1, 1) = wk_bar_fixed_k.block(0, source_idx, 1, 1);
    mk_bar_fixed.block(0, target_id, n_state, 1) = mk_bar_fixed_k.block(0, source_idx, n_state, 1);
    Pk_bar_fixed.block(0, n_state*target_id, n_state, n_state) = 
        Pk_bar_fixed_k.block(0, n_state*source_idx, n_state, n_state);
        
    wk_bar_display.block(0, target_id, 1, 1) = wk_bar_fixed_k.block(0, source_idx, 1, 1);
    mk_bar_display.block(0, target_id, n_state, 1) = mk_bar_fixed_k.block(0, source_idx, n_state, 1);
    Pk_bar_display.block(0, n_state*target_id, n_state, n_state) = 
        Pk_bar_fixed_k.block(0, n_state*source_idx, n_state, n_state);
}

void PhdFilter::cleanup_memory() 
{
    static int cleanup_counter = 0;
    cleanup_counter++;
    
    // 每10帧执行一次全面清理（更频繁）
    if (cleanup_counter >= 10) {
        cleanup_counter = 0;
        
        for (int i = 0; i < NUM_DRONES; i++) {
            // 条件1：清理无效目标
            if (wk_bar_display(i) < 0.01f) {
                velocity_history[i].clear();
                position_history[i].clear();
            }
            // 条件2：即使目标有效，也限制历史长度
            else if (velocity_history[i].size() > HISTORY_SIZE * 2) {
                // 保留最近的 HISTORY_SIZE 个数据，删除旧的
                while (velocity_history[i].size() > HISTORY_SIZE) {
                    velocity_history[i].pop_front();
                }
                while (position_history[i].size() > HISTORY_SIZE) {
                    position_history[i].pop_front();
                }
                ROS_WARN_STREAM("清理目标 " << i << " 的过量历史数据");
            }
        }
        
        // 强制内存回收
        for (int i = 0; i < NUM_DRONES; i++) {
            velocity_history[i].shrink_to_fit();
            position_history[i].shrink_to_fit();
        }
    }
}  



void PhdFilter::initialize_matrix(float cam_cu, float cam_cv, float cam_f, float meas_dt)  //初始化矩阵
{
    ROS_INFO("first initialize matrix");
    //initialize
    Z_k = Eigen::MatrixXf::Zero(n_meas,NUM_DRONES);
    Z_k_previous = Eigen::MatrixXf::Zero(n_meas,NUM_DRONES);
    ang_vel_k = Eigen::MatrixXf::Zero(n_input,1);

    mk_minus_1 = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    wk_minus_1 = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    Pk_minus_1 = Eigen::MatrixXf::Zero(n_state,NUM_DRONES*4);

    mk = Eigen::MatrixXf::Zero(n_state,NUM_DRONES+NUM_DRONES*NUM_DRONES);
    wk = Eigen::MatrixXf::Zero(1,NUM_DRONES+NUM_DRONES*NUM_DRONES);
    Pk = Eigen::MatrixXf::Zero(n_state,n_state*(NUM_DRONES+NUM_DRONES*NUM_DRONES) );

    mk_bar = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    wk_bar = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    Pk_bar = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);

    mk_bar_fixed = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    wk_bar_fixed = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    Pk_bar_fixed = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);

    mk_bar_display = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    wk_bar_display = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    Pk_bar_display = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);

    mk_k_minus_1 = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    wk_k_minus_1 = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    Pk_k_minus_1 = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);
    P_k_k = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);
    S = Eigen::MatrixXf::Zero(n_meas,n_meas*NUM_DRONES);

    F = Eigen::MatrixXf::Zero(n_state,n_state);
    A = Eigen::MatrixXf::Zero(n_state,n_state);
    H = Eigen::MatrixXf::Zero(n_meas,n_state);
    Q = Eigen::MatrixXf::Zero(n_state,n_state);
    R = Eigen::MatrixXf::Zero(n_meas, n_meas);
    K = Eigen::MatrixXf::Zero(n_state,n_meas*NUM_DRONES);

    X_k = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    X_k_previous = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);

    B = Eigen::MatrixXf::Zero(n_state,n_input*NUM_DRONES);

    mk_k_minus_1_beforePrediction = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    Detections = Eigen::MatrixXf::Zero(4,NUM_DRONES);


    cu = cam_cu; //光心x坐标
    cv = cam_cv; //光心y坐标
    f = cam_f;   //焦距
    dt = 0.01;
    initialize_velocity_history();
}

void PhdFilter::set_num_drones(int num_drones_in)  //设置无人机数量
{
    NUM_DRONES = num_drones_in;
}

void PhdFilter::initialize(float q_pos, float q_vel, float r_meas, float p_pos_init, float p_vel_init,
                        float prune_weight_threshold, float prune_mahalanobis_threshold, float extract_weight_threshold)  //初始化
{
    //ROS_ERROR("======= Initialize ======= \n");
    Eigen::MatrixXf P_k_init;
    P_k_init = Eigen::MatrixXf(n_state,n_state); //根据输入的参数初始化状态协方差矩阵
    P_k_init <<
            p_pos_init, 0,          0,          0,
            0,          p_vel_init, 0,          0,
            0,          0,          p_pos_init, 0,
            0,          0,          0,          p_vel_init;
    P_k_init = P_k_init * 1;


    for(int i = 0; i < Z_k.cols(); i ++)
    {   
        //store Z into mk (x,y)
        // ROS_ERROR_STREAM("ZK: \n" << Z_k << endl); 
        // mk_minus_1.block(0, i, n_state, 1) = H.transpose()*Z_k.block(0,i, n_meas,1);
        mk_minus_1.block(0, i, n_state, 1) << Z_k.block(0, i, 1, 1), 0, Z_k.block(1, i, 1, 1), 0;   //将Z_k的测量值转换为mk_minus_1状态向量
        // ROS_ERROR_STREAM("mk_minus_1: \n" << mk_minus_1 << endl);         
        //store pre-determined weight into wk (from matlab)
        wk_minus_1(i) = .0016;  //给每个目标分配一个初始权重
        //store pre-determined weight into Pk (from paper)
        Pk_minus_1.block(0,i*4, n_state,n_state) = P_k_init; //给每个目标分配一个初始状态协方差矩阵4*4左右排列
    }

    A << 1,dt_imu,0,0,
            0,1,0,0,
            0,0,1,dt_imu,
            0,0,0,1;
    
    //Measurement Matrix  H测量矩阵只提取位置、忽略速度（因为相机只能测位置）
    H << 1, 0, 0 , 0,
         0, 0, 1, 0;

    //Process noise covariance, given in Vo&Ma.  过程噪声协方差矩阵（运动模型的不确定性）
    Q << q_pos,     0,      0,          0,
         0,         q_vel,  0,          0,
         0,         0,      q_pos,      0,
         0,         0,      0,          q_vel;

    //Measurement Noise  R测量噪声协方差矩阵（相机测量的不确定性）
    R << r_meas,    0,
         0,         r_meas;

    numTargets_Jk_minus_1 = NUM_DRONES;
}

/* 
 * Prediction step is done async. The prediction is called whenever we get an IMU message.
 * The update step is called once we get a measurement from the network
*/
void PhdFilter::asynchronous_predict_existing() //异步预测
{
    ROS_INFO("======= 0. asynch predict ======= \n");
    //update A
    update_A_matrix(dt_imu);

    wk_minus_1 = prob_survival * wk_minus_1;   //更改目标权重，这里后续可以根据目标存在的概率来更改
    Eigen::MatrixXf Bu_temp = Eigen::MatrixXf::Zero(n_state,n_meas); //Bu_temp是状态转移矩阵和输入矩阵的乘积
    F = Eigen::MatrixXf(n_state,n_state); //修正后的状态转移矩阵
    float omega_x = ang_vel_k(0);  //X轴角速度
    float omega_y = ang_vel_k(1);  //Y轴角速度
    float omega_z = ang_vel_k(2);  //Z轴角速度
    float pu = 0; //临时变量，目标x位置
    float pv = 0; //临时变量，目标y位置

    Eigen::MatrixXf P_temp; //临时协方差矩阵
    P_temp = Eigen::MatrixXf(n_state,n_state); //4*4

    //ROS_INFO("size mk-1: %lu, size B: %lu",mk_minus_1.cols(), B.cols());
    //ROS_INFO_STREAM("A:\n" << A << endl);
    for (int i = 0; i < mk_minus_1.cols(); i++)
    {
        // 注释：原计划根据加速度调整过程噪声，实际未启用
        // float qAcc = sqrt((mk_k_minus_1(1, i))*(mk_k_minus_1(1, i)) + (mk_k_minus_1(3, i)*(mk_k_minus_1(3, i))));
        // qAcc = ((qAcc*2)+1);
        // ROS_INFO_STREAM("!!!!!!!!!!QACC = " << qAcc);
        Eigen::MatrixXf Q_temp = Q; //过程噪声协方差矩阵
        Q_temp = Q;// * qAcc;
        // ROS_INFO("iteration: %d", i);
        // ROS_INFO_STREAM("B:\n" << B.block(0,n_input*i, n_state,n_input)); 
        Bu_temp = B.block(0,n_input*i, n_state,n_input) * ang_vel_k; //
        // ROS_INFO_STREAM("Bu:\n" << Bu_temp); 
        mk_minus_1.block(0,i, n_state,1) = A * mk_minus_1.block(0,i, n_state,1) + Bu_temp;  
        P_temp = Pk_minus_1.block(0,n_state*i, n_state,n_state);
        pu = mk_minus_1(0,i);
        pv = mk_minus_1(2,i);

        F(0,0) = (dt_imu*omega_y*(2*cu - 2*pu))/f - (dt_imu*omega_x*(cv - pv))/f + 1;      F(0,1) = dt;   F(0,2) = dt_imu*omega_z - (dt_imu*omega_x*(cu - pu))/f;                            F(0,3) = 0;
        F(1,0) = 0;                                                                        F(1,1) = 1;    F(1,2) = 0;                                                                        F(1,3) = 0;
        F(2,0) = (dt_imu*omega_y*(cv - pv))/f - dt_imu*omega_z;                            F(2,1) = 0;    F(2,2) = (dt_imu*omega_y*(cu - pu))/f - (dt_imu*omega_x*(2*cv - 2*pv))/f + 1;      F(2,3) = dt;
        F(3,0) = 0;                                                                        F(3,1) = 0;    F(3,2) = 0;                                                                        F(3,3) = 1;

        P_temp = Q + F* P_temp * F.transpose();  //// 预测协方差：不确定性 = 过程噪声 + F×上一协方差×F^T（传递不确定性）
        Pk_minus_1.block(0,n_state*i, n_state,n_state) = P_temp; //计算出新的协方差矩阵
    }

    //更新预测后的状态、权重、协方差（用于后续更新步骤）
    //X_k = mk_minus_1;  //尝试不更新这个
    wk_k_minus_1 = wk_minus_1;
    mk_k_minus_1 = mk_minus_1;
    Pk_k_minus_1 = Pk_minus_1;
    numTargets_Jk_k_minus_1 = numTargets_Jk_minus_1;

    ROS_INFO_STREAM("WK|K-1:\n" << wk_k_minus_1 << endl);
    ROS_INFO_STREAM("mK|K-1:\n" << mk_k_minus_1 << endl);
    ROS_INFO_STREAM("PK|K-1:\n" << Pk_k_minus_1 << endl);
}


void PhdFilter::phd_construct() //滤波器构建
{
    ROS_INFO("======= 2. construct ======= \n");
    wk_k_minus_1 = wk_minus_1;
    mk_k_minus_1 = mk_minus_1;
    Pk_k_minus_1 = Pk_minus_1;
    numTargets_Jk_k_minus_1 = numTargets_Jk_minus_1;

    Eigen::MatrixXf PHt, HPHt, identity4;  //临时变量：用于计算卡尔曼增益  
    PHt = Eigen::MatrixXf(n_state,n_meas); //4*2
    HPHt = Eigen::MatrixXf(n_meas,n_meas); //2*2

    identity4 = Eigen::MatrixXf::Identity(4,4);

    for(int j = 0; j < numTargets_Jk_k_minus_1; j ++)
    {
        PHt = Pk_k_minus_1.block(0,n_state*j, n_state,n_state) * H.transpose(); // 计算P×H^T（协方差映射到测量空间）
        HPHt = H*PHt; //计算H×P×H^T（预测协方差在测量空间的投影）
        K.block(0,n_meas*j, n_state,n_meas) = PHt * (HPHt + R).inverse(); //计算卡尔曼增益：P×H^T×（H×P×H^T + R）^-1
        S.block(0,n_meas*j, n_meas,n_meas) = HPHt + R;  // 计算创新协方差S：S = H×P×H^T + R（测量空间的总不确定性）
        Eigen::MatrixXf t1 = (identity4 - K.block(0,n_meas*j, n_state,n_meas)*H)*Pk_k_minus_1.block(0,n_state*j, n_state,n_state);
        P_k_k.block(0,n_state*j, n_state,n_state) = t1;  // 计算更新后协方差：P_k_k = (I - K×H) × P_prev
    }
    ROS_INFO_STREAM("m_k_minus_1:\n" << mk_minus_1 << "\n");
    ROS_INFO_STREAM("P_k_k: " << endl << P_k_k << endl);
    ROS_INFO_STREAM("K: " << endl << K << endl);
}

void PhdFilter::phd_update() //更新
{
    // ROS_ERROR("======= 3. update ======= \n");
    
    //1. set up matrix size
    mahalDistance = Eigen::MatrixXf::Zero(1,numTargets_Jk_k_minus_1 * detected_size_k + numTargets_Jk_k_minus_1);
    wk = Eigen::MatrixXf::Zero(1,numTargets_Jk_k_minus_1 * detected_size_k + numTargets_Jk_k_minus_1);
    mk = Eigen::MatrixXf::Zero(n_state,numTargets_Jk_k_minus_1 * detected_size_k + numTargets_Jk_k_minus_1);
    Pk = Eigen::MatrixXf::Zero(n_state, n_state * (numTargets_Jk_k_minus_1 * detected_size_k + numTargets_Jk_k_minus_1));

    // 临时变量：用于计算权重和状态更新
    Eigen::MatrixXf thisZ, meanDelta_pdf, cov_pdf;
    Eigen::MatrixXf w_new, w_new_exponent;
    
    Eigen::MatrixXf associationWeights(detected_size_k, NUM_DRONES); // 关联权重矩阵
    associationWeights.setZero(); 

    thisZ = Eigen::MatrixXf(n_meas,1); //2×1：单个测量值
    meanDelta_pdf = Eigen::MatrixXf(n_meas,1); //2×1：残差（测量值 - 预测值）
    cov_pdf = Eigen::MatrixXf(n_meas,n_meas); //2×2：创新协方差块
    w_new = Eigen::MatrixXf(1,1); //1×1：临时权重
    w_new_exponent = Eigen::MatrixXf(1,1); //1×1：高斯分布的指数部分

    int index = 0;  // 临时索引：候选目标在mk/wk中的位置
    L = 0;  // 测量值计数

    // 第一部分：处理未被检测到的目标（存活但未被观测到）
    for (int i = 0; i < numTargets_Jk_k_minus_1; i++ )
    {
        wk(i) = (1-prob_detection) * wk_k_minus_1(i);  //i的权重 = (1-检测概率0.9) × 上一时刻权重
        mk.block(0,i, n_state,1)  = mk_k_minus_1.block(0,i, n_state,1);  //状态保持之前的预测值不变
        Pk.block(0,n_state*i, n_state,n_state) = Pk_k_minus_1.block(0,n_state*i, n_state,n_state); //协方差保持之前的预测值不变
        mahalDistance(i) = 0;  //马氏距离设为0
    }

    // 第二部分：处理被检测到的目标（融合测量值）
    for (int z = 0; z < detected_size_k; z++)   //对于每个目标检测到的目标进行遍历
    {
        L = L+1;  //测量值计数增加（为了后边计算index索引）

        for (int j = 0; j < numTargets_Jk_k_minus_1; j++)  //遍历设的无人机数量
        {
            thisZ = Z_k.block(0,z, n_meas,1);  //将第z个目标检测中心位置赋值给thisZ

            index = (L) * numTargets_Jk_k_minus_1 + j; //计算索引无人机设定数量为0的时候，0 1 2 3是之前的结果，这里就从4开始，4 5 6 7 然后8 9 10 11以此类推
            ROS_INFO("Index: %d, L: %d\n", index, L);
            //update weight (multivar prob distr)
            meanDelta_pdf = thisZ - H*mk_k_minus_1.block(0,j, n_state,1); //计算残差（测量值 - 预测值）
            ROS_INFO_STREAM("Mean Delta PDF: " << meanDelta_pdf << endl);

            cov_pdf = S.block(0,n_meas*j, n_meas,n_meas); // 获取第j个目标的创新协方差块
            ROS_INFO_STREAM("Cov PDF: " << cov_pdf << endl);
            w_new_exponent = -0.5 * (meanDelta_pdf.transpose() * cov_pdf.inverse() * meanDelta_pdf).transpose() * (meanDelta_pdf.transpose() * cov_pdf.inverse() * meanDelta_pdf) ; //计算马氏距离的平方（高斯分布的指数部分）
            //w_new_exponent = -0.5 * (meanDelta_pdf.transpose() * cov_pdf.inverse() * meanDelta_pdf);  //这里之前应该是写错了
            ROS_INFO_STREAM("W_new_exponent: " << w_new_exponent << endl);

            mahalDistance(index) = fabs(w_new_exponent(0)); //绝对值

            double q_val = wk_k_minus_1(j) * pow(2*PI, -1/2) * pow(cov_pdf.determinant(),-0.5) * exp(w_new_exponent(0,0));  //第j个目标在当前测量下的后验存在概率密度
            ROS_INFO_STREAM("q_val: " << q_val << endl);
            wk(index)= q_val;  //更新目标权重
            associationWeights(z,j) = q_val;  // 记录第z个测量值与第j个目标的关联权重

            double w_numerator = prob_detection * wk_k_minus_1(j) * q_val; //原来是计划用这个权重分子，现在是直接用了q_val,所以这行目前没有起到作用

            //update mean
            mk.block(0,index, n_state,1) = mk_k_minus_1.block(0,j, n_state,1);  //重复
            mk.block(0,index, n_state, 1) =  mk_k_minus_1.block(0,j, n_state,1) + K.block(0,n_meas*j, n_state,n_meas)*meanDelta_pdf;  //卡尔曼滤波更新
            //update cov
            Pk.block(0,n_state*index, n_state,n_state) = P_k_k.block(0,n_state*j, n_state,n_state);

            //            ROS_INFO("wk: %f",wk(index));
            //            ROS_INFO_STREAM << "mk: "<< mk <<  endl;

        }
        ROS_INFO_STREAM("mk(pre): " << endl << mk << endl);
        ROS_INFO_STREAM("pk(pre): " << endl << Pk << endl);
        ROS_INFO_STREAM("wk(pre): " << endl << wk << endl);

        //normalize weights归一化权重
        float weight_tally = 0;
        float old_weight;

        //sum weights计算当前测量对应的所有候选目标的权重总和
        for(int i = 0; i < numTargets_Jk_k_minus_1; i ++)
        {
            index = (L) * numTargets_Jk_k_minus_1 + i;
            weight_tally = weight_tally + wk(index);
        }


        //divide sum weights归一化：权重 = 旧权重 / (杂波强度 + 权重总和)
        for(int i = 0; i < numTargets_Jk_k_minus_1; i ++)
        {
            index = (L) * numTargets_Jk_k_minus_1 + i;
            old_weight = wk(index);
            //float measZx = Z_k(0,i);
            //float measZy = Z_k(1,i);
            wk(index) = old_weight / (clutter_intensity(mk_k_minus_1(0,i),mk_k_minus_1(2,i))/1.0+ weight_tally);  //    原来是X_k改成了mk_k_minus_1
            //wk(index) = old_weight / ( weight_tally);
        }
    }

    // ROS_ERROR_STREAM("++++++++++associationWeights: " << endl << setprecision(3) << associationWeights << endl);
    // ROS_ERROR_STREAM("wk(post_post): " << endl << setprecision(3) << wk << endl);
    // ROS_ERROR_STREAM("mk(post_post): " << endl << setprecision(3) << mk << endl);
    // 新增部分：处理未匹配测量值（放入前四列的空位置）
    // ==============================================
    // 1. 定义参数
    const float birth_weight = 1.0f;        // 新生目标初始权重
    const float birth_pos_var = 1.0f;       // 位置初始不确定性
    const float birth_vel_var = 0.5f;       // 速度初始不确定性
    //const float match_threshold = 0.01f;     // 关联权重低于此值视为未匹配
    const float empty_threshold = 0.05f;    // 权重低于此值视为"空列"（可复用）

    // 2. 找出未匹配的测量值（与所有现有目标关联权重都低）
    //std::vector<int> unmatched_measurements;
    // for (int z = 0; z < detected_size_k; z++) {
    //     bool is_matched = false;
    //     // 检查测量值z与所有目标的关联权重是否有一个超过阈值
    //     for (int j = 0; j < NUM_DRONES; j++) {
    //         // 用“测量z与目标j的关联权重”判断，而非目标j的最终权重
    //         if (associationWeights(z, j) > 0) {
    //             is_matched = true;
    //             break;
    //         }
    //     }
    //     if (!is_matched) {
    //         unmatched_measurements.push_back(z);  // 这才是真正未被使用的检测结果
    //         ROS_INFO("测量值z=%d未匹配任何目标，加入未匹配列表", z);
    //         cout<<"((((((((((((((((((((((()))))))))))))))))))))))"<<endl;
    //     }
    // }
    // 调整：1. 降低阈值，适应低权重；2. 优先匹配“全0列”；3. 基于“最大值”判断匹配
    const float match_threshold = 1e-6;  // 降低阈值（如1e-6），容纳重新出现目标的低权重
    std::vector<int> unmatched_measurements;

    // 先标记“全0列”（被遮挡的目标，需要优先分配测量值）
    std::vector<int> zero_columns;  // 存储全为0的目标列索引
    for (int j = 0; j < NUM_DRONES; j++) {
        bool is_zero_col = true;
        for (int z = 0; z < detected_size_k; z++) {
            if (associationWeights(z, j) > 1e-9) {  // 列中存在非0值
                is_zero_col = false;
                break;
            }
        }
        if (is_zero_col) {
            zero_columns.push_back(j);
            ROS_INFO("目标列%d全为0（被遮挡），标记为待分配", j);
        }
    }

    // 对每个测量值z，判断是否匹配
    for (int z = 0; z < detected_size_k; z++) {
        // 步骤1：找到该测量值z在所有目标中的最大关联权重及对应目标
        float max_weight = 0;
        int best_j = -1;  // 最可能匹配的目标索引
        for (int j = 0; j < NUM_DRONES; j++) {
            if (associationWeights(z, j) > max_weight) {
                max_weight = associationWeights(z, j);
                best_j = j;
            }
        }

        // 步骤2：判断是否匹配（基于“最大权重是否有效”+“是否是全0列的目标”）
        bool is_matched = false;
        if (best_j != -1) {
            // 条件1：最大权重超过阈值，视为有效匹配
            if (max_weight > match_threshold) {
                is_matched = true;
            }
            // 条件2：即使权重低，但如果对应“全0列”（被遮挡目标），也视为匹配（优先分配）
            else if (std::find(zero_columns.begin(), zero_columns.end(), best_j) != zero_columns.end()) {
                is_matched = true;
                
            }
        }

        if (!is_matched) {
            unmatched_measurements.push_back(z);
            ROS_INFO("测量值z=%d未匹配任何目标，加入未匹配列表", z);
        }
    }

    // 3. 找出真正的空列（权重低且历史状态无效）
    //std::vector<int> empty_columns;
    //std::vector<int> occluded_frame_count(NUM_DRONES, 0);  // 记录每列连续低权重帧数（需作为类成员变量）

    // 第一次使用时初始化计数器（确保只执行一次）
    if (!is_occlusion_counter_init) {
        occluded_frame_count.resize(NUM_DRONES, 0);  // 初始化大小为NUM_DRONES，值全为0
        is_occlusion_counter_init = true;  // 标记为已初始化
        ROS_INFO("首次初始化连续遮挡计数器，大小=%d", NUM_DRONES);
    }

    empty_columns.clear();  // 关键：清空上一帧的空列信息
    for (int j = 0; j < NUM_DRONES; j++) {
        // 若当前权重低，累加连续遮挡帧数
        if (wk(j) < empty_threshold) {
            occluded_frame_count[j]++;
            cout<<"lalalalaadd---------------------------------------------------------"<<occluded_frame_count[j]<<endl;
        } else {
            occluded_frame_count[j] = 0;  // 权重恢复，重置计数
        }
        if(occluded_frame_count[j]>50) occluded_frame_count[j] =50;
        // 真正的空列：连续多帧权重低（确保不是瞬间波动）
        if (occluded_frame_count[j] >= MIN_OCCLUDED_FRAMES) {
            empty_columns.push_back(j);
            cout<<"find out column:::::::::::::"<<empty_columns[0]<<endl;
            ROS_INFO("目标列%d连续%d帧权重低，视为空列（权重=%.6f）", j, occluded_frame_count[j], wk(j));
        }
    }


    // 4. 将未匹配测量值放入空列（复用遮挡产生的位置）
    int assign_count = std::min(unmatched_measurements.size(), empty_columns.size());
    for (int i = 0; i < assign_count; i++)
    {
        int z_idx = unmatched_measurements[i];  // 未匹配测量的索引
        int col = empty_columns[i];             // 空列的索引（0-3）

        // 用新生目标信息覆盖空列
        Eigen::VectorXf z_meas = Detections.block(0, z_idx, n_meas, 1);  // 测量值(x,y)
        mk.block(0, col, n_state, 1) << z_meas(0), 0, z_meas(1), 0;  // 初始化状态
        wk(col) = birth_weight;  // 赋予初始权重

        // 初始化协方差（不确定性）
        Eigen::MatrixXf P_birth = Eigen::MatrixXf::Zero(n_state, n_state);
        P_birth << birth_pos_var, 0, 0, 0,
                   0, birth_vel_var, 0, 0,
                   0, 0, birth_pos_var, 0,
                   0, 0, 0, birth_vel_var;
        Pk.block(0, n_state * col, n_state, n_state) = P_birth;

        ROS_ERROR_STREAM("NEW OBJECT!!!" << col << ",MEASUREMENT: " << z_meas.transpose());
    }

}

void PhdFilter::phd_prune() //剪枝
{

    //ROS_ERROR_STREAM("======= 4.Pruning =======");
    Eigen::MatrixXi::Index maxRow, maxCol;  // 临时索引
    Eigen::MatrixXi I, I_copy;  // 保留的目标索引
    Eigen::MatrixXf I_weights;  // 保留的目标权重

    I = Eigen::MatrixXi(1, 0);
    I_copy = Eigen::MatrixXi(1, 0);
    I_weights = Eigen::MatrixXf(1, 0);

    int I_counter = 0;  // 保留的目标计数
    float weight_threshold = 0.1;  // 权重阈值（来自配置）
    float mahalanobis_threshold = 4.0;  // 马氏距离阈值（来自配置）
    int l = 0;  // 合并迭代计数

    // 2. 第一步：修剪低权重目标（保留高权重+新生目标）
   for(int i=0; i<wk.cols(); i++)
    {
        if(wk(i) > weight_threshold)
        {
            I_counter += 1;
            I.conservativeResize(1, I_counter);  // 增加索引矩阵列数为保留目标的个数
            I_weights.conservativeResize(1, I_counter);
            I(0, I_counter-1) = i;
            I_weights(0, I_counter-1) = wk(i);
        }
    }
    I_copy = I;
    //ROS_ERROR_STREAM("wk is:\n" << setprecision(3) << wk << endl);
    //ROS_ERROR_STREAM("I is:\n" << I << "\nI_weights is:\n" << I_weights << endl);

    ROS_INFO_STREAM("I is:\n" << I << "\nWK is:\n" << wk);
    
    // 初始化修剪后的矩阵（存储合并后的目标）
    Eigen::MatrixXf wk_bar_fixed_k = Eigen::MatrixXf::Zero(1, 0);  // 合并后的权重
    Eigen::MatrixXf mk_bar_fixed_k = Eigen::MatrixXf::Zero(4, 0);  // 合并后的状态
    Eigen::MatrixXf Pk_bar_fixed_k = Eigen::MatrixXf::Zero(4, 0);  // 合并后的状态协方差
    Eigen::MatrixXi index_order = Eigen::MatrixXi::Zero(1, 0); // 合并顺序

    Eigen::MatrixXf highWeights = Eigen::MatrixXf::Zero(1, I.cols());

    Eigen::MatrixXi L = Eigen::MatrixXi::Zero(1, I.cols());

    // 第二步：合并相似目标（马氏距离<阈值）这里还需要考虑新生目标
    while(I.cols() != 0)
    {
        l++;
        ROS_INFO_STREAM("Current l: " << l);
        if(I_copy.cols() == 0) // 无目标可处理，退出循环
            break;

        if(l > NUM_DRONES * n_meas)  // 防止无限循环，限制最大合并次数
            break;
        
        // 提取保留目标的权重
        Eigen::MatrixXf highWeights = Eigen::MatrixXf::Zero(1, I.cols());
        for(int i=0; i<I_copy.cols(); i++)
        {
            highWeights(i) = wk(I_copy(i));
        }
        ROS_INFO_STREAM("HighWeights is: " << highWeights);

        // 找到权重最大的目标作为合并中心
        float maxW = 0.0;
        int maxW_index = -1;
        for(int i=0; i<highWeights.cols(); i++)
        {
            if(maxW < highWeights(i))
            {
                maxW = highWeights(i);
                maxW_index = i;
            }
        }
        ROS_INFO_STREAM("maxW is: " << maxW << " and its index is " << maxW_index);

        int j = I_copy(maxW_index); // 权重最大的目标的索引
        
        //增加了检查合并中心是否是新生目标，若是则不合并
        

        
        L = Eigen::MatrixXi::Zero(1, 0); // 存储与j相似的目标索引
        
        // 计算所有保留目标与j的马氏距离
        for(int i=0; i<I.cols(); i++)
        {
            int thisI = I(i);  // 当前目标索引
            Eigen::MatrixXf deltaM = mk.block(0,thisI,n_state,1) - mk.block(0,j,n_state,1); // 状态差：当前目标状态 - 中心目标状态
            float mahalanobis_distance = (deltaM.transpose() * Pk.block(0,n_state*thisI,n_state,n_state).inverse() * deltaM)(0);  // 马氏距离：(状态差)^T × (协方差逆) × 状态差（考虑不确定性的距离）
            ROS_INFO_STREAM("index for mahalanobis distance: " << i << " in wk " << thisI << " distance: " << mahalanobis_distance);

            if(mahalanobis_distance < mahalanobis_threshold)  // 马氏距离小于阈值，就视为相似目标
            {
                L.conservativeResize(1, L.cols()+1);
                L(L.cols()-1) = thisI;  //// 记录相似目标索引
                if(thisI-j == 0)
                {
                    //Need to do something here, seems unimportant
                }
            }
        }
        ROS_INFO_STREAM("L is: " << L);

        // 合并相似目标：计算合并后的权重、状态、协方差
        float w_bar_k_l = 0; // Sum of all associated weights// 合并后的总权重（权重之和）
        for(int i=0; i<L.cols(); i++)
        {
            int thisI = L(i);
            w_bar_k_l += wk(thisI);
        }
        ROS_INFO_STREAM("w_bar_k_l is: " << w_bar_k_l);

        Eigen::MatrixXf m_bar_k_l = Eigen::MatrixXf::Zero(n_state,1); // 合并后的状态：权重加权平均
        for(int i=0; i<L.cols(); i++)
        {
            int thisI = L(i);
            m_bar_k_l += (wk(thisI) * mk.block(0, thisI, n_state, 1)) / w_bar_k_l;
        }
        ROS_INFO_STREAM("m_bar_k_l is: " << m_bar_k_l);

        Eigen::MatrixXf pVal = Eigen::MatrixXf::Zero(n_state, n_state);  // 合并后的协方差：权重加权平均 + 状态偏差
        for(int i=0; i<L.cols(); i++)
        {
            int thisI = L(i);
            Eigen::MatrixXf deltaM = m_bar_k_l.block(0, 0, n_state, 1) - mk.block(0,j,n_state,1);
            pVal += wk(thisI) * (Pk.block(0, n_state*thisI, n_state, n_state));
        }
        pVal = pVal/w_bar_k_l;
        ROS_INFO_STREAM("pVal is: " << pVal);

        // 记录合并顺序和结果
        Eigen::MatrixXf oldP = Pk.block(0, n_state*j, n_state, n_state);
        index_order.conservativeResize(1, index_order.cols()+1);
        index_order(index_order.cols()-1) = j;
        //ROS_ERROR_STREAM("index order: " << index_order << endl);
        //ROS_ERROR_STREAM("L: " << L << endl);
        // Remove all indices in L from I
        // 从I中移除合并后的目标
        for(int i=0; i<L.cols(); i++)
        {
            ROS_INFO_STREAM("i in L.cols: " << i << endl);
            int thisI = L(i);
            for(int ii=0; ii<I.cols(); ii++)
            {
                ROS_INFO_STREAM("ii in I.cols: " << ii << endl);
                if(thisI == I(ii))
                {   
                    ROS_INFO_STREAM("Removing ThisI: " << thisI << " at index: " << ii);
                    removeColumni(I, ii);
                    removeColumni(I_copy, ii);
                    //ROS_ERROR_STREAM("I is:\n" << I << endl);
                }
            }
        }

        // 移除重复的无人机索引（避免同一无人机被多次跟踪）
        // int deleteDroneIndex = j%NUM_DRONES;
        // for(int i=0; i<I_copy.cols(); i++)
        // {
        //     if(I_copy(i) % numTargets_Jk_k_minus_1 == deleteDroneIndex)
        //     {
        //         ROS_INFO_STREAM("Removing indices for drone: " << deleteDroneIndex << " at index: " << i);
        //         removeColumni(I_copy, i);
        //         removeColumni(I, i);
        //     } 
        // }
        ROS_INFO_STREAM("New I: " << I);
        ROS_INFO_STREAM("New I_copy: " << I_copy);

        wk_bar_fixed_k.conservativeResize(1, wk_bar_fixed_k.cols()+1);
        wk_bar_fixed_k(wk_bar_fixed_k.cols()-1) = w_bar_k_l;
        ROS_INFO_STREAM("wk_bar_fixed_k:\n" << wk_bar_fixed_k);


        mk_bar_fixed_k.conservativeResize(n_state, mk_bar_fixed_k.cols()+1);
        mk_bar_fixed_k.block(0, mk_bar_fixed_k.cols()-1, n_state, 1) = m_bar_k_l;
        ROS_INFO_STREAM("mk_bar_fixed_k:\n" << mk_bar_fixed_k);

        Pk_bar_fixed_k.conservativeResize(n_state, Pk_bar_fixed_k.cols()+n_state);
        Pk_bar_fixed_k.block(0, Pk_bar_fixed_k.cols()-n_state,n_state,n_state) = pVal;
        ROS_INFO_STREAM("Pk_bar_fixed_k:\n" << Pk_bar_fixed_k);
    }
    ROS_ERROR_STREAM("wk_bar_fixed_k:\n" << wk_bar_fixed_k);
    ROS_ERROR_STREAM("wk_bar_fixed_k number: " << wk_bar_fixed_k.cols());
    // ROS_ERROR_STREAM("wk_bar_fixed_k is:\n" << wk_bar_fixed_k << "\n");
    // ROS_ERROR_STREAM("mk_bar_fixed_k is:\n" << mk_bar_fixed_k << "\n");
    // ROS_INFO_STREAM("Pk_bar_fixed_k is:\n" << Pk_bar_fixed_k << "\n");
 
    // 调整索引
    Eigen::MatrixXi newIndex = index_order;
    for(int i=0; i<newIndex.cols(); i++)
    {
        newIndex(i) = newIndex(i)%numTargets_Jk_k_minus_1; //整除取余
        //cout<<"newIndex(i)==================================="<<newIndex(i)<<endl;
    }

    for(int i=0; i<newIndex.cols(); i++)
    {
        if(newIndex(i) == 0)
        {
            //Do nothing here, Matlab uses indices as 1, 2, 3 and cpp uses 0, 1, 2 !
            newIndex(i) == newIndex(i);
        }
    }

    // 确保索引不超过无人机数量
    for(int i=0; i<newIndex.cols(); i++)
    {
        if(newIndex(i) >= NUM_DRONES)
        {
            newIndex(i) = newIndex(i) % NUM_DRONES;
        }
    }
    cout<<"newIndex.cols()======="<<newIndex.cols()<<endl;
    // 最终修剪结果存入mk_bar_fixed等矩阵（用于状态提取）
    // mk_bar_fixed = Eigen::MatrixXf::Zero(n_state,NUM_DRONES);
    // wk_bar_fixed = Eigen::MatrixXf::Zero(1,NUM_DRONES);
    // Pk_bar_fixed = Eigen::MatrixXf::Zero(n_state,n_state*NUM_DRONES);
    // 步骤1：更新连续遮挡计数器
    if (detected_size_k < NUM_DRONES) {
        // 检测框数量少于设定值，累加连续遮挡帧数
        occlusion_counter++;
     ROS_INFO("当前检测框数量=%d < 设定值=%d，连续遮挡帧数=%d", 
              detected_size_k, NUM_DRONES, occlusion_counter);
    } else {
     // 检测框数量足够，重置计数器（遮挡解除）
     occlusion_counter = 0;
      ROS_INFO("检测框数量恢复，连续遮挡计数器重置为0");
    }

    // 步骤2：仅当连续遮挡达到5帧时，才执行清零
    if (occlusion_counter >= OCCLUSION_THRESHOLD) {
        mk_bar_fixed.setZero();  // 清零状态矩阵
        wk_bar_fixed.setZero();  // 清零权重矩阵
        Pk_bar_fixed.setZero();  // 清零协方差矩阵
        ROS_INFO("连续%d帧遮挡，执行清零操作", OCCLUSION_THRESHOLD);
    } else {
        // 未达到阈值，保留原有状态（不清零）
        // 若需要更新有效目标的状态，可在此处添加部分更新逻辑
        ROS_INFO("连续遮挡帧数不足%d，不执行清零", OCCLUSION_THRESHOLD);
    }

    
    mk_bar_display = Eigen::MatrixXf::Constant(n_state, NUM_DRONES, -1);
    wk_bar_display = Eigen::MatrixXf::Constant(1, NUM_DRONES, -1);
    Pk_bar_display = Eigen::MatrixXf::Constant(n_state, n_state*NUM_DRONES, -1);
    
    // for(int i=0; i<newIndex.cols(); i++)
    // {
    //     if(i > NUM_DRONES)
    //     {
    //         continue;
    //     }
    //     //cout<<"hahahaha"<<endl;
    //     int sortedIndex = newIndex(i);    //按新索引排序
    //     wk_bar_fixed.block(0, sortedIndex, 1, 1) = wk_bar_fixed_k.block(0, i, 1, 1);
    //     mk_bar_fixed.block(0, sortedIndex, n_state, 1) = mk_bar_fixed_k.block(0, i, n_state, 1);
    //     Pk_bar_fixed.block(0, n_state*sortedIndex, n_state, n_state) = Pk_bar_fixed_k.block(0, n_state*i, n_state, n_state);

    //     //这里我新定义了一组变量，用于显示，这组变量每次都会初始化成0
    //     wk_bar_display.block(0, sortedIndex, 1, 1) = wk_bar_fixed_k.block(0, i, 1, 1);
    //     mk_bar_display.block(0, sortedIndex, n_state, 1) = mk_bar_fixed_k.block(0, i, n_state, 1);
    //     Pk_bar_display.block(0, n_state*sortedIndex, n_state, n_state) = Pk_bar_fixed_k.block(0, n_state*i, n_state, n_state);
    // }
    // // ROS_ERROR_STREAM("wk_bar_fixed is:\n" << wk_bar_fixed << "\n");
    // // ROS_ERROR_STREAM("mk_bar_fixed is:\n" << mk_bar_fixed << "\n");
    // // ROS_INFO_STREAM("Pk_bar_fixed is:\n" << Pk_bar_fixed << "\n");
    // // ROS_ERROR_STREAM("wk_bar_display is:\n" << wk_bar_display << "\n");
    // // ROS_ERROR_STREAM("mk_bar_display is:\n" << mk_bar_display << "\n");
    // // ROS_INFO_STREAM("Pk_bar_display is:\n" << Pk_bar_display << "\n");

    // numTargets_Jk_minus_1 = wk_bar_fixed.cols();  // 更新目标数量
    // ROS_INFO_STREAM("numTargets_Jk_minus_1: " << numTargets_Jk_minus_1);

    // === 简化的智能索引分配逻辑 ===

    // 1. 创建权重-索引对，用于排序
    // === 使用速度连续性的智能索引分配逻辑 ===

    // 1. 创建权重-索引对，用于排序
    std::vector<std::pair<float, int>> weighted_targets;
    for(int i = 0; i < wk_bar_fixed_k.cols(); i++) {
        weighted_targets.push_back(std::make_pair(wk_bar_fixed_k(i), i));
    }

    // 按权重降序排序
    std::sort(weighted_targets.begin(), weighted_targets.end(), 
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return a.first > b.first;
            });
    ROS_ERROR_STREAM("===  weighted_targets ===");
    for(size_t i = 0; i < weighted_targets.size(); i++) {
        ROS_ERROR_STREAM("ID" << i << " weight=" << weighted_targets[i].first 
                        << ", original id=" << weighted_targets[i].second);

    }

    // 2. 记录ID使用状态和最终分配结果
    std::vector<bool> id_used(NUM_DRONES, false);  // 记录每个ID是否已被使用
    std::vector<int> final_assignment(weighted_targets.size(), -1); // 记录每个目标分配的最终ID
    int assigned_count = 0;

    // 3. 第一轮分配：按权重优先级，使用首选ID
    for(int i = 0; i < weighted_targets.size() && assigned_count < NUM_DRONES; i++) {
        int source_idx = weighted_targets[i].second;
        int preferred_id = newIndex(source_idx) % NUM_DRONES;
        
        // 确保索引在有效范围内
        if(preferred_id < 0) preferred_id = 0;
        if(preferred_id >= NUM_DRONES) preferred_id = NUM_DRONES - 1;
        
        if(!id_used[preferred_id]) {
            // 首选ID可用，直接分配
            final_assignment[source_idx] = preferred_id;
            id_used[preferred_id] = true;
            assigned_count++;
            ROS_INFO_STREAM("目标 " << source_idx << " (权重=" << weighted_targets[i].first 
                        << ") 分配到首选ID " << preferred_id);
        } else {
            // 索引冲突！使用速度连续性辅助判断
            ROS_ERROR_STREAM("检测到索引冲突: ID " << preferred_id << " 已被占用");
            
            // 获取候选目标的状态
            Eigen::Vector2f candidate_vel(mk_bar_fixed_k(1, source_idx), mk_bar_fixed_k(3, source_idx));
            Eigen::Vector2f candidate_pos(mk_bar_fixed_k(0, source_idx), mk_bar_fixed_k(2, source_idx));
            
            // 计算候选目标的运动连续性得分
            float candidate_velocity_score = calculate_velocity_consistency(preferred_id, candidate_vel);
            float candidate_position_score = calculate_position_consistency(preferred_id, candidate_pos);
            float candidate_motion_score = 0.7f * candidate_velocity_score + 0.3f * candidate_position_score;
            
            // 查找当前占用该ID的目标
            int current_occupier = -1;
            for(int j = 0; j < i; j++) {
                int other_idx = weighted_targets[j].second;
                if(final_assignment[other_idx] == preferred_id) {
                    current_occupier = other_idx;
                    break;
                }
            }
            
            if(current_occupier != -1) {
                // 计算当前占用者的运动连续性得分
                Eigen::Vector2f current_vel(mk_bar_fixed_k(1, current_occupier), mk_bar_fixed_k(3, current_occupier));
                Eigen::Vector2f current_pos(mk_bar_fixed_k(0, current_occupier), mk_bar_fixed_k(2, current_occupier));
                
                float current_velocity_score = calculate_velocity_consistency(preferred_id, current_vel);
                float current_position_score = calculate_position_consistency(preferred_id, current_pos);
                float current_motion_score = 0.7f * current_velocity_score + 0.3f * current_position_score;
                
                ROS_INFO_STREAM("冲突分析:");
                ROS_INFO_STREAM("  - 候选目标 " << source_idx << ": 速度连续性=" << candidate_velocity_score 
                            << ", 位置连续性=" << candidate_position_score << ", 综合运动得分=" << candidate_motion_score);
                ROS_INFO_STREAM("  - 当前占用者 " << current_occupier << ": 速度连续性=" << current_velocity_score 
                            << ", 位置连续性=" << current_position_score << ", 综合运动得分=" << current_motion_score);
                
                // 如果候选目标明显更适合这个ID，进行替换
                float replacement_threshold = 0.15f; // 替换阈值，可调整
                if(candidate_motion_score > current_motion_score + replacement_threshold) {
                    ROS_INFO_STREAM("替换决策: 候选目标更适合ID " << preferred_id << " (得分差异: " 
                                << (candidate_motion_score - current_motion_score) << ")");
                    
                    // 为当前占用者寻找新的ID
                    bool reassigned = false;
                    for(int new_id = 0; new_id < NUM_DRONES && !reassigned; new_id++) {
                        if(!id_used[new_id]) {
                            final_assignment[current_occupier] = new_id;
                            id_used[new_id] = true;
                            reassigned = true;
                            ROS_INFO_STREAM("重新分配目标 " << current_occupier << " 到 ID " << new_id);
                        }
                    }
                    
                    if(reassigned) {
                        // 分配候选目标到首选ID
                        final_assignment[source_idx] = preferred_id;
                        // id_used[preferred_id] 保持true
                        assigned_count++;
                        ROS_INFO_STREAM("候选目标 " << source_idx << " 分配到 ID " << preferred_id);
                    } else {
                        ROS_ERROR_STREAM("无法重新分配当前占用者，候选目标 " << source_idx << " 将寻找其他ID");
                        // 继续处理，为候选目标寻找其他ID
                    }
                } else {
                    ROS_INFO_STREAM("保留决策: 当前占用者更适合ID " << preferred_id << " (得分差异: " 
                                << (current_motion_score - candidate_motion_score) << ")");
                }
            }
            
            // 如果候选目标还没有被分配（无论是替换失败还是保留决策），为其寻找其他ID
            if(final_assignment[source_idx] == -1) {
                // 使用运动连续性来寻找最合适的替代ID
                int best_alt_id = -1;
                float best_motion_score = -1.0f;
                
                for(int alt_id = 0; alt_id < NUM_DRONES; alt_id++) {
                    if(!id_used[alt_id]) {
                        // 计算在这个ID上的运动连续性
                        float alt_velocity_score = calculate_velocity_consistency(alt_id, candidate_vel);
                        float alt_position_score = calculate_position_consistency(alt_id, candidate_pos);
                        float alt_motion_score = 0.7f * alt_velocity_score + 0.3f * alt_position_score;
                        
                        if(alt_motion_score > best_motion_score) {
                            best_motion_score = alt_motion_score;
                            best_alt_id = alt_id;
                        }
                    }
                }
                
                if(best_alt_id != -1) {
                    final_assignment[source_idx] = best_alt_id;
                    id_used[best_alt_id] = true;
                    assigned_count++;
                    ROS_INFO_STREAM("候选目标 " << source_idx << " 基于运动连续性分配到替代ID " << best_alt_id 
                                << " (运动得分: " << best_motion_score << ")");
                } else {
                    ROS_WARN_STREAM("无法为目标 " << source_idx << " 找到可用ID");
                }
            }
        }
    }

    // 4. 应用最终分配结果到输出矩阵
    for(int i = 0; i < weighted_targets.size(); i++) {
        int source_idx = weighted_targets[i].second;
        if(final_assignment[source_idx] != -1) {
            int target_id = final_assignment[source_idx];
            assign_target_to_id(source_idx, target_id, wk_bar_fixed_k, mk_bar_fixed_k, Pk_bar_fixed_k);
        }
    }

    ROS_INFO_STREAM("基于速度连续性的智能分配完成: " << assigned_count << " 个目标被分配");

    // 更新速度历史
    update_velocity_history();



    //id赋值
    id_consensus.resize(1, NUM_DRONES);
    for(int i = 0; i < NUM_DRONES; i++) {
        id_consensus(i) = -1;  // 初始化为-1
    }
    
    for(int i = 0; i < weighted_targets.size(); i++) {
        int source_idx = weighted_targets[i].second;
        if(final_assignment[source_idx] != -1) {
            int target_id = final_assignment[source_idx];
            assign_target_to_id(source_idx, target_id, wk_bar_fixed_k, mk_bar_fixed_k, Pk_bar_fixed_k);
            
            // 更新id_consensus
            if(target_id < NUM_DRONES) {
                id_consensus(target_id) = target_id;
            }
        }
    }
    
    // 打印调试信息
    ROS_INFO_STREAM("PHD内部ID分配结果: " << id_consensus);

}



void PhdFilter::phd_state_extract() //状态提取
{

    ROS_INFO("============ 5. extract ============= ");
    Eigen::MatrixXf velocity, position;
    velocity = Eigen::MatrixXf(2,1);
    position = Eigen::MatrixXf(2,1);
    float gain_fine_tuned = 1.0;  // 速度计算增益（微调）
    float weight_threshold_for_extraction = 0.5;  // 提取的阈值

    ROS_INFO_STREAM("DT Cam: " << dt_cam << "\n");
    //update state for next iterations
    
    // X_k = mk_minus_1;
    if(k_iteration > 3)   // 迭代次数>3时（初始化完成），更新状态
    {
        //update state for next iterations
        // wk_minus_1 = wk_bar_fixed;
        // mk_minus_1 = mk_bar_fixed;
        // Pk_minus_1 = Pk_bar_fixed.cwiseAbs();

        // X_k = mk_minus_1;
        // cout << "--- X_k: " << endl << X_k << endl;
        for(int i=0; i<wk_bar_display.cols(); i++)
        {
            if(wk_bar_display(i)==-1)
            {
                X_k.block(0, i, n_state, 1).setConstant(-1);
                //X_k.block(0, i, n_state, 1) = mk_bar_display.block(0, i, n_state, 1);
                wk_minus_1.block(0, i, 1, 1) = wk_bar_fixed.block(0, i, 1, 1);
                mk_minus_1.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
                Pk_minus_1.block(0, n_state*i, n_state, n_state) = Pk_bar_fixed.block(0, n_state*i, n_state, n_state).cwiseAbs();
                cout<<"xx"<<endl;
            }
            else if(wk_bar_display(i) < weight_threshold_for_extraction  && wk_bar_display(i) != -1)  // 权重低于阈值（不可靠）的时候，状态就沿用预测值
            {
                ROS_INFO("!!11");
                X_k.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
                mk_bar_fixed.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
                wk_bar_fixed(i) = wk_minus_1(i);
                Pk_bar_fixed.block(0, n_state*i, n_state, n_state) = Pk_minus_1.block(0, n_state*i, n_state, n_state).cwiseAbs();
            }
            else  // 权重高于阈值（可靠）的时候，状态就用当前的修建后的结果
            {
                ROS_INFO("!!22");
                X_k.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
                ROS_INFO("!!aa");
                wk_minus_1.block(0, i, 1, 1) = wk_bar_fixed.block(0, i, 1, 1);
                ROS_INFO("!!bb");
                mk_minus_1.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
                ROS_INFO("!!cc");
                Pk_minus_1.block(0, n_state*i, n_state, n_state) = Pk_bar_fixed.block(0, n_state*i, n_state, n_state).cwiseAbs();
                ROS_INFO("!!33");
            }
            
        }
        //ROS_ERROR_STREAM("X_k is:\n" << X_k << "\n");
        // for(int i=0; i<wk_bar_fixed.cols(); i++)
        // {
        //     if(wk_bar_fixed(i) < weight_threshold_for_extraction)  // 权重低于阈值（不可靠）的时候，状态就沿用预测值
        //     {
        //         ROS_INFO("!!11");
        //         X_k.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
        //         mk_bar_fixed.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
        //         wk_bar_fixed(i) = wk_minus_1(i);
        //         Pk_bar_fixed.block(0, n_state*i, n_state, n_state) = Pk_minus_1.block(0, n_state*i, n_state, n_state).cwiseAbs();
        //     }
        //     else  // 权重高于阈值（可靠）的时候，状态就用当前的修建后的结果
        //     {
        //         ROS_INFO("!!22");
        //         X_k.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
        //         ROS_INFO("!!aa");
        //         wk_minus_1.block(0, i, 1, 1) = wk_bar_fixed.block(0, i, 1, 1);
        //         ROS_INFO("!!bb");
        //         mk_minus_1.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
        //         ROS_INFO("!!cc");
        //         Pk_minus_1.block(0, n_state*i, n_state, n_state) = Pk_bar_fixed.block(0, n_state*i, n_state, n_state).cwiseAbs();
        //         ROS_INFO("!!33");
        //     }
        // }
        ROS_INFO_STREAM("mK in extract: \n" << mk_minus_1);
        ROS_INFO_STREAM("wK in extract: \n" << wk_bar_fixed);
        ROS_INFO_STREAM("PK in extract: \n" << Pk_bar_fixed);
        ROS_INFO_STREAM("XK in extract: \n" << X_k);
    }
    else  //迭代次数≤3（初始化阶段），直接采用修剪结果
    {
        wk_minus_1 = wk_bar_fixed;
        mk_minus_1 = mk_bar_fixed;
        Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
        X_k = mk_minus_1;
    }
    if (k_iteration > 3)
    {
        for (int i = 0; i < wk_bar_fixed.cols(); i++)
        {
            position.block<1, 1>(0, 0) = (X_k.block<1,1>(0,i) - X_k_previous.block<1,1>(0,i));
            position.block<1, 1>(1, 0) = (X_k.block<1,1>(2,i) - X_k_previous.block<1,1>(2,i));
            velocity = position/ (dt_cam*gain_fine_tuned) ;
            mk_minus_1.block<1,1>(1,i) = velocity.block<1,1>(0,0);
            mk_minus_1.block<1,1>(3,i) = velocity.block<1,1>(1,0);
            ROS_INFO_STREAM("--- position: " << endl << position << endl);
            ROS_INFO_STREAM("--- dt_cam: " << endl << dt_cam << endl);
            ROS_INFO_STREAM("--- dt: " << endl << dt << endl);
            ROS_INFO_STREAM("--- velocity: " << endl << velocity << endl);
        }

    }
    wk_minus_1 = wk_bar_fixed;
    mk_minus_1 = mk_bar_fixed;
    Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
    X_k_previous = X_k;
    
}


float PhdFilter::clutter_intensity(const float ZmeasureX, const float ZmeasureY) 
{
    float xMin = 0;
    float xMax = 224;
    float yMin = 0;
    float yMax = 224;
    float uniform_dist = 0;
    float clutter_intensity = 0;
    float lambda = 12.5*pow(10,-8);
    float volume = 4*pow(10,6);

    if(ZmeasureX < xMin) return 0;
    else if (ZmeasureX > xMax) return 0;
    else if (ZmeasureY < yMin) return 0;
    else if (ZmeasureY > yMax) return 0;
    else
    {
        uniform_dist = 1 / ( (xMax - xMin)*(yMax - yMin) ); // Convert this to a constant initialized at startup

    }

    clutter_intensity = lambda * volume * uniform_dist;

    return clutter_intensity;
}

void PhdFilter::removeColumn(Eigen::MatrixXd& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void PhdFilter::removeColumnf(Eigen::MatrixXf& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void PhdFilter::removeColumni(Eigen::MatrixXi& matrix, unsigned int colToRemove)
{
    unsigned int numRows = matrix.rows();
    unsigned int numCols = matrix.cols()-1;

    if( colToRemove < numCols )
        matrix.block(0,colToRemove,numRows,numCols-colToRemove) = matrix.block(0,colToRemove+1,numRows,numCols-colToRemove);

    matrix.conservativeResize(numRows,numCols);
}

void PhdFilter::update_A_matrix(float input_dt)
{
    A << 1,input_dt,0,0,
            0,1,0,0,
            0,0,1,input_dt,
            0,0,0,1;
}

void PhdFilter::update_F_matrix(float input_dt)
{
    ROS_INFO_STREAM("This should not happen !\n");
    return;
}