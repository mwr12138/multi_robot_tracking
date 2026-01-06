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
static const float ASSOC_COST_TH = 40.5f;

/*cost =
    0.6 * pos_err +      当前位置误差
    0.3 * pred_err +     预测误差
    0.1 * (1 - vel_consistency);   0完全一致  1完全不一致12138
*/
static const float ASSOC_DIST_TH = 10.0f;   // 距离阈值，后面可以调
static const int MAX_MISSED = 10;           // 连续没匹配的最大帧数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
//新增：更新tracks_轨迹的更新应该只出现在这一步
void PhdFilter::updateTracks(const std::vector<Candidate>& candidates) 

{
    ROS_ERROR_STREAM("1111111111111111111111111111111111111111111111111");
    ROS_ERROR_STREAM("Number of candidates: " << candidates.size());
    for(int k = 0; k < candidates.size(); k++)
    {
        ROS_ERROR_STREAM("candidates[" << k << "].x is:\n" << candidates[k].x << "\n");
    }
    //ROS_ERROR_STREAM("candidates[0].x is:\n" << candidates[0].x << "\n");
    ROS_ERROR_STREAM("Number of existing tracks: " << tracks_.size());

    // 记录每个 candidate 是否已被使用
    std::vector<bool> candidate_used(candidates.size(), false);
    std::vector<bool> track_used(tracks_.size(), false);

    // ===== 1. 尝试用 candidate 更新已有 Track =====
    for (size_t i = 0; i < tracks_.size(); ++i) {

        auto& tr = tracks_[i];

        if (!tr.active)
            continue;

        if (tr.missed_count > OCCLUSION_THRESHOLD)
            continue;   //  关键：失去 ID 保护权

        float best_cost = 1e9f;
        int best_idx = -1;

        for (int j = 0; j < candidates.size(); ++j) {
            if (candidate_used[j])
                continue;

            Eigen::Vector2f cand_pos;
            cand_pos << candidates[j].x(0), candidates[j].x(2);

            Eigen::Vector2f pred_pos = predict_position(tr);

            float pos_err = (cand_pos - Eigen::Vector2f(tr.x(0), tr.x(2))).norm();

            float pred_err = (cand_pos - pred_pos).norm();

            Eigen::Vector2f cand_vel = Eigen::Vector2f::Zero();
            if (!tr.position_history.empty()) {
                cand_vel = (cand_pos - tr.position_history.back()) / dt_cam;
            }

            float vel_consistency = calculate_velocity_consistency(tr, cand_vel);

            float cost =    0.6f * pos_err +
                            0.3f * pred_err +
                            0.1f * (1.0f - vel_consistency);

            if (cost < best_cost) {
                best_cost = cost;
                best_idx = j;
            }
        }
        ROS_ERROR_STREAM("Track " << tr.id << " best_idx=" << best_idx << " best_cost=" << best_cost);

        if (best_idx >= 0 && best_cost < ASSOC_COST_TH) {

            Eigen::Vector2f pos;
            pos << candidates[best_idx].x(0), candidates[best_idx].x(2);

            if (!tr.position_history.empty()) {
                Eigen::Vector2f vel = (pos - tr.position_history.back()) / dt_cam;

                tr.velocity_history.push_back(vel);
                if (tr.velocity_history.size() > HISTORY_SIZE)
                    tr.velocity_history.pop_front();
            }

            tr.position_history.push_back(pos);
            if (tr.position_history.size() > HISTORY_SIZE)
                tr.position_history.pop_front();

            tr.x = candidates[best_idx].x;
            tr.P = candidates[best_idx].P;
            tr.confidence = candidates[best_idx].w;
            tr.missed_count = 0;

            candidate_used[best_idx] = true;
            track_used[i] = true;  //一个 Track 一帧只能“被续命一次”
        } 
        else {
            // 没匹配上
            tr.missed_count++;
        }
    }

    // ===== 2. 为未使用的 candidate 创建新 Track 创建新ID=====
    for (int j = 0; j < candidates.size(); ++j) {
        if (candidate_used[j]) continue;

        // 寻找一个当前没有被任何 active 轨迹占用的 ID (0 到 NUM_DRONES-1)
        int free_id = -1;
        for (int id_search = 0; id_search < NUM_DRONES; ++id_search) {
            bool id_occupied = false;              
            for (const auto& tr : tracks_) {
                if (tr.active && tr.id == id_search) {
                    id_occupied = true;
                    break;
                }
            }
            if (!id_occupied) {
                free_id = id_search;
                break;
            }

            // 如果找到了空闲 ID，就用这个 ID 创建新轨迹
            if (free_id != -1) {
                Tracknew tr;
                tr.id = free_id; // 使用找到的 0~9 之间的空闲 ID
                tr.x = candidates[j].x;
                tr.P = candidates[j].P;
                tr.confidence = candidates[j].w;
                tr.missed_count = 0;
                tr.active = true; 
                Eigen::Vector2f pos;                   
                pos << tr.x(0), tr.x(2);
                tr.position_history.push_back(pos);
                
                tracks_.push_back(tr);
                ROS_INFO("New Target! Assigned ID: %d", tr.id);
            }
        }

        bool assigned = false;

        // 4.3：尝试复活失效 Track
        for (size_t i = 0; i < tracks_.size(); ++i) {

            auto& tr = tracks_[i];

            if (tr.active)
                continue;

            float dist = (tr.x.head(2) - candidates[j].x.head(2)).norm();

            if (dist < ASSOC_DIST_TH) {

                tr.x = candidates[j].x;
                tr.P = candidates[j].P;
                tr.confidence = candidates[j].w;
                tr.missed_count = 0;
                tr.active = true;

                assigned = true;
                break;
            }
        }

        if (assigned)
            continue;

        // 4.4：真的没法用，才新建 Track
        Tracknew tr;
        tr.id = next_track_id_++;
        tr.x = candidates[j].x;
        tr.P = candidates[j].P;
        tr.confidence = candidates[j].w;
        tr.missed_count = 0;
        tr.active = true;

        Eigen::Vector2f pos;
        pos << tr.x(0), tr.x(2);
        tr.position_history.push_back(pos);

        tracks_.push_back(tr);
    }

    // ===== 3. 关闭长期未匹配的 Track =====
    for (auto& tr : tracks_) {
        if (tr.missed_count > MAX_MISSED) {
            tr.active = false;
        }
        
    }
}


// 基于历史位置预测当前位置
Eigen::Vector2f PhdFilter::predict_position(const Tracknew& tr) const
{
    if (tr.position_history.size() < 2)
        return tr.position_history.back();

    Eigen::Vector2f last = tr.position_history.back();
    Eigen::Vector2f prev = tr.position_history[tr.position_history.size() - 2];

    Eigen::Vector2f vel = (last - prev) / dt_cam;
    return last + vel * dt_cam;
}

// 计算速度连续性得分
float PhdFilter::calculate_velocity_consistency(const Tracknew& tr, const Eigen::Vector2f& candidate_velocity) const
{
    if (tr.velocity_history.empty())
        return 1.0f;  // 没历史，不惩罚

    Eigen::Vector2f v_avg(0, 0);
    for (const auto& v : tr.velocity_history)
        v_avg += v;

    v_avg /= tr.velocity_history.size();

    float diff = (v_avg - candidate_velocity).norm();
    return std::exp(-diff);   // [0,1]，越小差异越接近 1
}



void PhdFilter::phd_track()
{
    #ifdef TIMING：
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
        //cleanup_memory();//清理内存
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



//初始化历史记录
void PhdFilter::initialize_velocity_history() 
{
    velocity_history.resize(NUM_DRONES);
    position_history.resize(NUM_DRONES);
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
    ROS_ERROR_STREAM("wk is:\n" << setprecision(3) << wk << endl);
    ROS_ERROR_STREAM("I is:\n" << I << "\nI_weights is:\n" << I_weights << endl);
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
    for(int i=0; i<newIndex.cols(); i++)
    {
        if(i > NUM_DRONES)
        {
            continue;
        }

        int sortedIndex = newIndex(i);
        wk_bar_fixed.block(0, sortedIndex, 1, 1) = wk_bar_fixed_k.block(0, i, 1, 1);
        mk_bar_fixed.block(0, sortedIndex, n_state, 1) = mk_bar_fixed_k.block(0, i, n_state, 1);
        Pk_bar_fixed.block(0, n_state*sortedIndex, n_state, n_state) = Pk_bar_fixed_k.block(0, n_state*i, n_state, n_state);
    }
    ROS_INFO_STREAM("wk_bar_fixed is:\n" << wk_bar_fixed << "\n");
    ROS_INFO_STREAM("mk_bar_fixed is:\n" << mk_bar_fixed << "\n");
    ROS_INFO_STREAM("Pk_bar_fixed is:\n" << Pk_bar_fixed << "\n");

    numTargets_Jk_minus_1 = wk_bar_fixed.cols();
    ROS_INFO_STREAM("numTargets_Jk_minus_1: " << numTargets_Jk_minus_1);
}

// void PhdFilter::phd_state_extract() //状态提取
// {

//     ROS_INFO("============ 5. extract ============= ");
//     Eigen::MatrixXf velocity, position;
//     velocity = Eigen::MatrixXf(2,1);
//     position = Eigen::MatrixXf(2,1);
//     float gain_fine_tuned = 1.0;  // 速度计算增益（微调）
//     float weight_threshold_for_extraction = 0.5;  // 提取的阈值

//     ROS_INFO_STREAM("DT Cam: " << dt_cam << "\n");
//     //update state for next iterations
    
//     // X_k = mk_minus_1;
//     if(k_iteration > 3)   // 迭代次数>3时（初始化完成），更新状态
//     {
//         //update state for next iterations
//         // wk_minus_1 = wk_bar_fixed;
//         // mk_minus_1 = mk_bar_fixed;
//         // Pk_minus_1 = Pk_bar_fixed.cwiseAbs();

//         // X_k = mk_minus_1;
//         // cout << "--- X_k: " << endl << X_k << endl;
//         for(int i=0; i<wk_bar_display.cols(); i++)
//         {
//             if(wk_bar_display(i)==-1)
//             {
//                 X_k.block(0, i, n_state, 1).setConstant(-1);
//                 //X_k.block(0, i, n_state, 1) = mk_bar_display.block(0, i, n_state, 1);
//                 wk_minus_1.block(0, i, 1, 1) = wk_bar_fixed.block(0, i, 1, 1);
//                 mk_minus_1.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
//                 Pk_minus_1.block(0, n_state*i, n_state, n_state) = Pk_bar_fixed.block(0, n_state*i, n_state, n_state).cwiseAbs();
//                 cout<<"xx"<<endl;
//             }
//             else if(wk_bar_display(i) < weight_threshold_for_extraction  && wk_bar_display(i) != -1)  // 权重低于阈值（不可靠）的时候，状态就沿用预测值
//             {
//                 ROS_INFO("!!11");
//                 X_k.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
//                 mk_bar_fixed.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
//                 wk_bar_fixed(i) = wk_minus_1(i);
//                 Pk_bar_fixed.block(0, n_state*i, n_state, n_state) = Pk_minus_1.block(0, n_state*i, n_state, n_state).cwiseAbs();
//             }
//             else  // 权重高于阈值（可靠）的时候，状态就用当前的修剪后的结果
//             {
//                 ROS_INFO("!!22");
//                 candidates_.clear();
//                 for (int i = 0; i < wk_bar_fixed.cols(); i++)
//                 {
//                     if (wk_bar_fixed(i) < weight_threshold_for_extraction)
//                         continue;

//                     Candidate c;
//                     c.x = mk_bar_fixed.block(0, i, n_state, 1);
//                     c.P = Pk_bar_fixed.block(0, n_state*i, n_state, n_state);
//                     c.w = wk_bar_fixed(i);

//                     candidates_.push_back(c);
//                 }

//                 X_k.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
//                 ROS_INFO("!!aa");
//                 wk_minus_1.block(0, i, 1, 1) = wk_bar_fixed.block(0, i, 1, 1);
//                 ROS_INFO("!!bb");
//                 mk_minus_1.block(0, i, n_state, 1) = mk_bar_fixed.block(0, i, n_state, 1);
//                 ROS_INFO("!!cc");
//                 Pk_minus_1.block(0, n_state*i, n_state, n_state) = Pk_bar_fixed.block(0, n_state*i, n_state, n_state).cwiseAbs();
//                 ROS_INFO("!!33");
//             }
            
//         }

//         ROS_INFO_STREAM("mK in extract: \n" << mk_minus_1);
//         ROS_INFO_STREAM("wK in extract: \n" << wk_bar_fixed);
//         ROS_INFO_STREAM("PK in extract: \n" << Pk_bar_fixed);
//         ROS_INFO_STREAM("XK in extract: \n" << X_k);
//     }
//     else  //迭代次数≤3（初始化阶段），直接采用修剪结果
//     {
//         wk_minus_1 = wk_bar_fixed;
//         mk_minus_1 = mk_bar_fixed;
//         Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
//         X_k = mk_minus_1;
//     }
//     if (k_iteration > 3)
//     {
//         for (int i = 0; i < wk_bar_fixed.cols(); i++)
//         {
//             position.block<1, 1>(0, 0) = (X_k.block<1,1>(0,i) - X_k_previous.block<1,1>(0,i));
//             position.block<1, 1>(1, 0) = (X_k.block<1,1>(2,i) - X_k_previous.block<1,1>(2,i));
//             velocity = position/ (dt_cam*gain_fine_tuned) ;
//             mk_minus_1.block<1,1>(1,i) = velocity.block<1,1>(0,0);
//             mk_minus_1.block<1,1>(3,i) = velocity.block<1,1>(1,0);
//             ROS_INFO_STREAM("--- position: " << endl << position << endl);
//             ROS_INFO_STREAM("--- dt_cam: " << endl << dt_cam << endl);
//             ROS_INFO_STREAM("--- dt: " << endl << dt << endl);
//             ROS_INFO_STREAM("--- velocity: " << endl << velocity << endl);
//         }

//     }
//     wk_minus_1 = wk_bar_fixed;
//     mk_minus_1 = mk_bar_fixed;
//     Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
//     X_k_previous = X_k;
    

//     ROS_ERROR_STREAM("mk_bar_fixed: \n" << mk_bar_fixed);
//     ROS_ERROR_STREAM("mk_bar_display: \n" << mk_bar_display);
//     ROS_ERROR_STREAM("wk_bar_fixed: \n" << mk_bar_fixed);
//     ROS_ERROR_STREAM("mk_bar_display: \n" << mk_bar_display);

//     std::vector<Candidate> candidates;

//     for (int i = 0; i < NUM_DRONES; ++i) {
//         if (wk_bar_display(i) > 0.3f) {
//             Candidate c;
//             c.x = mk_bar_display.col(i);
//             c.P = Pk_bar_display.block(0, n_state*i, n_state, n_state);
//             c.w = wk_bar_display(i);
//             candidates.push_back(c);
//         }
//     }
    
//     updateTracks(candidates);


// }
void PhdFilter::phd_state_extract() // 状态提取
{
    ROS_INFO("============ 5. extract ============= ");
    Eigen::MatrixXf velocity, position;
    velocity = Eigen::MatrixXf(2,1);
    position = Eigen::MatrixXf(2,1);
    float gain_fine_tuned = 1.0; 
    float weight_threshold_for_extraction = 0.5; 

    // --- 【修改 1：数据源头重定向】 ---
    // 原本你在这里从 mk_bar_display 拿数据，这是错的，因为 display 此时还没被赋值。
    // 应该直接从剪枝合并后的结果 mk_bar_fixed 拿数据。
    std::vector<Candidate> candidates_for_matching;
    for (int i = 0; i < wk_bar_fixed.cols(); i++) {
        if (wk_bar_fixed(i) > 0.3f) { // 降低门限，确保能抓到目标
            Candidate c;
            c.x = mk_bar_fixed.col(i);
            c.P = Pk_bar_fixed.block(0, n_state*i, n_state, n_state);
            c.w = wk_bar_fixed(i);
            candidates_for_matching.push_back(c);
        }
    }

    // --- 【修改 2：在这里执行 ID 分配】 ---
    // 只有执行了 updateTracks，才会根据 Candidate 生成或更新带有 ID 的 tracks_
    updateTracks(candidates_for_matching);

    // --- 【修改 3：根据追踪结果填充 Display 矩阵】 ---
    // 这步保证了 mk_bar_display 不再是全 0
    mk_bar_display.setConstant(-1);
    wk_bar_display.setConstant(-1);
    for (const auto& tr : tracks_) {
        if (tr.active && tr.id < NUM_DRONES) {
            mk_bar_display.col(tr.id) = tr.x;
            wk_bar_display(tr.id) = tr.confidence;
        }
    }

    // --- 【以下保留你的原逻辑，但修复内部索引】 ---
    if(k_iteration > 3)   
    {
        // 注意：这里的循环上限应为 NUM_DRONES，因为我们要更新每个 ID 的状态
        for(int i=0; i < NUM_DRONES; i++)
        {
            // 如果该 ID 当前没被激活
            if(wk_bar_display(i) == -1)
            {
                X_k.block(0, i, n_state, 1).setConstant(-1);
                // 此时 mk_minus_1 保持上一帧预测
            }
            // 权重低，沿用预测值
            else if(wk_bar_display(i) < weight_threshold_for_extraction)
            {
                X_k.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
            }
            // 权重高，更新反馈
            else 
            {
                X_k.block(0, i, n_state, 1) = mk_bar_display.col(i);
                mk_minus_1.col(i) = mk_bar_display.col(i);
                wk_minus_1(i) = wk_bar_display(i);
            }
        }
    }
    else // 初始化阶段
    {
        wk_minus_1 = wk_bar_fixed;
        mk_minus_1 = mk_bar_fixed;
        Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
        X_k = mk_bar_fixed;
    }

    // 速度差分逻辑（保持你的原样，但确保 X_k_previous 尺寸一致）
    if (k_iteration > 3)
    {
        for (int i = 0; i < NUM_DRONES; i++)
        {
            if (X_k(0,i) != -1 && X_k_previous(0,i) != -1) {
                position(0, 0) = (X_k(0,i) - X_k_previous(0,i));
                position(1, 0) = (X_k(2,i) - X_k_previous(2,i));
                velocity = position / (dt_cam * gain_fine_tuned);
                // 更新反馈给下一帧的速度估计
                mk_minus_1(1,i) = velocity(0,0);
                mk_minus_1(3,i) = velocity(1,0);
            }
        }
    }

    // 最终同步
    X_k_previous = X_k;

    ROS_ERROR_STREAM("Final Display Check - mk_bar_display col 0: \n" << mk_bar_display.col(0).transpose());
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