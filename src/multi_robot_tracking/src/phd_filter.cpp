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
// ==========================================
// 放在 phd_filter.cpp 的 updateTracks 函数上方
// ==========================================

// 权重参数 (根据实际情况微调)
static const int MAX_MISSED = 20;           // 连续没匹配的最大帧数
// ==========================================
// 参数调整区 (针对 224x224 坐标系优化)
// ==========================================

// 权重：刚开始跟丢通常是因为方向/速度惩罚太重，我们先调低它们
static const float W_DIST = 0.8f;       // 距离权重 (主力)
static const float W_DIR  = 0.1f;       // 方向权重 (辅助)
static const float W_VEL  = 0.1f;       // 速度大小权重 (辅助)

// 阈值：关键修改点！
static const float GATE_DIST_BASE = 400.0f; // 搜索半径加大，防止跟丢
static const float DUPLICATE_DIST = 20.0f; // 【关键】互斥半径加大！224像素下，20像素内视为同一个目标
static const float NEW_TRACK_TH   = 0.3f;  // 新生目标权重门限
static const float MERGE_CAND_DIST = 15.0f; // 【新增】候选点合并距离

struct MatchPair {
    int track_idx;
    int cand_idx;
    float cost;
    bool operator<(const MatchPair& other) const { return cost < other.cost; }
};

void PhdFilter::updateTracks(const std::vector<Candidate>& raw_candidates) 
{
    // --- 0. 初始化 ---
    if (tracks_.size() != NUM_DRONES) {
        tracks_.resize(NUM_DRONES);
        for(int i=0; i<NUM_DRONES; ++i) { 
            tracks_[i].id = i; tracks_[i].active = false; 
            tracks_[i].missed_count = 999; tracks_[i].velocity = Eigen::Vector2f::Zero();
        }
    }

    // --- 1. 候选点预处理 (合并双黄蛋) ---
    // PHD滤波器有时会对一个强目标输出2个靠得很近的点，必须先合并，否则必定导致多ID
    std::vector<Candidate> candidates = raw_candidates; // 拷贝一份
    std::vector<bool> skip_cand(candidates.size(), false);
    
    // 简单的合并逻辑：如果两个候选点太近，保留权重大的，把小的标记为无效
    for(int i=0; i<candidates.size(); ++i) {
        if(skip_cand[i]) continue;
        for(int j=i+1; j<candidates.size(); ++j) {
            if(skip_cand[j]) continue;
            Eigen::Vector2f p1; p1 << candidates[i].x(0), candidates[i].x(2);
            Eigen::Vector2f p2; p2 << candidates[j].x(0), candidates[j].x(2);
            
            if((p1 - p2).norm() < MERGE_CAND_DIST) {
                // 距离过近，视为同一个目标的抖动
                // 保留权重大的
                if(candidates[i].w >= candidates[j].w) {
                    skip_cand[j] = true;
                } else {
                    skip_cand[i] = true;
                    break; // i 已经被废了，跳出内层
                }
            }
        }
    }

    // --- 2. 内部去重 (Track Self-Collision) ---
    // 防止两个 ID 粘在一起
    for (int i = 0; i < NUM_DRONES; ++i) {
        if (!tracks_[i].active) continue;
        for (int j = i + 1; j < NUM_DRONES; ++j) {
            if (!tracks_[j].active) continue;
            Eigen::Vector2f p1; p1 << tracks_[i].x(0), tracks_[i].x(2);
            Eigen::Vector2f p2; p2 << tracks_[j].x(0), tracks_[j].x(2);
            
            if ((p1 - p2).norm() < DUPLICATE_DIST) {
                ROS_WARN("ID Collision! %d and %d merged.", i, j);
                // 谁的历史长保留谁，或者保留 ID 小的
                tracks_[j].active = false; 
                tracks_[j].missed_count = 999; 
            }
        }
    }

    std::vector<bool> candidate_used(candidates.size(), false);
    std::vector<bool> track_matched(NUM_DRONES, false);
    std::vector<MatchPair> all_matches;

    // --- 3. 构建代价矩阵 (Cost Matrix) ---
    for (int i = 0; i < NUM_DRONES; ++i) {
        auto& tr = tracks_[i];
        if (!tr.active) continue;

        // 运动预测
        Eigen::Vector2f cur_pos; cur_pos << tr.x(0), tr.x(2);
        Eigen::Vector2f pred_pos = cur_pos + tr.velocity * dt_cam;
        float search_radius = GATE_DIST_BASE + tr.velocity.norm() * dt_cam * 2.0f; // 扩大搜索范围

        for (int j = 0; j < candidates.size(); ++j) {
            if (skip_cand[j]) continue; // 跳过被合并掉的候选点
            
            Eigen::Vector2f cand_pos; cand_pos << candidates[j].x(0), candidates[j].x(2);
            float dist_err = (cand_pos - pred_pos).norm();

            if (dist_err > search_radius) continue; 

            // 代价计算
            float c_dist = dist_err;
            float c_dir = 0.0f;
            float c_vel = 0.0f;

            // 【关键修改】：只有当轨迹稳定（历史数据>5帧）才启用高级特征
            // 否则只用距离匹配。这能解决“起步阶段跟丢”的问题。
            if (tr.position_history.size() > 5) {
                Eigen::Vector2f cand_vel_est = (cand_pos - cur_pos) / dt_cam;
                float v_tr = tr.velocity.norm();
                float v_ca = cand_vel_est.norm();
                if (v_tr > 0.1f && v_ca > 0.1f) {
                    float cos_theta = tr.velocity.dot(cand_vel_est) / (v_tr * v_ca);
                    c_dir = 1.0f - cos_theta; 
                }
                c_vel = std::abs(v_tr - v_ca);
            }

            float total_cost = (W_DIST * c_dist) + (W_DIR * c_dir * 20.0f) + (W_VEL * c_vel);
            
            MatchPair mp; mp.track_idx = i; mp.cand_idx = j; mp.cost = total_cost;
            all_matches.push_back(mp);
        }
    }

    // --- 4. 全局分配 ---
    std::sort(all_matches.begin(), all_matches.end());
    for (const auto& mp : all_matches) {
        if (!track_matched[mp.track_idx] && !candidate_used[mp.cand_idx]) {
            auto& tr = tracks_[mp.track_idx];
            int cid = mp.cand_idx;

            // 计算瞬时速度
            Eigen::Vector2f new_pos; new_pos << candidates[cid].x(0), candidates[cid].x(2);
            Eigen::Vector2f old_pos; old_pos << tr.x(0), tr.x(2);
            Eigen::Vector2f instant_vel = (new_pos - old_pos) / dt_cam;

            // 更新
            tr.x = candidates[cid].x;
            tr.P = candidates[cid].P;
            tr.confidence = candidates[cid].w;
            tr.missed_count = 0;
            
            // 速度平滑更新
            if(tr.velocity.norm() == 0) tr.velocity = instant_vel;
            else tr.velocity = 0.5f * instant_vel + 0.5f * tr.velocity;

            tr.position_history.push_back(new_pos);
            if (tr.position_history.size() > HISTORY_SIZE) tr.position_history.pop_front();

            track_matched[mp.track_idx] = true;
            candidate_used[cid] = true;
        }
    }

    // 未匹配的增加丢失计数
    for (int i = 0; i < NUM_DRONES; ++i) {
        if (tracks_[i].active && !track_matched[i]) tracks_[i].missed_count++;
    }

    // --- 5. 复活逻辑 (Revival) ---
    for (int j = 0; j < candidates.size(); ++j) {
        if (candidate_used[j] || skip_cand[j]) continue;
        Eigen::Vector2f cand_pos; cand_pos << candidates[j].x(0), candidates[j].x(2);

        // 互斥：复活点周围不能有活着的 ID (20像素)
        bool overlap = false;
        for(int k=0; k<NUM_DRONES; ++k) {
            if(tracks_[k].active) {
                Eigen::Vector2f tp; tp << tracks_[k].x(0), tracks_[k].x(2);
                if((cand_pos - tp).norm() < DUPLICATE_DIST) { overlap = true; break; }
            }
        }
        if(overlap) continue;

        int best_id = -1; 
        float min_d = 1e9f;
        for(int i=0; i<NUM_DRONES; ++i) {
            if(!tracks_[i].active && tracks_[i].missed_count < 30 && !tracks_[i].position_history.empty()) {
                float d = (cand_pos - tracks_[i].position_history.back()).norm();
                if(d < GATE_DIST_BASE * 1.5f && d < min_d) { min_d = d; best_id = i; }
            }
        }

        if(best_id != -1) {
            auto& tr = tracks_[best_id];
            tr.active = true; tr.missed_count = 0;
            tr.x = candidates[j].x; tr.P = candidates[j].P; tr.confidence = candidates[j].w;
            Eigen::Vector2f p; p << tr.x(0), tr.x(2);
            tr.position_history.push_back(p);
            candidate_used[j] = true;
        }
    }

    // --- 6. 新生逻辑 (Birth) ---
    for (int j = 0; j < candidates.size(); ++j) {
        if (candidate_used[j] || skip_cand[j]) continue;
        if (candidates[j].w < NEW_TRACK_TH) continue;

        Eigen::Vector2f cand_pos; cand_pos << candidates[j].x(0), candidates[j].x(2);
        
        // 互斥：绝不在现有 ID 旁边生孩子
        bool overlap = false;
        for(int k=0; k<NUM_DRONES; ++k) {
            if(tracks_[k].active) {
                Eigen::Vector2f tp; tp << tracks_[k].x(0), tracks_[k].x(2);
                if((cand_pos - tp).norm() < DUPLICATE_DIST) { overlap = true; break; }
            }
        }
        if(overlap) continue;

        int free_id = -1; int max_m = -1;
        for(int i=0; i<NUM_DRONES; ++i) {
            if(!tracks_[i].active && tracks_[i].missed_count > max_m) {
                max_m = tracks_[i].missed_count; free_id = i;
            }
        }

        if(free_id != -1) {
            auto& tr = tracks_[free_id];
            tr.active = true; tr.id = free_id;
            tr.x = candidates[j].x; tr.P = candidates[j].P; tr.confidence = candidates[j].w;
            tr.missed_count = 0; tr.velocity = Eigen::Vector2f::Zero();
            tr.position_history.clear(); tr.velocity_history.clear();
            Eigen::Vector2f p; p << tr.x(0), tr.x(2);
            tr.position_history.push_back(p);
            candidate_used[j] = true;
        }
    }

    // --- 7. 清理 ---
    for (int i = 0; i < NUM_DRONES; ++i) {
        if (tracks_[i].active && tracks_[i].missed_count > MAX_MISSED) tracks_[i].active = false;
    }
    
    
    // // === 新增：计算最近 5 帧 P(0,0) 的方差 ===
    // // ==========================================
    // for (int i = 0; i < NUM_DRONES; ++i) {
    //     auto& tr = tracks_[i];
        
    //     // 只计算活跃的，或者刚丢失不久的
    //     if (tr.active || tr.missed_count < 10) {
            
    //         // 1. 存入当前帧的 P(0,0)
    //         // 注意：这里取的是位置X的协方差
    //         tr.p00_history.push_back(tr.P(0,0));

    //         // 2. 保持队列长度为 5
    //         if (tr.p00_history.size() > 5) {
    //             tr.p00_history.pop_front();
    //         }

    //         // 3. 计算方差
    //         if (tr.p00_history.size() > 1) {
    //             // 计算均值
    //             float sum = 0.0f;
    //             for (float val : tr.p00_history) sum += val;
    //             float mean = sum / tr.p00_history.size();

    //             // 计算方差 sum((x - mean)^2) / N
    //             float sq_sum = 0.0f;
    //             for (float val : tr.p00_history) {
    //                 sq_sum += (val - mean) * (val - mean);
    //             }
    //             tr.p00_variance_5_frames = sq_sum / tr.p00_history.size();
    //         } else {
    //             tr.p00_variance_5_frames = 0.0f;
    //         }
    //     } else {
    //         // 如果由不活跃变活跃，可能需要清空历史（可选）
    //         if (!tr.p00_history.empty()) tr.p00_history.clear();
    //     }
    // }


}
void PhdFilter::initTracks() {
    tracks_.clear();
    tracks_.resize(NUM_DRONES); // 直接开辟好固定大小
    for(int i=0; i<NUM_DRONES; ++i) {
        tracks_[i].id = i;        // ID 永远等于索引
        tracks_[i].active = false;
        tracks_[i].missed_count = 999; // 初始状态为空
        tracks_[i].x = Eigen::VectorXf::Zero(n_state);
        tracks_[i].P = Eigen::MatrixXf::Identity(n_state, n_state);
    }
}

// static const int MAX_MISSED = 10;           // 连续没匹配的最大帧数
// // 定义一个结构体来存储潜在的配对
// struct AssociationPair {
//     int track_idx;
//     int cand_idx;
//     float cost;
    
//     // 重载 < 运算符，用于排序
//     bool operator<(const AssociationPair& other) const {
//         return cost < other.cost;
//     }
// };

// void PhdFilter::updateTracks(const std::vector<Candidate>& candidates) 
// {
//     // ==========================================
//     // 1. 计算代价矩阵 (Cost Matrix Calculation)
//     // ==========================================
    
//     // 存储所有可能的配对 (只要在半径内，都算潜在配对)
//     std::vector<AssociationPair> all_associations;

//     // 动态权重 (最近一帧权重最大)
//     const float W_DIR[5] = {0.40f, 0.25f, 0.15f, 0.10f, 0.10f};

//     for (size_t i = 0; i < tracks_.size(); ++i) {
//         auto& tr = tracks_[i];
//         if (!tr.active) continue;

//         // --- 确定搜索半径 ---
//         float search_radius = 40.0f; 
//         if (tr.position_history.size() >= 2) {
//              float last_step = (tr.position_history.back() - tr.position_history[tr.position_history.size()-2]).norm();
//              search_radius = std::max(last_step * 2.0f, 40.0f); 
//         }

//         // --- 遍历所有候选点，计算 Cost ---
//         for (int j = 0; j < candidates.size(); ++j) {
            
//             // 1. 距离门控 (Gating)
//             float dist = (candidates[j].x.head(2) - tr.x.head(2)).norm();
//             if (dist > search_radius) continue; // 太远了，根本没资格竞争

//             // 2. 计算综合代价 (Cost)
//             float score = dist; // 基础分是距离

//             // 3. 加上行为/方向惩罚 (竞争的核心判据)
//             // 如果两个轨迹离该点距离差不多，谁的方向更顺，谁的 Cost 就更低
//             if (tr.velocity_history.size() >= 3) {
//                 Eigen::Vector2f v_obs = (candidates[j].x.head(2) - tr.position_history.back()) / dt_cam;
//                 float v_obs_norm = v_obs.norm();
                
//                 // 计算加权方向一致性
//                 float weighted_cos = 0.0f;
//                 float total_w = 0.0f;
//                 int hist_idx = 0;
//                 // 反向遍历历史速度
//                 for(auto it = tr.velocity_history.rbegin(); it != tr.velocity_history.rend() && hist_idx < 5; ++it, ++hist_idx){
//                      if(it->norm() > 0.1f) {
//                         weighted_cos += W_DIR[hist_idx] * (v_obs.dot(*it) / (v_obs_norm * it->norm() + 1e-5f));
//                         total_w += W_DIR[hist_idx];
//                      }
//                 }

//                 if (total_w > 0.0f) {
//                     float avg_cos = weighted_cos / total_w; 
//                     // 如果方向不一致 (avg_cos < 0.5)，增加罚分
//                     // 这就是解决“竞争”的关键：谁方向对，谁罚分少
//                     if (avg_cos < 0.5f) {
//                         score += (1.0f - avg_cos) * 30.0f; 
//                     }
//                 }
//             }

//             // 将这对组合加入候选池
//             // 阈值设宽一点 (比如 80)，让所有可能的组合都进来参与排序
//             if (score < 80.0f) {
//                 all_associations.push_back({(int)i, j, score});
//             }
//         }
//     }

//     // ==========================================
//     // 2. 全局排序与分配 (Global Assignment)
//     // ==========================================

//     // 关键一步：按 Cost 从小到大排序
//     // 这解决了局部最优问题。全场最好的匹配会排在第一个。
//     std::sort(all_associations.begin(), all_associations.end());

//     std::vector<bool> track_matched(tracks_.size(), false);
//     std::vector<bool> candidate_used(candidates.size(), false);

//     for (const auto& assoc : all_associations) {
//         int t_idx = assoc.track_idx;
//         int c_idx = assoc.cand_idx;

//         // 如果这个 Track 还没配对，且这个 Candidate 也没被用过
//         if (!track_matched[t_idx] && !candidate_used[c_idx]) {
            
//             // === 配对成功 ===
//             update_track_data(tracks_[t_idx], candidates[c_idx]);
            
//             track_matched[t_idx] = true;
//             candidate_used[c_idx] = true;
            
//             // ROS_INFO("Global Match: Track %d <-> Cand %d (Cost: %.1f)", tracks_[t_idx].id, c_idx, assoc.cost);
//         }
//         // 如果 else：说明这个 Candidate 已经被一个 Cost 更低的 Track 抢走了
//         // 或者这个 Track 已经找到了一个 Cost 更低的 Candidate
//         // 这种情况下，自动跳过，寻找下一个最优解
//     }

//     // 处理没抢到点的 Track (Missed)
//     for (size_t i = 0; i < tracks_.size(); ++i) {
//         if (tracks_[i].active && !track_matched[i]) {
//             tracks_[i].missed_count++;
//         }
//     }

//     // ==========================================
//     // 3. 复活逻辑 (Revival) - 针对不活跃轨迹
//     // ==========================================
//     for (int j = 0; j < candidates.size(); ++j) {
//         if (candidate_used[j]) continue;

//         for (auto& tr : tracks_) {
//             if (tr.active) continue; 
            
//             // 复活要求距离很近，且没有被人用过
//             float dist = (tr.x.head(2) - candidates[j].x.head(2)).norm();
//             if (dist < 30.0f) {
//                 update_track_data(tr, candidates[j]);
//                 tr.active = true;
//                 candidate_used[j] = true;
//                 break; 
//             }
//         }
//     }

//     // ==========================================
//     // 4. 新轨迹创建 (严进原则)
//     // ==========================================
//     for (int j = 0; j < candidates.size(); ++j) {
//         if (candidate_used[j]) continue;

//         // 权重门槛
//         if (candidates[j].w < 0.85f) continue;

//         // 重影/幽灵点抑制 (Ghost Suppression)
//         // 即使没匹配上，如果离现有轨迹太近，也不要开新号，认为它是噪声
//         bool is_ghost = false;
//         for (const auto& tr : tracks_) {
//             float dist = (tr.x.head(2) - candidates[j].x.head(2)).norm();
//             if (dist < 40.0f) { is_ghost = true; break; }
//         }
//         if (is_ghost) continue;

//         // 创建逻辑
//         int free_id = -1;
//         for (int id_search = 0; id_search < NUM_DRONES; ++id_search) {
//             bool occupied = false;
//             for (const auto& t : tracks_) if (t.active && t.id == id_search) { occupied = true; break; }
//             if (!occupied) { free_id = id_search; break; }
//         }

//         if (free_id != -1) {
//             create_new_track(candidates[j], free_id);
//         } else if (candidates[j].w > 0.95f) { // 必须非常确信才开临时号
//             create_new_track(candidates[j], next_track_id_++);
//         }
//     }

//     // ==========================================
//     // 5. 清理逻辑
//     // ==========================================
//     auto it = tracks_.begin();
//     while (it != tracks_.end()) {
//         if (it->missed_count > 5) it->active = false;
        
//         // 临时 ID 删得快，固定 ID 留得久
//         int del_th = (it->id >= NUM_DRONES) ? 2 : 20;
        
//         if (!it->active && it->missed_count > del_th) {
//             it = tracks_.erase(it);
//         } else {
//             ++it;
//         }
//     }
// }

// 辅助函数：更新轨迹数据
void PhdFilter::update_track_data(Tracknew& tr, const Candidate& cand) {
    Eigen::Vector2f current_pos = cand.x.head(2);
    Eigen::Vector2f last_pos = tr.position_history.back();
    Eigen::Vector2f current_vel = (current_pos - last_pos) / dt_cam;

    tr.velocity_history.push_back(current_vel);
    if (tr.velocity_history.size() > 10) tr.velocity_history.pop_front();

    tr.position_history.push_back(current_pos);
    if (tr.position_history.size() > 10) tr.position_history.pop_front();

    tr.x = cand.x;
    tr.P = cand.P;
    tr.confidence = cand.w;
    tr.missed_count = 0;
}

// 辅助函数：创建轨迹
void PhdFilter::create_new_track(const Candidate& cand, int id) {
    Tracknew tr;
    tr.id = id;
    tr.x = cand.x;
    tr.P = cand.P;
    tr.confidence = cand.w;
    tr.active = true;
    tr.missed_count = 0;
    tr.position_history.push_back(cand.x.head(2));
    tracks_.push_back(tr);
}

// static const float ASSOC_COST_TH = 50.5f;

// /*cost =
//     0.6 * pos_err +      当前位置误差
//     0.3 * pred_err +     预测误差
//     0.1 * (1 - vel_consistency);   0完全一致  1完全不一致12138
// */
// static const float ASSOC_DIST_TH = 20.0f;   // 距离阈值，后面可以调
// static const int MAX_MISSED = 10;           // 连续没匹配的最大帧数                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
// //新增：更新tracks_轨迹的更新应该只出现在这一步
// void PhdFilter::updateTracks(const std::vector<Candidate>& candidates) 

// {
//     //打印 candidates 信息
//     ROS_ERROR_STREAM("1111111111111111111111111111111111111111111111111");
//     ROS_ERROR_STREAM("Number of candidates: " << candidates.size());
//     for(int k = 0; k < candidates.size(); k++)
//     {
//         ROS_ERROR_STREAM("candidates[" << k << "].x is:\n" << candidates[k].x << "\n");
//     }
//     ROS_ERROR_STREAM("Number of existing tracks: " << tracks_.size());
//     // candidate是新的候选，tracks_是已有的轨迹
//     // 记录每个 candidate 是否已被使用 防止同一个 Candidate 被分给两个 Track
//     std::vector<bool> candidate_used(candidates.size(), false);
//     // 记录每个 track 是否已被使用 防止同一个 Track 被分给两个 Candidate
//     std::vector<bool> track_used(tracks_.size(), false);

//     // ===== 1. 老轨迹匹配 尝试用 candidate 更新已有 Track 标准的贪婪匹配逻辑，遍历所有活跃轨迹来寻找最佳匹配=====
//     for (size_t i = 0; i < tracks_.size(); ++i) {

//         auto& tr = tracks_[i];

//         if (!tr.active)
//             continue;

//         if (tr.missed_count > OCCLUSION_THRESHOLD)
//             continue;   //  关键：失去 ID 保护权

//         float best_cost = 1e9f;
//         int best_idx = -1;

//         for (int j = 0; j < candidates.size(); ++j) {
//             if (candidate_used[j])
//                 continue;

//             Eigen::Vector2f cand_pos;
//             cand_pos << candidates[j].x(0), candidates[j].x(2);

//             Eigen::Vector2f pred_pos = predict_position(tr);

//             float pos_err = (cand_pos - Eigen::Vector2f(tr.x(0), tr.x(2))).norm();

//             float pred_err = (cand_pos - pred_pos).norm();

//             Eigen::Vector2f cand_vel = Eigen::Vector2f::Zero();
//             if (!tr.position_history.empty()) {
//                 cand_vel = (cand_pos - tr.position_history.back()) / dt_cam;
//             }

//             float vel_consistency = calculate_velocity_consistency(tr, cand_vel);

//             float cost =    0.6f * pos_err +
//                             0.3f * pred_err +
//                             0.1f * (1.0f - vel_consistency);

//             if (cost < best_cost) {
//                 best_cost = cost;
//                 best_idx = j;
//             }
//         }
//         ROS_ERROR_STREAM("Track " << tr.id << " best_idx=" << best_idx << " best_cost=" << best_cost);

//         if (best_idx >= 0 && best_cost < ASSOC_COST_TH) {

//             Eigen::Vector2f pos;
//             pos << candidates[best_idx].x(0), candidates[best_idx].x(2);

//             if (!tr.position_history.empty()) {
//                 Eigen::Vector2f vel = (pos - tr.position_history.back()) / dt_cam;

//                 tr.velocity_history.push_back(vel);
//                 if (tr.velocity_history.size() > HISTORY_SIZE)
//                     tr.velocity_history.pop_front();
//             }

//             tr.position_history.push_back(pos);
//             if (tr.position_history.size() > HISTORY_SIZE)
//                 tr.position_history.pop_front();

//             tr.x = candidates[best_idx].x;
//             tr.P = candidates[best_idx].P;
//             tr.confidence = candidates[best_idx].w;
//             tr.missed_count = 0;

//             candidate_used[best_idx] = true; // 标记这个 candidate 已被使用
//             track_used[i] = true;  //一个 Track 一帧只能“被续命一次”
//         } 
//         else {
//             // 没匹配上
//             tr.missed_count++; 
//         }
//     }

//     // ===== 2. 新轨迹创建 为未使用的 candidate 创建新 Track 创建新ID=====
//     for (int j = 0; j < candidates.size(); ++j) {
//         if (candidate_used[j]) continue;

//         // 寻找一个当前没有被任何 active 轨迹占用的 ID (0 到 NUM_DRONES-1)
//         int free_id = -1;
//         for (int id_search = 0; id_search < NUM_DRONES; ++id_search) {
//             bool id_occupied = false;              
//             for (const auto& tr : tracks_) {
//                 if (tr.active && tr.id == id_search) {
//                     id_occupied = true;
//                     break;
//                 }
//             }
//             if (!id_occupied) {
//                 free_id = id_search;
//                 break;
//             }
//         }

//         // 如果找到了空闲 ID，就用这个 ID 创建新轨迹
//         if (free_id != -1) {
//             Tracknew tr;
//             tr.id = free_id; // 使用找到的 0~9 之间的空闲 ID
//             tr.x = candidates[j].x;
//             tr.P = candidates[j].P;
//             tr.confidence = candidates[j].w;
//             tr.missed_count = 0;
//             tr.active = true; 
//             Eigen::Vector2f pos;                   
//             pos << tr.x(0), tr.x(2);
//             tr.position_history.push_back(pos);
            
//             tracks_.push_back(tr);
//             ROS_INFO("New Target! Assigned ID: %d", tr.id);
//             candidate_used[j] = true; // 标记这个 candidate 已被使用
//             continue; // 继续处理下一个 candidate
//         }

//         bool assigned = false;

//         // 4.3：尝试复活失效 Track
//         for (size_t i = 0; i < tracks_.size(); ++i) {

//             auto& tr = tracks_[i];

//             if (tr.active)
//                 continue;

//             float dist = (tr.x.head(2) - candidates[j].x.head(2)).norm();

//             if (dist < ASSOC_DIST_TH) {

//                 tr.x = candidates[j].x;
//                 tr.P = candidates[j].P;
//                 tr.confidence = candidates[j].w;
//                 tr.missed_count = 0;
//                 tr.active = true;

//                 assigned = true;
//                 break;
//             }
//         }

//         if (assigned){
//             candidate_used[j] = true;
//             continue;
//         }



//         // 4.4：真的没法用，才新建 Track
//         Tracknew tr;
//         tr.id = next_track_id_++;
//         tr.x = candidates[j].x;
//         tr.P = candidates[j].P;
//         tr.confidence = candidates[j].w;
//         tr.missed_count = 0;
//         tr.active = true;

//         Eigen::Vector2f pos;
//         pos << tr.x(0), tr.x(2);
//         tr.position_history.push_back(pos);

//         tracks_.push_back(tr);
//     }




//     // ===== 3. 关闭长期未匹配的 Track =====
//     // --- 修改点 5：使用迭代器进行物理删除 ---
//     auto it = tracks_.begin();
//     while (it != tracks_.end()) {
//         // 逻辑 1：标记失效
//         if (it->missed_count > MAX_MISSED) {
//             it->active = false;
//         }
        
//         // 逻辑 2：彻底删除（解决 206 个轨迹的问题）
//         // 如果已经不活跃了，且丢了很久（比如 15 帧），就彻底删掉，释放 ID
//         if (!it->active && it->missed_count > 15) {
//             ROS_INFO("Deleting track ID %d to free memory", it->id);
//             it = tracks_.erase(it); // <--- 这行代码会让 tracks_.size() 降下来
//         } else {
//             ++it; // 只有没删除的时候才移动迭代器
//         }
//     }
// }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// void PhdFilter::updateTracks(const std::vector<Candidate>& candidates) 
// {
//     // === 调试打印 ===
//     ROS_ERROR_THROTTLE(1.0, "UpdateTracks: %lu candidates, %lu tracks", candidates.size(), tracks_.size());
//     ROS_ERROR_STREAM("UpdateTracks: " << candidates.size() << " candidates, " << tracks_.size() << " tracks");
//     // 记录使用状态
//     std::vector<bool> candidate_used(candidates.size(), false);
    
//     // ===========================================
//     // 1. 阶段一：维护现有活跃轨迹 (老员工续命)
//     // ===========================================
//     for (auto& tr : tracks_) {
//         if (!tr.active) continue;
//         if (tr.missed_count > OCCLUSION_THRESHOLD) continue; // 遮挡保护期外才允许匹配

//         float best_cost = 1e9f;
//         int best_idx = -1;

//         // --- 寻找最佳 Candidate ---
//         for (int j = 0; j < candidates.size(); ++j) {
//             if (candidate_used[j]) continue;

//             // 1.1 数据准备
//             Eigen::Vector2f cand_pos;
//             cand_pos << candidates[j].x(0), candidates[j].x(2);
//             Eigen::Vector2f pred_pos = predict_position(tr);
            
//             // 1.2 计算分项误差
//             float pos_err = (cand_pos - Eigen::Vector2f(tr.x(0), tr.x(2))).norm();
//             float pred_err = (cand_pos - pred_pos).norm();
            
//             Eigen::Vector2f cand_vel = Eigen::Vector2f::Zero();
//             if (!tr.position_history.empty()) {
//                 cand_vel = (cand_pos - tr.position_history.back()) / dt_cam;
//             }
//             float vel_consistency = calculate_velocity_consistency(tr, cand_vel);

//             // 1.3 综合 Cost
//             float cost = 0.9f * pos_err + 0.05f * pred_err + 0.05f * (1.0f - vel_consistency);

//             if (cost < best_cost) {
//                 best_cost = cost;
//                 best_idx = j;
//             }
//         }

//         // --- 匹配判定 ---
//         if (best_idx >= 0 && best_cost < ASSOC_COST_TH) {
//             // 更新轨迹状态
//             tr.x = candidates[best_idx].x;
//             tr.P = candidates[best_idx].P;
//             tr.confidence = candidates[best_idx].w;
//             tr.missed_count = 0;

//             // 更新历史记录
//             Eigen::Vector2f pos;
//             pos << tr.x(0), tr.x(2); // 修复：添加类型定义，防止编译报错
//             tr.position_history.push_back(pos);
//             if (tr.position_history.size() > HISTORY_SIZE) tr.position_history.pop_front();

//             if (!tr.position_history.empty()) {
//                  // 简单的速度计算
//                  // ... (你的速度历史代码)
//             }

//             candidate_used[best_idx] = true;
//         } else {
//             tr.missed_count++;
//         }
//     }

//     // ===========================================
//     // 2. 阶段二：处理未匹配的 Candidate (新员工入职)
//     // ===========================================
//     for (int j = 0; j < candidates.size(); ++j) {
//         if (candidate_used[j]) continue; // 已经被匹配的跳过

//         // 策略优先级：
//         // 1. 尝试复活老轨迹 (可选，看你需求，这里先注释掉，专注于ID复用)
//         // 2. 寻找空闲 ID 创建新轨迹
        
//         // --- 核心修复逻辑：寻找空闲 ID ---
//         int free_id = -1;
//         for (int id_search = 0; id_search < NUM_DRONES; ++id_search) {
//             bool id_occupied = false;
//             for (const auto& tr : tracks_) {
//                 // 如果轨迹活跃 且 ID 相同，则该 ID 被占用
//                 if (tr.active && tr.id == id_search) {
//                     id_occupied = true;
//                     break;
//                 }
//             }
//             if (!id_occupied) {
//                 free_id = id_search;
//                 break; // 找到最小的空闲 ID，停止寻找
//             }
//         }

//         // --- 创建新轨迹 ---
//         if (free_id != -1) {
//             Tracknew tr;
//             tr.id = free_id; // 使用复用的 ID
//             tr.x = candidates[j].x;
//             tr.P = candidates[j].P;
//             tr.confidence = candidates[j].w;
//             tr.missed_count = 0;
//             tr.active = true;

//             Eigen::Vector2f pos;
//             pos << tr.x(0), tr.x(2);
//             tr.position_history.push_back(pos);

//             tracks_.push_back(tr);
            
//             // [修复点]：必须标记已使用，防止重复处理
//             candidate_used[j] = true; 
            
//             ROS_INFO("Created new track with ID: %d", free_id);
//         } else {
//             ROS_WARN_THROTTLE(1, "Max drones reached! Cannot assign ID to new candidate.");
//         }
        
//         // 注意：我已经删除了原本后面的 "assigned" 复活逻辑和 "next_track_id++" 逻辑。
//         // 因为只要有空闲 ID，我们就用空闲 ID。
//         // 如果没有空闲 ID (free_id == -1)，说明满员了，那也不应该再创建新 ID 了。
//     }

//     // ===========================================
//     // 3. 阶段三：物理清理 (解雇离职员工)
//     // ===========================================
//     auto it = tracks_.begin();
//     while (it != tracks_.end()) {
//         // 1. 标记失效
//         if (it->missed_count > MAX_MISSED) {
//             it->active = false;
//         }

//         // 2. 物理删除 (修复 206 个轨迹的问题)
//         // 如果 active 为 false 且 missed_count 很大（比如超过 20 帧没看到）
//         // 或者 刚刚 active=false 但我们想立刻释放 ID
//         if (!it->active && it->missed_count > 15) { 
//             // 这里的 15 给了一个缓冲期，防止刚丢就删，删了马上又回来导致 ID 闪烁
//             ROS_INFO("Physically deleting track ID: %d", it->id);
//             it = tracks_.erase(it); // erase 返回下一个有效的迭代器
//         } else {
//             ++it;
//         }
//     }
// }


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
    initTracks();
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
    cout<<"q_pos: "<<q_pos<<endl;
    cout<<"q_vel: "<<q_vel<<endl;
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
        for(int i = 0; i < NUM_DRONES; i++) 
    {
        // 提取每个无人机对应的4×4子矩阵的对角线元素
        Eigen::Vector4f diagonal = Pk_k_minus_1.block(0, n_state * i, n_state, n_state).diagonal();
        ROS_ERROR_STREAM("Drone predict " << i << " Pk diagonal: [" << diagonal(0) << ", " << diagonal(1) << ", " << diagonal(2) << ", " << diagonal(3) << "]");
    }
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
    const float birth_pos_var = 20.0f;       // 位置初始不确定性
    const float birth_vel_var = 20.0f;       // 速度初始不确定性
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
    for(int i = 0; i < NUM_DRONES; i++) 
    {
        // 提取每个无人机对应的4×4子矩阵的对角线元素
        Eigen::Vector4f diagonal = Pk.block(0, n_state * i, n_state, n_state).diagonal();
        ROS_ERROR_STREAM("Drone update " << i << " Pk diagonal: [" << diagonal(0) << ", " << diagonal(1) << ", " << diagonal(2) << ", " << diagonal(3) << "]");
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
    float weight_threshold = 0.03;  // 权重阈值（来自配置）
    float mahalanobis_threshold = 2.0;  // 马氏距离阈值（来自配置）
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
        for(int i = 0; i < NUM_DRONES; i++) 
    {
        // 提取每个无人机对应的4×4子矩阵的对角线元素
        Eigen::Vector4f diagonal = Pk_bar_fixed.block(0, n_state * i, n_state, n_state).diagonal();
        ROS_ERROR_STREAM("Drone prune " << i << " Pk diagonal: [" << diagonal(0) << ", " << diagonal(1) << ", " << diagonal(2) << ", " << diagonal(3) << "]");
    }
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












void PhdFilter::phd_state_extract() 
{
    ROS_INFO("============ 5. State Extraction & Feedback Loop =============");
    
    // 1. 初始化临时变量
    Eigen::MatrixXf velocity = Eigen::MatrixXf::Zero(2,1);
    Eigen::MatrixXf position = Eigen::MatrixXf::Zero(2,1);
    float weight_threshold_for_extraction = 0.4f; // 提取门限调低至0.4，增加交叉时的鲁棒性
    
    // 2. 收集剪枝后的候选目标 (Candidates)
    // 注意：从 mk_bar_fixed 获取，这是经过 prune 和 merge 后的干净数据
    //std::vector<Candidate> candidates_for_matching;
    candidates_for_matching.clear();
    for (int i = 0; i < wk_bar_fixed.cols(); i++) {
        // 只要权重不是微不足道的，都送入 ID 匹配器
        if (wk_bar_fixed(i) > 0.1f) { 
            Candidate c;
            c.x = mk_bar_fixed.col(i);
            c.P = Pk_bar_fixed.block(0, n_state * i, n_state, n_state);
            c.w = wk_bar_fixed(i);
            candidates_for_matching.push_back(c);
        }
    }

    // 3. 执行 ID 分配 (数据关联)
    // 这一步会将 candidates 分配给 tracks_[0...NUM_DRONES-1]
    updateTracks(candidates_for_matching);

    // 4. 填充 Display 矩阵并【同步反馈】给 mk_minus_1
    // 这一步最关键：保证下一帧的“预测”是从这一帧“确认”的位置开始的
    mk_bar_display.setConstant(-1);
    wk_bar_display.setConstant(0); 

    for (int i = 0; i < NUM_DRONES; i++) {
        if (tracks_[i].active) {
            // A. 同步到展示矩阵
            mk_bar_display.col(i) = tracks_[i].x;
            wk_bar_display(i) = tracks_[i].confidence;

            // B. 同步到滤波器内部状态 (Feedback)
            // 即使权重不到 0.5，也要更新位置，否则预测会停在原地
            mk_minus_1.col(i) = tracks_[i].x;
            wk_minus_1(i) = tracks_[i].confidence;
            Pk_minus_1.block(0, n_state * i, n_state, n_state) = tracks_[i].P;

            // C. 决定最终输出给主程序的 X_k
            if (tracks_[i].confidence > weight_threshold_for_extraction) {
                X_k.col(i) = tracks_[i].x;
            } else {
                // 权重偏低（如交叉中），输出预测值保持轨迹连续
                X_k.col(i) = tracks_[i].x; 
            }
        } else {
            // 目标彻底丢失
            X_k.col(i).setConstant(-1);
            wk_minus_1(i) = 0.0f; // 下一帧预测时，这部分权重将近乎为0
        }
    }

    // 5. 速度差分补偿
    // 基于 ID 对齐后的位置差计算速度，并更新回反馈矩阵 mk_minus_1
    if (k_iteration > 5) {
        for (int i = 0; i < NUM_DRONES; i++) {
            // 只有当前帧和前一帧都有效时才计算速度
            if (X_k(0, i) != -1 && X_k_previous(0, i) != -1) {
                position(0, 0) = (X_k(0, i) - X_k_previous(0, i));
                position(1, 0) = (X_k(2, i) - X_k_previous(2, i));
                
                // dt_cam 是两帧之间的时间间隔
                velocity = position / dt_cam; 

                // 将计算出的实时速度反馈给下一帧的预测
                // 这样下一帧的预测位置 = 当前位置 + velocity * dt
                mk_minus_1(1, i) = velocity(0, 0);
                mk_minus_1(3, i) = velocity(1, 0);
                
                ROS_INFO("Drone %d Velocity: VX=%.2f, VY=%.2f", i, velocity(0, 0), velocity(1, 0));
            }
        }
    }

    // 6. 保存历史位置
    X_k_previous = X_k;

    ROS_ERROR_STREAM("Feedback Loop Complete. Valid Tracks: " << (wk_bar_display.array() > 0).count());

}














// void PhdFilter::phd_state_extract() // 状态提取
// {
//     ROS_INFO("============ 5. extract ============= ");
//     Eigen::MatrixXf velocity, position;
//     velocity = Eigen::MatrixXf(2,1);
//     position = Eigen::MatrixXf(2,1);
//     float gain_fine_tuned = 1.0; 
//     float weight_threshold_for_extraction = 0.5; 

//     // --- 【修改 1：数据源头重定向】 ---
//     // 原本你在这里从 mk_bar_display 拿数据，这是错的，因为 display 此时还没被赋值。
//     // 应该直接从剪枝合并后的结果 mk_bar_fixed 拿数据。
//     std::vector<Candidate> candidates_for_matching;
//     for (int i = 0; i < wk_bar_fixed.cols(); i++) {
//         if (wk_bar_fixed(i) > 0.3f) { // 降低门限，确保能抓到目标
//             Candidate c;
//             c.x = mk_bar_fixed.col(i);
//             c.P = Pk_bar_fixed.block(0, n_state*i, n_state, n_state);
//             c.w = wk_bar_fixed(i);
//             candidates_for_matching.push_back(c);
//         }
//     }

//     // --- 【修改 2：在这里执行 ID 分配】 ---
//     // 只有执行了 updateTracks，才会根据 Candidate 生成或更新带有 ID 的 tracks_
//     updateTracks(candidates_for_matching);

//     // --- 【修改 3：根据追踪结果填充 Display 矩阵】 ---
//     // 这步保证了 mk_bar_display 不再是全 0
//     mk_bar_display.setConstant(-1);
//     wk_bar_display.setConstant(-1);
//     for (const auto& tr : tracks_) {
//         if (tr.active && tr.id < NUM_DRONES) {
//             mk_bar_display.col(tr.id) = tr.x;
//             wk_bar_display(tr.id) = tr.confidence;
//         }
//     }

//     // --- 【以下保留你的原逻辑，但修复内部索引】 ---
//     if(k_iteration > 3)   
//     {
//         // 注意：这里的循环上限应为 NUM_DRONES，因为我们要更新每个 ID 的状态
//         for(int i=0; i < NUM_DRONES; i++)
//         {
//             // 如果该 ID 当前没被激活
//             if(wk_bar_display(i) == -1)
//             {
//                 X_k.block(0, i, n_state, 1).setConstant(-1);
//                 // 此时 mk_minus_1 保持上一帧预测
//             }
//             // 权重低，沿用预测值
//             else if(wk_bar_display(i) < weight_threshold_for_extraction)
//             {
//                 X_k.block(0, i, n_state, 1) = mk_minus_1.block(0, i, n_state, 1);
//             }
//             // 权重高，更新反馈
//             else 
//             {
//                 X_k.block(0, i, n_state, 1) = mk_bar_display.col(i);
//                 mk_minus_1.col(i) = mk_bar_display.col(i);
//                 wk_minus_1(i) = wk_bar_display(i);
//             }
//         }
//     }
//     else // 初始化阶段
//     {
//         wk_minus_1 = wk_bar_fixed;
//         mk_minus_1 = mk_bar_fixed;
//         Pk_minus_1 = Pk_bar_fixed.cwiseAbs();
//         X_k = mk_bar_fixed;
//     }

//     // 速度差分逻辑（保持你的原样，但确保 X_k_previous 尺寸一致）
//     if (k_iteration > 3)
//     {
//         for (int i = 0; i < NUM_DRONES; i++)
//         {
//             if (X_k(0,i) != -1 && X_k_previous(0,i) != -1) {
//                 position(0, 0) = (X_k(0,i) - X_k_previous(0,i));
//                 position(1, 0) = (X_k(2,i) - X_k_previous(2,i));
//                 velocity = position / (dt_cam * gain_fine_tuned);
//                 // 更新反馈给下一帧的速度估计
//                 mk_minus_1(1,i) = velocity(0,0);
//                 mk_minus_1(3,i) = velocity(1,0);
//             }
//         }
//     }

//     // 最终同步
//     X_k_previous = X_k;

//     ROS_ERROR_STREAM("Final Display Check - mk_bar_display col 0: \n" << mk_bar_display.col(0).transpose());
// }

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