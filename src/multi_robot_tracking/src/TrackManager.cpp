#include "TrackManager.h"
#include <algorithm>
#include <cmath>

TrackManager::TrackManager(int num_drones, float confirm_threshold, 
                          float delete_threshold, int max_invisible_frames)
    : confirm_threshold_(confirm_threshold)
    , delete_threshold_(delete_threshold)
    , max_invisible_frames_(max_invisible_frames)
    , next_id_(0) {
    initializeTracks(num_drones);
}

void TrackManager::initializeTracks(int num_drones) {
    tracks_.clear();
    for (int i = 0; i < num_drones; i++) {
        auto track = std::make_shared<Track>(next_id_++);
        tracks_.push_back(track);
    }
    ROS_INFO("轨迹管理器初始化完成，创建了 %d 个轨迹", num_drones);
}

void TrackManager::update(const Eigen::MatrixXf& X_k, const Eigen::MatrixXf& weights) {
    // 重置活跃轨迹列表
    active_track_ids_.clear();
    
    // 第一步：更新现有轨迹
    std::vector<bool> assigned(tracks_.size(), false);
    std::vector<int> update_order;
    
    // 按权重排序确定更新顺序（高权重优先）
    for (int i = 0; i < weights.cols(); i++) {
        if (weights(i) > delete_threshold_) {
            update_order.push_back(i);
        }
    }
    
    // 按权重降序排序
    std::sort(update_order.begin(), update_order.end(), 
              [&weights](int a, int b) { return weights(a) > weights(b); });
    
    // 为每个检测寻找最佳匹配的轨迹
    for (int detection_idx : update_order) {
        Eigen::VectorXf state = X_k.col(detection_idx);
        float weight = weights(detection_idx);
        
        int best_track_idx = findBestMatch(state, weight);
        if (best_track_idx != -1) {
            updateTrack(best_track_idx, state, weight);
            assigned[best_track_idx] = true;
        } else {
            // 没有找到匹配，创建新轨迹
            createNewTrack(state, weight);
        }
    }
    
    // 第二步：处理未更新的轨迹（目标消失）
    for (size_t i = 0; i < tracks_.size(); i++) {
        if (!assigned[i]) {
            tracks_[i]->invisible_count++;
            tracks_[i]->age++;
            
            // 状态预测（简单匀速模型）
            if (tracks_[i]->state == CONFIRMED) {
                tracks_[i]->last_state(0) += tracks_[i]->last_state(1); // x += vx
                tracks_[i]->last_state(2) += tracks_[i]->last_state(3); // y += vy
                smoothTrackState(*tracks_[i]);
            }
        }
        
        // 第三步：更新轨迹状态
        if (tracks_[i]->state == TENTATIVE) {
            if (tracks_[i]->total_visible_count >= 3) { // 连续3帧可见则确认
                tracks_[i]->state = CONFIRMED;
                ROS_INFO("轨迹 %d 已确认", tracks_[i]->id);
            } else if (tracks_[i]->invisible_count > 2) { // 暂定轨迹快速消失
                tracks_[i]->state = DELETED;
            }
        }
        
        if (tracks_[i]->state == CONFIRMED) {
            if (tracks_[i]->invisible_count > max_invisible_frames_) {
                tracks_[i]->state = DELETED;
                ROS_WARN("轨迹 %d 因长时间不可见被删除", tracks_[i]->id);
            }
        }
        
        // 收集活跃轨迹
        if (tracks_[i]->state != DELETED) {
            active_track_ids_.push_back(tracks_[i]->id);
        }
    }
    
    // 第四步：删除旧轨迹并创建新槽位
    deleteOldTracks();
}

int TrackManager::findBestMatch(const Eigen::VectorXf& state, float weight) {
    int best_track_idx = -1;
    float best_score = -1.0f;
    float position_threshold = 50.0f; // 位置匹配阈值
    
    for (size_t i = 0; i < tracks_.size(); i++) {
        auto& track = tracks_[i];
        
        // 跳过已删除的轨迹
        if (track->state == DELETED) continue;
        
        // 计算位置距离
        float dx = state(0) - track->last_state(0);
        float dy = state(2) - track->last_state(2);
        float distance = std::sqrt(dx*dx + dy*dy);
        
        // 计算速度方向一致性
        float velocity_similarity = 0.0f;
        if (track->velocity_history.size() >= 2) {
            Eigen::Vector2f current_vel(state(1), state(3));
            Eigen::Vector2f last_vel = track->velocity_history.back();
            
            if (current_vel.norm() > 0.1f && last_vel.norm() > 0.1f) {
                float cos_angle = current_vel.dot(last_vel) / (current_vel.norm() * last_vel.norm());
                velocity_similarity = (cos_angle + 1.0f) / 2.0f; // 转换为[0,1]
            }
        }
        
        // 综合评分：距离近 + 速度一致 + 轨迹稳定
        float distance_score = std::max(0.0f, 1.0f - distance / position_threshold);
        float stability_score = track->state == CONFIRMED ? 1.0f : 0.5f;
        float total_score = 0.6f * distance_score + 0.3f * velocity_similarity + 0.1f * stability_score;
        
        if (total_score > best_score && distance < position_threshold) {
            best_score = total_score;
            best_track_idx = i;
        }
    }
    
    return best_track_idx;
}

void TrackManager::updateTrack(int track_idx, const Eigen::VectorXf& state, float weight) {
    auto& track = tracks_[track_idx];
    
    // 更新状态
    track->last_state = state;
    track->invisible_count = 0;
    track->total_visible_count++;
    track->age++;
    track->max_weight = std::max(track->max_weight, weight);
    
    // 更新历史
    track->position_history.push_back(Eigen::Vector2f(state(0), state(2)));
    track->velocity_history.push_back(Eigen::Vector2f(state(1), state(3)));
    
    // 限制历史长度
    if (track->position_history.size() > 20) {
        track->position_history.erase(track->position_history.begin());
        track->velocity_history.erase(track->velocity_history.begin());
    }
    
    // 状态平滑
    smoothTrackState(*track);
    
    // 提升为确认状态
    if (track->state == TENTATIVE && track->total_visible_count >= 3) {
        track->state = CONFIRMED;
    }
}

void TrackManager::createNewTrack(const Eigen::VectorXf& state, float weight) {
    // 寻找可用的轨迹槽位（已删除的）
    for (auto& track : tracks_) {
        if (track->state == DELETED) {
            // 重用轨迹
            *track = Track(next_id_++);
            track->last_state = state;
            track->max_weight = weight;
            updateTrack(std::distance(&tracks_[0], &track), state, weight);
            ROS_INFO("创建新轨迹 %d", track->id);
            return;
        }
    }
    
    // 如果没有可用槽位，创建新轨迹（扩展轨迹列表）
    auto new_track = std::make_shared<Track>(next_id_++);
    new_track->last_state = state;
    new_track->max_weight = weight;
    tracks_.push_back(new_track);
    updateTrack(tracks_.size() - 1, state, weight);
    ROS_WARN("创建额外轨迹 %d (超出初始槽位)", new_track->id);
}

void TrackManager::deleteOldTracks() {
    // 当前实现中，轨迹状态标记为DELETED但不实际删除，以便重用
    // 可以添加逻辑来限制总轨迹数量
}

void TrackManager::smoothTrackState(Track& track) {
    // 简单移动平均平滑
    const float alpha = 0.3f; // 平滑系数
    
    if (track.position_history.size() >= 2) {
        // 位置平滑
        track.smoothed_state(0) = alpha * track.last_state(0) + (1-alpha) * track.smoothed_state(0);
        track.smoothed_state(2) = alpha * track.last_state(2) + (1-alpha) * track.smoothed_state(2);
        
        // 速度平滑（基于历史位置）
        if (track.position_history.size() >= 2) {
            const auto& current_pos = track.position_history.back();
            const auto& prev_pos = track.position_history[track.position_history.size()-2];
            Eigen::Vector2f smoothed_vel = (current_pos - prev_pos) / 1.0f; // 假设1帧时间
            
            track.smoothed_state(1) = smoothed_vel(0);
            track.smoothed_state(3) = smoothed_vel(1);
        }
    } else {
        track.smoothed_state = track.last_state;
    }
}

std::vector<int> TrackManager::getActiveTrackIds() const {
    return active_track_ids_;
}

Eigen::MatrixXi TrackManager::getTrackIdsMatrix() const {
    Eigen::MatrixXi id_matrix(1, active_track_ids_.size());
    for (size_t i = 0; i < active_track_ids_.size(); i++) {
        id_matrix(i) = active_track_ids_[i];
    }
    return id_matrix;
}

void TrackManager::getTrackState(int track_id, Eigen::VectorXf& state) const {
    for (const auto& track : tracks_) {
        if (track->id == track_id && track->state != DELETED) {
            state = track->smoothed_state; // 返回平滑后的状态
            return;
        }
    }
    state = Eigen::VectorXf::Zero(4); // 返回零状态
}

void TrackManager::printTracks() const {
    ROS_INFO("=== 轨迹状态 ===");
    int active_count = 0;
    for (const auto& track : tracks_) {
        if (track->state != DELETED) {
            ROS_INFO("轨迹 %d: 状态=%s, 年龄=%d, 不可见=%d, 总可见=%d, 位置=(%.1f,%.1f)", 
                    track->id,
                    track->state == TENTATIVE ? "暂定" : "确认",
                    track->age,
                    track->invisible_count,
                    track->total_visible_count,
                    track->last_state(0), track->last_state(2));
            active_count++;
        }
    }
    ROS_INFO("活跃轨迹数量: %d/%zu", active_count, tracks_.size());
}

int TrackManager::getNumConfirmedTracks() const {
    int count = 0;
    for (const auto& track : tracks_) {
        if (track->state == CONFIRMED) {
            count++;
        }
    }
    return count;
}