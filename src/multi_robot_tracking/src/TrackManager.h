#ifndef TRACK_MANAGER_H
#define TRACK_MANAGER_H

#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <ros/ros.h>

class TrackManager {
public:
    // 轨迹状态
    enum TrackState {
        TENTATIVE,      // 暂定状态（新出现的目标）
        CONFIRMED,      // 确认状态（稳定跟踪的目标）
        DELETED         // 删除状态（已消失的目标）
    };
    
    // 单个轨迹信息
    struct Track {
        int id;                         // 轨迹ID
        TrackState state;               // 轨迹状态
        int age;                        // 轨迹年龄（总帧数）
        int invisible_count;            // 连续不可见帧数
        int total_visible_count;        // 总可见帧数
        Eigen::VectorXf last_state;     // 最后已知状态 [x, vx, y, vy]
        Eigen::VectorXf smoothed_state; // 平滑后的状态
        float max_weight;               // 历史最大权重
        std::vector<Eigen::Vector2f> position_history;  // 位置历史
        std::vector<Eigen::Vector2f> velocity_history;  // 速度历史
        
        Track(int track_id) 
            : id(track_id), state(TENTATIVE), age(0), invisible_count(0), 
              total_visible_count(0), max_weight(0.0f) {
            last_state = Eigen::VectorXf::Zero(4);
            smoothed_state = Eigen::VectorXf::Zero(4);
        }
    };

    TrackManager(int num_drones, float confirm_threshold = 0.5f, 
                float delete_threshold = 0.1f, int max_invisible_frames = 10);
    
    // 主要接口
    void update(const Eigen::MatrixXf& X_k, const Eigen::MatrixXf& weights);
    std::vector<int> getActiveTrackIds() const;
    Eigen::MatrixXi getTrackIdsMatrix() const;
    void getTrackState(int track_id, Eigen::VectorXf& state) const;
    
    // 工具函数
    void printTracks() const;
    int getNumConfirmedTracks() const;
    
private:
    // 内部函数
    void initializeTracks(int num_drones);
    int findBestMatch(const Eigen::VectorXf& state, float weight);
    void updateTrack(int track_idx, const Eigen::VectorXf& state, float weight);
    void createNewTrack(const Eigen::VectorXf& state, float weight);
    void deleteOldTracks();
    void smoothTrackState(Track& track);
    
    // 参数
    float confirm_threshold_;      // 确认阈值
    float delete_threshold_;       // 删除阈值
    int max_invisible_frames_;     // 最大不可见帧数
    int next_id_;                  // 下一个可用的轨迹ID
    
    // 轨迹存储
    std::vector<std::shared_ptr<Track>> tracks_;
    std::vector<int> active_track_ids_;  // 当前活跃的轨迹ID
};

#endif