import os
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

class HOTA:
    """HOTA指标评估器（修复版）"""
    def __init__(self):
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = ['HOTA_TP', 'HOTA_FN', 'HOTA_FP']
        self.float_array_fields = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA', 'OWTA']
        self.float_fields = ['HOTA(0)', 'LocA(0)', 'HOTALocA(0)']
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields

    def _compute_final_fields(self, res):
        """计算最终指标字段"""
        res['DetRe'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'])
        res['DetPr'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FP'])
        res['DetA'] = res['HOTA_TP'] / np.maximum(1, res['HOTA_TP'] + res['HOTA_FN'] + res['HOTA_FP'])
        
        # 修正：AssA应基于TP关联计算
        res['AssA'] = res['AssA'] / np.maximum(1, res['HOTA_TP'])
        res['HOTA'] = np.sqrt(res['DetA'] * res['AssA'])
        res['OWTA'] = np.sqrt(res['DetRe'] * res['AssA'])
        
        # 修正：使用alpha=0的值（而非索引0）
        res['HOTA(0)'] = res['HOTA'][0]
        res['LocA(0)'] = res['LocA'][0]
        res['HOTALocA(0)'] = res['HOTA(0)'] * res['LocA(0)']
        return res

    def eval_sequence(self, data):
        """评估单个序列（修复版）"""
        # 初始化结果
        res = {field: np.zeros(len(self.array_labels)) for field in self.float_array_fields + self.integer_array_fields}
        for field in self.float_fields:
            res[field] = 0.0
        res['LocA'] = np.zeros(len(self.array_labels))  # 单独初始化LocA

        # 空序列处理
        if data['num_tracker_dets'] == 0:
            res['HOTA_FN'] = data['num_gt_dets'] * np.ones(len(self.array_labels))
            res['LocA'] = np.ones(len(self.array_labels))
            res['LocA(0)'] = 1.0
            return res
        if data['num_gt_dets'] == 0:
            res['HOTA_FP'] = data['num_tracker_dets'] * np.ones(len(self.array_labels))
            res['LocA'] = np.ones(len(self.array_labels))
            res['LocA(0)'] = 1.0
            return res

        # 修正：使用正确的全局关联统计
        num_gt_ids = data['num_gt_ids']
        num_tracker_ids = data['num_tracker_ids']
        potential_matches_count = np.zeros((num_gt_ids, num_tracker_ids))
        
        # 第一遍：积累全局轨迹信息（修正公式）
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            similarity = data['similarity_scores'][t]
            
            # 更新全局计数（使用简单累加）
            for i, gt_id in enumerate(gt_ids_t):
                for j, trk_id in enumerate(tracker_ids_t):
                    if similarity[i, j] > 0:
                        potential_matches_count[gt_id, trk_id] += similarity[i, j]
        
        # 计算全局关联分数（修正：使用标准公式）
        sum_rows = potential_matches_count.sum(axis=1)
        sum_cols = potential_matches_count.sum(axis=0)
        global_alignment_score = potential_matches_count / np.maximum(1, sum_rows[:, np.newaxis] + sum_cols[np.newaxis, :] - potential_matches_count)
        
        # 第二遍：计算匹配（每个alpha独立处理）
        for a, alpha in enumerate(self.array_labels):
            matches_count = np.zeros((num_gt_ids, num_tracker_ids))
            
            for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
                # 空帧处理
                if len(gt_ids_t) == 0:
                    res['HOTA_FP'][a] += len(tracker_ids_t)
                    continue
                if len(tracker_ids_t) == 0:
                    res['HOTA_FN'][a] += len(gt_ids_t)
                    continue

                similarity = data['similarity_scores'][t]
                n = len(gt_ids_t)
                m = len(tracker_ids_t)
                
                # 创建代价矩阵（结合全局分数和当前alpha）
                cost_matrix = np.zeros((n, m))
                for i in range(n):
                    for j in range(m):
                        gt_id = gt_ids_t[i]
                        trk_id = tracker_ids_t[j]
                        
                        # 修正：结合全局关联和当前帧IoU
                        if similarity[i, j] >= alpha:
                            cost_matrix[i, j] = global_alignment_score[gt_id, trk_id] * similarity[i, j]
                        else:
                            cost_matrix[i, j] = -1  # 表示不可匹配
                
                # 匈牙利算法匹配
                match_rows, match_cols = linear_sum_assignment(-cost_matrix)
                
                # 统计有效匹配
                for i, j in zip(match_rows, match_cols):
                    if cost_matrix[i, j] > 0:  # 有效匹配
                        res['HOTA_TP'][a] += 1
                        res['LocA'][a] += similarity[i, j]
                        gt_id = gt_ids_t[i]
                        trk_id = tracker_ids_t[j]
                        matches_count[gt_id, trk_id] += 1
                
                # 统计未匹配的GT和检测
                matched_gt = set()
                matched_trk = set()
                for i, j in zip(match_rows, match_cols):
                    if cost_matrix[i, j] > 0:
                        matched_gt.add(i)
                        matched_trk.add(j)
                
                res['HOTA_FN'][a] += n - len(matched_gt)
                res['HOTA_FP'][a] += m - len(matched_trk)

            # 计算AssA（修正：使用标准公式）
            if res['HOTA_TP'][a] > 0:
                sum_matches_rows = matches_count.sum(axis=1)
                sum_matches_cols = matches_count.sum(axis=0)
                ass_a = matches_count / np.maximum(1, sum_matches_rows[:, np.newaxis] + sum_matches_cols[np.newaxis, :] - matches_count)
                res['AssA'][a] = np.sum(matches_count * ass_a)
        
        # 计算最终指标
        res['LocA'] = res['LocA'] / np.maximum(1, res['HOTA_TP'])
        res = self._compute_final_fields(res)
        return res

    def plot_results(self, res, tracker_name, output_dir):
        """绘制结果图表"""
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        
        # 只绘制关键指标
        fields_to_plot = ['HOTA', 'DetA', 'AssA', 'LocA', 'DetRe', 'DetPr']
        styles = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-']
        
        for i, field in enumerate(fields_to_plot):
            plt.plot(self.array_labels, res[field], styles[i], label=field)
        
        plt.xlabel('Alpha Threshold')
        plt.ylabel('Score')
        plt.title(f'{tracker_name} - UAV Tracking Evaluation')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        
        plot_path = os.path.join(output_dir, f'{tracker_name}_hota_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot to: {plot_path}")


def calculate_iou(bbox1, bbox2):
    """计算两个边界框的IoU（修复边缘情况）"""
    # 提取坐标
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # 转换坐标为(x1,y1,x2,y2)格式
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]
    
    # 计算交集坐标
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # 计算并集面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # 避免除以零
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def load_csv_data(file_path, is_gt=True):
    """加载CSV文件数据（修复ID处理）"""
    data = {}
    
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 读取头部
        
        for row in reader:
            # 跳过空行
            if not row or len(row) < 6:
                continue
                
            try:
                # 解析行数据：帧号, ID, x, y, w, h, [置信度], [类型], [可见度]
                frame = int(row[0])
                obj_id = int(row[1])
                bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                
                # 处理可选字段
                conf = 1.0
                obj_type = 1
                visibility = 1.0
                
                if len(row) > 6 and not is_gt:
                    conf = float(row[6])
                
                if len(row) > 7:
                    obj_type = int(row[7])
                
                if len(row) > 8:
                    visibility = float(row[8])
                    
            except (ValueError, IndexError, TypeError) as e:
                print(f"跳过无效行: {row} - 错误: {e}")
                continue
            
            # 只处理无人机类型 (type=1)
            if obj_type != 1:
                continue
                
            if frame not in data:
                data[frame] = []
            
            # 修正：允许0和负ID（MOT格式中0是有效ID）
            data[frame].append({
                'id': obj_id,
                'bbox': bbox,
                'conf': conf,
                'visible': visibility
            })
    
    # 确保帧号按顺序排列
    sorted_frames = sorted(data.keys())
    return {frame: data[frame] for frame in sorted_frames}


def prepare_eval_data(gt_data, tracker_data):
    """准备评估数据结构（修复ID映射）"""
    # 获取所有帧号（并集）
    all_frames = sorted(set(gt_data.keys()) | set(tracker_data.keys()))
    
    # 创建ID映射（原始ID -> 连续索引）
    gt_id_map = {}
    tracker_id_map = {}
    
    # 初始化数据结构
    eval_data = {
        'num_gt_dets': 0,
        'num_tracker_dets': 0,
        'num_gt_ids': 0,
        'num_tracker_ids': 0,
        'gt_ids': [],
        'tracker_ids': [],
        'similarity_scores': []
    }
    
    # 第一遍：收集所有ID用于映射（修正：包含0和负ID）
    for frame in all_frames:
        # 处理真值
        if frame in gt_data:
            for obj in gt_data[frame]:
                if obj['id'] not in gt_id_map:
                    # 修复语法错误：添加右方括号
                    gt_id_map[obj['id']] = len(gt_id_map)
        
        # 处理跟踪结果
        if frame in tracker_data:
            for obj in tracker_data[frame]:
                if obj['id'] not in tracker_id_map:
                    # 修复语法错误：添加右方括号
                    tracker_id_map[obj['id']] = len(tracker_id_map)
    
    eval_data['num_gt_ids'] = len(gt_id_map)
    eval_data['num_tracker_ids'] = len(tracker_id_map)
    
    # 第二遍：处理每一帧
    for frame in all_frames:
        gt_objs = gt_data.get(frame, [])
        tracker_objs = tracker_data.get(frame, [])
        
        # 创建ID数组（使用映射后的ID）
        gt_ids = [gt_id_map[obj['id']] for obj in gt_objs]
        tracker_ids = [tracker_id_map[obj['id']] for obj in tracker_objs]
        
        # 创建相似度矩阵
        similarity = np.zeros((len(gt_objs), (len(tracker_objs)))
        for i, gt_obj in enumerate(gt_objs):
            for j, trk_obj in enumerate(tracker_objs):
                similarity[i, j] = calculate_iou(gt_obj['bbox'], trk_obj['bbox'])
        
        # 添加到数据结构
        eval_data['gt_ids'].append(np.array(gt_ids))
        eval_data['tracker_ids'].append(np.array(tracker_ids))
        eval_data['similarity_scores'].append(similarity)
        
        # 更新检测计数
        eval_data['num_gt_dets'] += len(gt_objs)
        eval_data['num_tracker_dets'] += len(tracker_objs)
    
    return eval_data


def main(gt_csv, tracker_csv, tracker_name, output_dir):
    """主评估函数"""
    print("正在加载数据...")
    gt_data = load_csv_data(gt_csv, is_gt=True)
    tracker_data = load_csv_data(tracker_csv, is_gt=False)
    
    print("准备评估数据...")
    eval_data = prepare_eval_data(gt_data, tracker_data)
    
    print(f"统计信息: GT检测={eval_data['num_gt_dets']}, 跟踪检测={eval_data['num_tracker_dets']}")
    print(f"唯一ID: GT={eval_data['num_gt_ids']}, 跟踪={eval_data['num_tracker_ids']}")
    
    print("计算HOTA指标...")
    hota = HOTA()
    results = hota.eval_sequence(eval_data)
    
    # 打印关键结果
    print("\n===== 评估结果 =====")
    print(f"HOTA(0): {results['HOTA(0)']:.4f}")
    print(f"LocA(0): {results['LocA(0)']:.4f}")
    print(f"DetA (检测准确率): {results['DetA'][0]:.4f}")  # 使用alpha=0的值
    print(f"AssA (关联准确率): {results['AssA'][0]:.4f}")
    
    # 绘制结果
    print("\n生成结果图表...")
    hota.plot_results(results, tracker_name, output_dir)
    
    # 保存详细结果
    result_path = os.path.join(output_dir, f'{tracker_name}_results.txt')
    with open(result_path, 'w') as f:
        f.write("HOTA 详细结果:\n")
        f.write(f"{'Alpha':<6} {'HOTA':<8} {'DetA':<8} {'AssA':<8} {'LocA':<8} {'DetRe':<8} {'DetPr':<8}\n")
        for i, alpha in enumerate(hota.array_labels):
            f.write(f"{alpha:.2f}: {results['HOTA'][i]:.4f}  {results['DetA'][i]:.4f}  "
                    f"{results['AssA'][i]:.4f}  {results['LocA'][i]:.4f}  "
                    f"{results['DetRe'][i]:.4f}  {results['DetPr'][i]:.4f}\n")
        
        f.write("\n关键指标:\n")
        for field in hota.float_fields:
            f.write(f"{field}: {results[field]:.4f}\n")
    
    print(f"保存详细结果到: {result_path}")
    print("评估完成!")


if __name__ == "__main__":
    # 配置参数
    GT_CSV = "/home/mwr/tracking/src/multi_robot_tracking/gt.csv"  # 替换为真值文件路径
    TRACKER_CSV = "/home/mwr/tracking/src/multi_robot_tracking/tracking_results2.csv"  # 替换为跟踪结果文件路径
    TRACKER_NAME = "GM-PHDTracker"  # 您的跟踪器名称
    OUTPUT_DIR = "/home/mwr/tracking/src/multi_robot_tracking/evaluation_results"  # 结果输出目录
    
    # 运行评估
    main(GT_CSV, TRACKER_CSV, TRACKER_NAME, OUTPUT_DIR)
