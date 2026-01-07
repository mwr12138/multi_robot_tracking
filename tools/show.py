import cv2
import numpy as np
import os

def load_data(file_path):
    data_dict = {}
    max_f = 0
    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return None, 0
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            f_id, obj_id = int(parts[0]), int(parts[1])
            x, y, w, h = map(int, parts[2:6])
            if f_id not in data_dict: data_dict[f_id] = []
            data_dict[f_id].append({'id': obj_id, 'box': (x, y, w, h)})
            max_f = max(max_f, f_id)
    return data_dict, max_f

def render_frame(current_frame, data_dict, mode, img_size=(1080, 1920)):
    """
    mode: 0 为矩形框模式, 1 为中心点模式
    """
    canvas = np.ones((img_size[0], img_size[1], 3), dtype=np.uint8) * 255
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (50, 100, 255), (0, 165, 255), (42, 42, 165)]

    for f in range(1, current_frame + 1):
        if f not in data_dict: continue
        for item in data_dict[f]:
            obj_id, (x, y, w, h) = item['id'], item['box']
            color = colors[(obj_id - 1) % len(colors)]
            
            # 计算中心点
            center_x = x + w // 2
            center_y = y + h // 2

            if mode == 0: # 矩形框模式
                if f == current_frame:
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
                    cv2.putText(canvas, f"ID:{obj_id}", (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 1)
            
            else: # 中心点模式
                if f == current_frame:
                    # 当前位置画一个实心大圆点
                    cv2.circle(canvas, (center_x, center_y), 6, color, -1)
                    cv2.putText(canvas, f"ID:{obj_id}", (center_x + 10, center_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                else:
                    # 历史轨迹画一个小圆点
                    cv2.circle(canvas, (center_x, center_y), 2, color, -1)

    # UI 信息显示
    mode_text = "BOX MODE" if mode == 0 else "CENTER POINT MODE"
    cv2.putText(canvas, f"Frame: {current_frame}  [{mode_text}]", (50, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    cv2.putText(canvas, "M: Switch Mode | A/D: +/-1 | W/S: +/-10 | Q: Quit", (50, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    return canvas

def main():
    gt_path = "/media/mwr/新加卷/Drone/AIRMOT/images/test/DJI_M300_03/gt/gt.txt"
    #gt_path = "/media/mwr/新加卷/Drone/AIRMOT/images/test/Lab_FL5_02/gt/gt.txt"
    #gt_path = "/media/mwr/新加卷/Drone/AIRMOT/images/test/DJI_Phantom_03/gt/gt.txt"
    data, max_frame = load_data(gt_path)
    if data is None: return

    curr_f = 1
    display_mode = 0 # 默认矩形框
    window_name = "AIRMOT Interactive Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    cv2.createTrackbar("Jump", window_name, 1, max_frame, lambda x: None)

    while True:
        img = render_frame(curr_f, data, display_mode)
        cv2.imshow(window_name, img)
        
        key = cv2.waitKey(30) & 0xFF
        
        # 模式切换按键
        if key == ord('m'):
            display_mode = 1 - display_mode # 在 0 和 1 之间切换
        
        # 跳转逻辑
        slider_pos = cv2.getTrackbarPos("Jump", window_name)
        if slider_pos != curr_f and slider_pos != 0:
            curr_f = slider_pos

        if key == ord('d'):
            curr_f = min(curr_f + 1, max_frame)
        elif key == ord('a'):
            curr_f = max(curr_f - 1, 1)
        elif key == ord('w'):
            curr_f = min(curr_f + 10, max_frame)
        elif key == ord('s'):
            curr_f = max(curr_f - 10, 1)
        elif key == ord('q') or key == 27:
            break
        
        cv2.setTrackbarPos("Jump", window_name, curr_f)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
