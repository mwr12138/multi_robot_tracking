from collections import defaultdict

# ===============================
# 路径
# ===============================
gt_path = "/home/mwr/tracking/tools/gt.txt"
out_path = "/home/mwr/tracking/tools/gt4.txt"

COVER_TH = 0.5   # 被遮挡比例阈值（50%）

# ===============================
# 相交面积
# ===============================
def intersection_area(box1, box2):
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    iw = max(0, xb - xa)
    ih = max(0, yb - ya)
    return iw * ih

# ===============================
# 读取 GT
# ===============================
with open(gt_path, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

frames = defaultdict(list)
for line in lines:
    p = line.split(",")
    frame = int(p[0])
    tid   = int(p[1])
    box   = tuple(map(float, p[2:6]))
    frames[frame].append((tid, box, p))

# ===============================
# 遮挡锁定表
# key: (min_id, max_id)
# value: 被保留的 id
# ===============================
occlusion_lock = {}

output = []

# ===============================
# 主流程
# ===============================
for frame in sorted(frames.keys()):
    items = frames[frame]
    removed_ids = set()

    for i in range(len(items)):
        id1, box1, _ = items[i]
        area1 = box1[2] * box1[3]

        for j in range(i+1, len(items)):
            id2, box2, _ = items[j]
            area2 = box2[2] * box2[3]

            inter = intersection_area(box1, box2)
            if inter <= 0:
                continue

            cover1 = inter / area1
            cover2 = inter / area2

            pair = tuple(sorted((id1, id2)))

            # ========= 遮挡持续 =========
            if pair in occlusion_lock:
                if cover1 > COVER_TH or cover2 > COVER_TH:
                    keep = occlusion_lock[pair]
                    drop = id2 if keep == id1 else id1
                    removed_ids.add(drop)
                else:
                    # 遮挡结束
                    del occlusion_lock[pair]

            # ========= 新遮挡 =========
            else:
                if cover1 > COVER_TH or cover2 > COVER_TH:
                    # 谁被遮得多，谁删
                    if cover1 >= cover2:
                        drop = id1
                        keep = id2
                    else:
                        drop = id2
                        keep = id1

                    occlusion_lock[pair] = keep
                    removed_ids.add(drop)

    # 输出当前帧
    for tid, _, raw in items:
        if tid not in removed_ids:
            output.append(",".join(raw))

# ===============================
# 写文件
# ===============================
with open(out_path, "w") as f:
    for l in output:
        f.write(l + "\n")

print("Finished.")
print("Saved to:", out_path)

