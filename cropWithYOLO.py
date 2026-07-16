"""
用训练好的 YOLO26 模型检测并裁剪扫描照片
"""
import cv2
import os
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'runs', 'photo_detector_v5', 'weights', 'best.pt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# 自动检测所有扫描图
SCAN_IMAGES = sorted([f for f in os.listdir(BASE_DIR) if f.startswith('Image_') and f.endswith('.jpg')])


def _nms_filter(detections, iou_thresh=0.4):
    """简单NMS过滤：去除IoU > thresh的重叠框，保留置信度高的"""
    if not detections:
        return detections
    # 已按面积排序，需要按置信度排序
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
            # 计算IoU
            x1 = max(best[0], d[0])
            y1 = max(best[1], d[1])
            x2 = min(best[2], d[2])
            y2 = min(best[3], d[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area_a = (best[2] - best[0]) * (best[3] - best[1])
            area_b = (d[2] - d[0]) * (d[3] - d[1])
            union = area_a + area_b - inter
            iou = inter / union if union > 0 else 0
            if iou < iou_thresh:
                remaining.append(d)
        dets = remaining
    # 按面积从大到小重新排序
    keep.sort(key=lambda d: d[5], reverse=True)
    return keep


def crop_photos():
    print('=' * 50)
    print('  YOLO26 照片检测 & 裁剪')
    print('=' * 50)

    # 检查模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        print("请先完成训练！")
        return

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型
    print(f'加载模型: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)

    total_cropped = 0

    for fname in SCAN_IMAGES:
        img_path = os.path.join(BASE_DIR, fname)
        if not os.path.exists(img_path):
            print(f"跳过 {fname}: 文件不存在")
            continue

        print(f'\n处理: {fname}')
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        print(f'  原始尺寸: {w}x{h}')

        # 推理检测
        results = model.predict(
            source=img,
            imgsz=1024,
            conf=0.25,       # 置信度阈值
            iou=0.3,         # NMS IoU 阈值（更低以去除重叠框）
            device='0',
            verbose=False,
            max_det=10,      # 最大检测数
        )

        result = results[0]
        boxes = result.boxes

        if len(boxes) == 0:
            print(f'  未检测到照片，尝试降低置信度...')
            # 降低阈值重试
            results = model.predict(
                source=img,
                imgsz=1024,
                conf=0.10,
                iou=0.3,
                device='0',
                verbose=False,
                max_det=10,
            )
            result = results[0]
            boxes = result.boxes

        num_detected = len(boxes)
        print(f'  检测到 {num_detected} 张照片')

        if num_detected == 0:
            print(f'  警告: 仍无法检测到照片，跳过')
            continue

        # 按面积从大到小排序（方便命名）
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            area = (x2 - x1) * (y2 - y1)
            detections.append((x1, y1, x2, y2, conf, area))

        detections.sort(key=lambda d: d[5], reverse=True)

        # 去除高度重叠的框（自定义NMS：IoU > 0.4 只保留置信度高的）
        filtered = _nms_filter(detections, iou_thresh=0.4)
        if len(filtered) < len(detections):
            print(f'  NMS过滤: {len(detections)} -> {len(filtered)} 个框')
        detections = filtered

        # 假阳性过滤: 过滤掉太窄/太小的检测框(可能是背景条纹)
        filtered_dets = []
        for (x1, y1, x2, y2, conf, area) in detections:
            bw = x2 - x1
            bh = y2 - y1
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            # 过滤条件: 长宽比>5的细长条 或 面积小于图片面积1%的微小框
            if aspect > 5:
                print(f'  过滤细长条: {bw}x{bh} (ratio={aspect:.1f})')
                continue
            if area < (w * h * 0.01):
                print(f'  过滤微小框: {bw}x{bh} (area={area})')
                continue
            filtered_dets.append((x1, y1, x2, y2, conf, area))
        if len(filtered_dets) < len(detections):
            print(f'  假阳性过滤: {len(detections)} -> {len(filtered_dets)} 个框')
        detections = filtered_dets

        # 裁剪并保存
        base_name = os.path.splitext(fname)[0]
        for i, (x1, y1, x2, y2, conf, area) in enumerate(detections):
            # 添加少量边距 (2%)
            margin_x = int((x2 - x1) * 0.02)
            margin_y = int((y2 - y1) * 0.02)
            cx1 = max(0, x1 - margin_x)
            cy1 = max(0, y1 - margin_y)
            cx2 = min(w, x2 + margin_x)
            cy2 = min(h, y2 + margin_y)

            cropped = img[cy1:cy2, cx1:cx2]
            
            out_name = f"{base_name}_photo{i+1}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            cw, ch = cropped.shape[1], cropped.shape[0]
            print(f'  [{i+1}] {out_name} ({cw}x{ch}, conf={conf:.2f})')
            total_cropped += 1

        # 保存检测结果可视化 (显示过滤后的框)
        annotated = img.copy()
        for (bx1, by1, bx2, by2, bconf, _area) in detections:
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
            label = f'photograph {bconf:.2f}'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(annotated, (bx1, by1 - th - 10), (bx1 + tw + 4, by1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (bx1 + 2, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        vis_path = os.path.join(OUTPUT_DIR, f'{base_name}_detect.jpg')
        cv2.imwrite(vis_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'  检测可视化: {base_name}_detect.jpg')

    print(f'\n{"=" * 50}')
    print(f'  裁剪完成！共输出 {total_cropped} 张照片')
    print(f'  输出目录: {OUTPUT_DIR}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    crop_photos()
