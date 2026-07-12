"""
扫描照片自动裁剪工具 v5
基于 YOLO26n 训练的照片检测模型，自动识别扫描图中的照片并裁剪输出。

用法:
    python crop_photos.py                          # 处理 input/ 目录中的所有图片
    python crop_photos.py --input ./my_scans       # 指定输入目录
    python crop_photos.py --input scan.jpg         # 处理单张图片
    python crop_photos.py --conf 0.15              # 调整检测置信度
"""
import cv2
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# 默认路径 (相对于脚本所在目录)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(SCRIPT_DIR, 'model', 'best.pt')
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, 'input')
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'output')


def nms_filter(detections, iou_thresh=0.4):
    """NMS 过滤：去除 IoU > thresh 的重叠框，保留置信度高的"""
    if not detections:
        return detections
    dets = sorted(detections, key=lambda d: d[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        remaining = []
        for d in dets:
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
    keep.sort(key=lambda d: d[5], reverse=True)
    return keep


def process_image(img_path, model, output_dir, conf_thresh=0.25, save_vis=True):
    """处理单张扫描图，返回裁剪的照片数量"""
    fname = os.path.basename(img_path)
    base_name = os.path.splitext(fname)[0]

    print(f'\n处理: {fname}')
    img = cv2.imread(img_path)
    if img is None:
        print(f'  错误: 无法读取 {img_path}')
        return 0

    h, w = img.shape[:2]
    print(f'  原始尺寸: {w}x{h}')

    # 推理检测
    results = model.predict(
        source=img, imgsz=1024, conf=conf_thresh,
        iou=0.3, device='0', verbose=False, max_det=10,
    )
    result = results[0]
    boxes = result.boxes

    # 首次未检测到则降低阈值重试
    if len(boxes) == 0:
        print(f'  未检测到照片，降低置信度重试...')
        results = model.predict(
            source=img, imgsz=1024, conf=0.10,
            iou=0.3, device='0', verbose=False, max_det=10,
        )
        result = results[0]
        boxes = result.boxes

    num_detected = len(boxes)
    print(f'  检测到 {num_detected} 张照片')
    if num_detected == 0:
        print(f'  警告: 无法检测到照片，跳过')
        return 0

    # 提取检测框
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        area = (x2 - x1) * (y2 - y1)
        detections.append((x1, y1, x2, y2, conf, area))

    detections.sort(key=lambda d: d[5], reverse=True)

    # NMS 过滤
    filtered = nms_filter(detections, iou_thresh=0.4)
    if len(filtered) < len(detections):
        print(f'  NMS过滤: {len(detections)} -> {len(filtered)} 个框')
    detections = filtered

    # 假阳性过滤
    filtered_dets = []
    for (x1, y1, x2, y2, conf, area) in detections:
        bw, bh = x2 - x1, y2 - y1
        aspect = max(bw, bh) / max(min(bw, bh), 1)
        if aspect > 5:
            print(f'  过滤细长条: {bw}x{bh}')
            continue
        if area < (w * h * 0.01):
            print(f'  过滤微小框: {bw}x{bh}')
            continue
        filtered_dets.append((x1, y1, x2, y2, conf, area))
    detections = filtered_dets

    # 裁剪并保存
    cropped_count = 0
    for i, (x1, y1, x2, y2, conf, area) in enumerate(detections):
        margin_x = int((x2 - x1) * 0.02)
        margin_y = int((y2 - y1) * 0.02)
        cx1, cy1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        cx2, cy2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
        cropped = img[cy1:cy2, cx1:cx2]

        out_name = f"{base_name}_photo{i+1}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cw, ch = cropped.shape[1], cropped.shape[0]
        print(f'  [{i+1}] {out_name} ({cw}x{ch}, conf={conf:.2f})')
        cropped_count += 1

    # 保存检测可视化
    if save_vis:
        annotated = result.plot()
        vis_path = os.path.join(output_dir, f'{base_name}_detect.jpg')
        cv2.imwrite(vis_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'  检测可视化: {base_name}_detect.jpg')

    return cropped_count


def main():
    parser = argparse.ArgumentParser(
        description='扫描照片自动裁剪工具 v5 - 基于 YOLO26n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python crop_photos.py                       处理 input/ 中的所有图片
  python crop_photos.py --input ./my_scans    指定输入目录
  python crop_photos.py --input scan.jpg      处理单张图片
  python crop_photos.py --conf 0.15           降低置信度提高召回率
  python crop_photos.py --no-vis              不生成检测可视化图
        """
    )
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT,
                        help='输入图片路径 (目录或单张图片), 默认: input/')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT,
                        help='输出目录, 默认: output/')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                        help='模型文件路径, 默认: model/best.pt')
    parser.add_argument('--conf', '-c', type=float, default=0.25,
                        help='检测置信度阈值 (默认 0.25, 降低可提高召回率)')
    parser.add_argument('--no-vis', action='store_true',
                        help='不生成检测可视化图片')
    args = parser.parse_args()

    print('=' * 50)
    print('  扫描照片自动裁剪工具 v5')
    print('  基于 YOLO26n 照片检测模型')
    print('=' * 50)

    # 检查模型
    if not os.path.exists(args.model):
        print(f"\n错误: 模型文件不存在: {args.model}")
        print("请确保 model/best.pt 文件存在。")
        sys.exit(1)

    # 收集输入文件
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    if os.path.isfile(args.input):
        img_files = [args.input]
    elif os.path.isdir(args.input):
        img_files = sorted([
            os.path.join(args.input, f) for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in img_exts
        ])
    else:
        print(f"\n错误: 输入路径不存在: {args.input}")
        sys.exit(1)

    if not img_files:
        print(f"\n警告: 未在 {args.input} 中找到图片文件")
        print("请将扫描图放入 input/ 目录，或使用 --input 指定路径")
        sys.exit(0)

    print(f'输入: {len(img_files)} 张图片')
    print(f'输出: {args.output}')
    print(f'模型: {args.model}')
    print(f'置信度: {args.conf}')

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    print(f'\n加载模型...')
    model = YOLO(args.model)

    # 处理所有图片
    total = 0
    for img_path in img_files:
        total += process_image(
            img_path, model, args.output,
            conf_thresh=args.conf, save_vis=not args.no_vis
        )

    print(f'\n{"=" * 50}')
    print(f'  裁剪完成! 共输出 {total} 张照片')
    print(f'  输出目录: {os.path.abspath(args.output)}')
    print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
