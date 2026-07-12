"""
准备 YOLO26 训练数据集 (v4 - 修正标注后重新训练)
- 修正图0033(2→4张), 图0010(5→3张), 图0019(4张拆分)
- 标注24张扫描图中的照片位置
- 加入good/文件夹中的137张单张照片作为训练数据
- 数据增强扩充样本
- 生成 dataset.yaml
"""
import cv2
import numpy as np
import os
import shutil
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = BASE_DIR
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
GOOD_DIR = os.path.join(BASE_DIR, 'good')

# === 标注数据 ===
# 格式: [(x1, y1, x2, y2), ...] 像素坐标
ANNOTATIONS = {
    # === 原有20张 (20260711) ===
    'Image_20260711_0001.jpg': [  # 4249x3067, 2张黑白肖像并排
        (30, 30, 2080, 3030),
        (2130, 30, 4220, 3030),
    ],
    'Image_20260711_0002.jpg': [  # 4634x2398, 2张黑白照并排
        (50, 10, 1980, 2380),
        (2050, 10, 4600, 2380),
    ],
    'Image_20260711_0003.jpg': [  # 4499x3117, 3张(左2叠+右1大)
        (30, 20, 1680, 1500),
        (30, 1560, 1680, 3090),
        (1780, 100, 4470, 2800),
    ],
    'Image_20260711_0004.jpg': [  # 4542x3268, 3张家庭照(人工修正)
        (53, 637, 2712, 2618),
        (2957, 40, 4407, 1568),
        (3092, 1568, 4240, 3247),
    ],
    'Image_20260711_0005.jpg': [  # 4242x2982, 3张(左2叠+右1大)
        (30, 20, 1580, 1400),
        (30, 1450, 1580, 2950),
        (1650, 20, 4210, 2950),
    ],
    'Image_20260711_0006.jpg': [  # 4337x2978, 7张(中1彩色+周6黑白)
        (30, 30, 900, 900),
        (30, 950, 900, 1850),
        (30, 1900, 900, 2950),
        (1000, 30, 3350, 2950),
        (3400, 30, 4300, 900),
        (3400, 950, 4300, 1850),
        (3400, 1900, 4300, 2950),
    ],
    'Image_20260711_0007.jpg': [  # 4447x3051, 8张(中1大+周7小)
        (30, 30, 900, 850),
        (1000, 30, 2000, 750),
        (3200, 30, 4100, 850),
        (30, 900, 1000, 1700),
        (1100, 800, 3400, 2300),
        (3500, 900, 4420, 1700),
        (30, 1800, 1200, 2700),
        (3000, 1800, 4420, 2700),
    ],
    'Image_20260711_0008.jpg': [  # 4527x2858, 6张 2x3网格
        (30, 30, 1450, 1350),
        (1500, 30, 2950, 1350),
        (3000, 30, 4500, 1350),
        (30, 1400, 1450, 2830),
        (1500, 1400, 2950, 2830),
        (3000, 1400, 4500, 2830),
    ],
    'Image_20260711_0009.jpg': [  # 4556x3307, 5张
        (30, 30, 1500, 1800),
        (1600, 30, 4520, 1500),
        (30, 1850, 1200, 2800),
        (1250, 1550, 2800, 2600),
        (2850, 1550, 4520, 3270),
    ],
    'Image_20260711_0010.jpg': [  # 4381x3037, 3张(左2+右1)
        (30, 30, 2050, 1650),     # 左上
        (30, 1700, 2050, 3000),   # 左下
        (2100, 30, 4350, 3000),   # 右侧大照
    ],
    'Image_20260711_0011.jpg': [  # 2961x2068, 1张彩色照(骆驼+长城)
        (50, 30, 2910, 2030),
    ],
    'Image_20260711_0012.jpg': [  # 4318x2557, 3张(左1大+右2小)
        (30, 30, 2300, 2520),
        (2400, 30, 4280, 1200),
        (2400, 1250, 4280, 2520),
    ],
    'Image_20260711_0013.jpg': [  # 2660x3707, 1张竖版黑白肖像
        (80, 30, 2580, 3670),
    ],
    'Image_20260711_0014.jpg': [  # 4637x2714, 5张
        (30, 30, 2200, 2680),
        (2300, 30, 3300, 900),
        (3350, 30, 4600, 900),
        (2300, 950, 3300, 1800),
        (2300, 1850, 4600, 2680),
    ],
    'Image_20260711_0015.jpg': [  # 3198x2100, 2张黑白肖像并排
        (30, 30, 1550, 2070),
        (1600, 30, 3170, 2070),
    ],
    'Image_20260711_0016.jpg': [  # 4257x2555, 3张(人工修正)
        (493, 35, 1202, 993),
        (28, 1348, 1784, 2511),
        (1901, 741, 4203, 2497),
    ],
    'Image_20260711_0017.jpg': [  # 3508x2374, 1张黑白合影(11人)
        (30, 30, 3480, 2340),
    ],
    'Image_20260711_0018.jpg': [  # 3381x2209, 1张大合影(民兵连)
        (30, 30, 3350, 2180),
    ],
    'Image_20260711_0019.jpg': [  # 4667x3568, 4张(人工修正)
        (53, 486, 695, 1409),
        (847, 490, 1467, 1400),
        (330, 1904, 1364, 3353),
        (1636, 80, 4642, 3505),
    ],
    'Image_20260711_0020.jpg': [  # 4678x1996, 2张大合影并排
        (30, 30, 2250, 1960),
        (2300, 30, 4650, 1960),
    ],
    # === 新增4张 (20251011) ===
    'Image_20251011_0033.jpg': [  # 6847x4871, 4张2x2网格家庭照
        (80, 50, 3350, 2350),      # 左上: 麻将桌合影
        (3500, 50, 6770, 2350),    # 右上: 餐桌家庭
        (80, 2450, 3350, 4820),    # 左下: 户外母子
        (3500, 2450, 6770, 4820),  # 右下: 红背景祖孙
    ],
    'Image_20251011_0034.jpg': [  # 6728x4952, 4张2x2网格儿童照
        (150, 150, 3200, 2450),    # 左上: 幼儿肖像
        (3400, 150, 6600, 2450),   # 右上: 婴儿藤椅
        (150, 2550, 3200, 4850),   # 左下: 花衬衫幼儿
        (3400, 2550, 6600, 4850),  # 右下: 红帽婴儿
    ],
    'Image_20251011_0035.jpg': [  # 6816x4920, 4张2x2网格儿童照
        (100, 100, 3400, 2400),    # 左上: 蝴蝶道具
        (3500, 100, 6700, 2400),   # 右上: 两岁小孩骑车
        (100, 2500, 3400, 4850),   # 左下: 月亮道具女孩
        (3500, 2500, 6700, 4850),  # 右下: 地板上的幼儿
    ],
    'Image_20251011_0119.jpg': [  # 2799x6000, 1张竖版黑白肖像
        (80, 80, 2720, 5920),
    ],
}

# 训练集扫描图 (19张)
TRAIN_SCAN_FILES = [
    'Image_20260711_0001.jpg', 'Image_20260711_0002.jpg',
    'Image_20260711_0003.jpg', 'Image_20260711_0005.jpg',
    'Image_20260711_0006.jpg', 'Image_20260711_0008.jpg',
    'Image_20260711_0009.jpg', 'Image_20260711_0010.jpg',
    'Image_20260711_0011.jpg', 'Image_20260711_0013.jpg',
    'Image_20260711_0015.jpg', 'Image_20260711_0017.jpg',
    'Image_20260711_0018.jpg', 'Image_20260711_0020.jpg',
    'Image_20260711_0016.jpg',
    'Image_20260711_0019.jpg',  # 从验证集移入(含漏检小照片)
    'Image_20260711_0004.jpg',  # 从验证集移入(人工确认标注)
    # 新增
    'Image_20251011_0033.jpg', 'Image_20251011_0034.jpg',
    'Image_20251011_0035.jpg',
]

# 验证集扫描图 (3张)
VAL_SCAN_FILES = [
    'Image_20260711_0007.jpg',
    'Image_20260711_0012.jpg', 'Image_20260711_0014.jpg',
]


def create_dataset():
    """创建 YOLO 数据集"""
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, split, 'labels'), exist_ok=True)

    def boxes_to_yolo(boxes, w, h):
        labels = []
        for (x1, y1, x2, y2) in boxes:
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        return labels

    # === 1. 处理扫描图 (训练集) ===
    print("=== 处理扫描图训练集 ===")
    scan_train_count = 0
    for fname in TRAIN_SCAN_FILES:
        img_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(img_path):
            print(f"  跳过: {fname} 不存在")
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = ANNOTATIONS[fname]
        labels = boxes_to_yolo(boxes, w, h)

        # 保存原图
        base = os.path.splitext(fname)[0]
        dst_img = os.path.join(DATASET_DIR, 'train', 'images', fname)
        shutil.copy2(img_path, dst_img)
        dst_label = os.path.join(DATASET_DIR, 'train', 'labels', base + '.txt')
        with open(dst_label, 'w') as f:
            f.write('\n'.join(labels))
        scan_train_count += 1

        # 6种增强
        for aug_id, aug_fn in enumerate([
            dict(flip_h=True), dict(flip_v=True),
            dict(brightness=1.3), dict(brightness=0.7),
            dict(rotate=5), dict(noise=True),
        ]):
            aug_name = f"{base}_aug{aug_id:02d}"
            _save_augmented(img, boxes, w, h, 'train', aug_name, **aug_fn)

    print(f"  扫描图训练: {scan_train_count} 张")

    # === 2. 处理扫描图 (验证集) ===
    print("=== 处理扫描图验证集 ===")
    scan_val_count = 0
    for fname in VAL_SCAN_FILES:
        img_path = os.path.join(INPUT_DIR, fname)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        boxes = ANNOTATIONS[fname]
        labels = boxes_to_yolo(boxes, w, h)

        base = os.path.splitext(fname)[0]
        dst_img = os.path.join(DATASET_DIR, 'val', 'images', fname)
        shutil.copy2(img_path, dst_img)
        dst_label = os.path.join(DATASET_DIR, 'val', 'labels', base + '.txt')
        with open(dst_label, 'w') as f:
            f.write('\n'.join(labels))
        scan_val_count += 1

        # 2种增强
        for aug_id, aug_fn in enumerate([
            dict(flip_h=True), dict(brightness=1.2),
        ]):
            aug_name = f"{base}_aug{aug_id:02d}"
            _save_augmented(img, boxes, w, h, 'val', aug_name, **aug_fn)

    print(f"  扫描图验证: {scan_val_count} 张")

    # === 3. 加入good/文件夹的单张照片 (训练集) ===
    print("=== 处理good/文件夹单张照片 ===")
    good_count = 0
    good_files = sorted([f for f in os.listdir(GOOD_DIR) if f.lower().endswith('.jpg')])

    for fname in good_files:
        img_path = os.path.join(GOOD_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        # 每张照片本身就是1个photo, bbox=整图(留5px边距)
        m = min(5, w // 10, h // 10)
        boxes = [(m, m, w - m, h - m)]
        labels = boxes_to_yolo(boxes, w, h)

        # 使用唯一前缀避免文件名冲突
        good_fname = f"good_{fname}"
        base = os.path.splitext(good_fname)[0]

        # 保存原图
        dst_img = os.path.join(DATASET_DIR, 'train', 'images', good_fname)
        cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dst_label = os.path.join(DATASET_DIR, 'train', 'labels', base + '.txt')
        with open(dst_label, 'w') as f:
            f.write('\n'.join(labels))
        good_count += 1

        # 3种增强 (翻转、亮度、旋转)
        for aug_id, aug_fn in enumerate([
            dict(flip_h=True), dict(brightness=1.2), dict(rotate=3),
        ]):
            aug_name = f"{base}_aug{aug_id:02d}"
            _save_augmented(img, boxes, w, h, 'train', aug_name, **aug_fn)

    print(f"  good照片训练: {good_count} 张 (含 {good_count * 3} 张增强)")

    # === 4. 创建 dataset.yaml ===
    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 1\n")
        f.write("names: ['photograph']\n")

    train_imgs = len(os.listdir(os.path.join(DATASET_DIR, 'train', 'images')))
    val_imgs = len(os.listdir(os.path.join(DATASET_DIR, 'val', 'images')))
    print(f"\n数据集创建完成!")
    print(f"  训练集: {train_imgs} 张")
    print(f"  验证集: {val_imgs} 张")


def _save_augmented(img, boxes, w, h, split, aug_name,
                    flip_h=False, flip_v=False, brightness=None,
                    rotate=None, noise=False):
    img = img.copy()
    boxes = list(boxes)

    if flip_h:
        img = cv2.flip(img, 1)
        boxes = [(w - x2, y1, w - x1, y2) for (x1, y1, x2, y2) in boxes]

    if flip_v:
        img = cv2.flip(img, 0)
        boxes = [(x1, h - y2, x2, h - y1) for (x1, y1, x2, y2) in boxes]

    if brightness:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rotate:
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotate, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(180, 160, 140))
        new_boxes = []
        for (x1, y1, x2, y2) in boxes:
            corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            ones = np.ones((4, 1), dtype=np.float32)
            corners_h = np.hstack([corners, ones])
            transformed = M.dot(corners_h.T).T
            nx1 = max(0, int(np.min(transformed[:, 0])))
            ny1 = max(0, int(np.min(transformed[:, 1])))
            nx2 = min(w, int(np.max(transformed[:, 0])))
            ny2 = min(h, int(np.max(transformed[:, 1])))
            if nx2 - nx1 > 50 and ny2 - ny1 > 50:
                new_boxes.append((nx1, ny1, nx2, ny2))
        boxes = new_boxes

    if noise:
        noise_arr = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise_arr, 0, 255).astype(np.uint8)

    yolo_labels = []
    for (x1, y1, x2, y2) in boxes:
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    dst_img = os.path.join(DATASET_DIR, split, 'images', aug_name + '.jpg')
    cv2.imwrite(dst_img, img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    dst_label = os.path.join(DATASET_DIR, split, 'labels', aug_name + '.txt')
    with open(dst_label, 'w') as f:
        f.write('\n'.join(yolo_labels))


if __name__ == '__main__':
    create_dataset()
