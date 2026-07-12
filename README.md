# Scan Photo Cropper

基于 YOLO26n 的扫描照片自动裁剪工具。自动识别扫描图中的照片并裁剪为单独的高清图片。

## 功能特点

- **自动检测**: 基于 YOLO26n 深度学习模型，自动识别扫描图中的每张照片
- **智能过滤**: NMS 去重 + 假阳性过滤（细长条、微小框）
- **自适应重试**: 首次未检出时自动降低阈值重试
- **可视化输出**: 生成带检测框的可视化图片，方便验证结果
- **灵活输入**: 支持单张图片或批量目录处理

## 模型信息

| 指标 | 值 |
|------|-----|
| 模型架构 | YOLO26n (2.5M params, 5.8 GFLOPs) |
| mAP50 | 0.351 |
| Precision | 0.525 |
| Recall | 0.500 |
| 训练数据 | 688 张 (20 张扫描图 × 增强 + 137 张标注照片) |
| 训练环境 | RTX 3080, CUDA 12.6, ultralytics 8.4.92 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备模型

将训练好的模型 `best.pt` 放入 `model/` 目录。

### 3. 运行裁剪

```bash
# 使用发布版工具（推荐）
cd release/PhotoCropper
python crop_photos.py --input ./input

# 或直接使用源码
python cropWithYOLO.py
```

## 项目结构

```
scan-photo-cropper/
├── prepareDataset.py     # 数据集准备（标注 + 增强 + YAML 生成）
├── trainYOLO.py          # YOLO26n 训练脚本
├── cropWithYOLO.py       # 推理裁剪脚本
├── requirements.txt      # Python 依赖
├── .gitignore
├── release/
│   └── PhotoCropper/     # 发布版工具（含模型 + 脚本 + 批处理）
└── README.md
```

## 核心脚本说明

### prepareDataset.py

准备 YOLO 训练数据集：
- 定义 24 张扫描图的照片位置标注（像素坐标）
- 数据增强（翻转、亮度、旋转、噪声）
- 合并 good/ 目录中的 137 张单张照片
- 生成 dataset.yaml

### trainYOLO.py

训练 YOLO26n 检测模型：
- 基础模型: yolo26n.pt (预训练)
- 训练参数: epochs=200, patience=50, batch=8, imgsz=1024
- 优化器: AdamW (auto)
- 支持 GPU 加速 (CUDA)

### cropWithYOLO.py

推理裁剪流程：
1. 加载训练好的模型
2. 对每张扫描图进行目标检测
3. NMS 过滤重叠框 (IoU > 0.4)
4. 假阳性过滤 (长宽比 > 5 或面积 < 1%)
5. 裁剪并保存为单独图片
6. 生成检测可视化图

## 版本迭代记录

| 版本 | mAP50 | Precision | Recall | 主要改进 |
|------|-------|-----------|--------|----------|
| v1 | 0.158 | - | - | 基础版本 |
| v2 | 0.287 | - | - | 增加训练数据 |
| v3 | 0.311 | - | - | 更多增强 |
| v4 | 0.329 | 0.498 | 0.374 | 修正标注，修复 6/7 问题 |
| **v5** | **0.351** | **0.525** | **0.500** | **修正坐标，移入训练集，修复 7/7 问题** |

## License

MIT
