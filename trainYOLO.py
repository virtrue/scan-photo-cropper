"""
用 YOLO26n 微调训练照片检测模型
"""
import os
import sys

# 设置 UTF-8 编码
os.environ['PYTHONIOENCODING'] = 'utf-8'

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolo26n.pt')
DATASET_YAML = os.path.join(BASE_DIR, 'dataset', 'dataset.yaml')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

def train():
    print('=' * 50)
    print('  YOLO26n 照片检测模型训练 (v5)')
    print('=' * 50)
    
    # 加载预训练模型
    print(f'加载基础模型: {MODEL_PATH}')
    model = YOLO(MODEL_PATH)
    
    # 训练参数 (v5: 修正坐标+0019/0004移入训练集)
    print('开始训练... (v5: 修正坐标+增加训练样本)')
    results = model.train(
        data=DATASET_YAML,
        epochs=200,
        imgsz=1024,
        batch=8,
        name='photo_detector_v5',  # v5版本
        project=os.path.join(BASE_DIR, 'runs'),
        patience=50,              # 50轮无提升则早停
        optimizer='auto',
        lr0=0.005,
        lrf=0.001,
        # 数据增强（对小目标更友好）
        hsv_h=0.015,             # 色调增强
        hsv_s=0.4,               # 饱和度增强
        hsv_v=0.3,               # 亮度增强
        degrees=5,               # 旋转角度（小一点）
        translate=0.1,           # 平移
        scale=0.15,              # 缩放（小一点保护小目标）
        fliplr=0.5,              # 水平翻转概率
        flipud=0.3,              # 垂直翻转概率
        mosaic=0.3,              # Mosaic增强（降低，保护小目标）
        mixup=0.05,              # MixUp（降低）
        # 保存
        save=True,
        save_period=10,
        plots=True,
        device='0',              # 使用GPU (RTX 3080)
        workers=2,               # Windows兼容，少量workers
        amp=False,               # 关闭AMP(避免网络下载)
        verbose=True,
    )
    
    # 获取最佳模型路径
    best_model = os.path.join(BASE_DIR, 'runs', 'photo_detector_v5', 'weights', 'best.pt')
    print(f'\n训练完成！最佳模型: {best_model}')
    
    return best_model


if __name__ == '__main__':
    best = train()
