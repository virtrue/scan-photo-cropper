"""
扫描照片自动切割工具 - AI版 (rembg U2-Net)
使用 AI 模型移除背景，然后检测并裁剪每张照片
"""
import cv2
import numpy as np
import os
import sys
from rembg import remove
from PIL import Image
import io

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(INPUT_DIR, 'output')
MIN_AREA_RATIO = 0.03       # Min 3% of image area
MAX_AREA_RATIO = 0.90       # Max 90%
CROP_PADDING = 20           # Pixels padding around crop


def ai_detect_photos(image_path):
    """Use rembg (U2-Net AI) to detect photo regions by removing background."""
    print(f"  AI 背景移除中...")
    
    # Read image
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    # Use rembg to remove background -> returns RGBA PNG
    result_bytes = remove(img_bytes)
    
    # Convert to OpenCV format (RGBA)
    nparr = np.frombuffer(result_bytes, np.uint8)
    result_rgba = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if result_rgba is None:
        print("  AI 处理失败")
        return [], None
    
    # Extract alpha channel (foreground mask)
    alpha = result_rgba[:, :, 3]
    
    # Threshold to binary mask
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    # Try to separate adjacent photos using erosion
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    eroded = cv2.erode(mask, kernel_erode, iterations=2)
    
    # Dilate back
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    separated = cv2.dilate(eroded, kernel_dilate, iterations=2)
    
    h, w = mask.shape
    total_area = w * h
    
    # Find contours on separated mask
    contours, _ = cv2.findContours(separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    min_area = total_area * MIN_AREA_RATIO
    max_area = total_area * MAX_AREA_RATIO
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        
        x, y, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / rh if rh > 0 else 0
        if aspect < 0.3 or aspect > 3.0:
            continue
        
        fill_ratio = area / (rw * rh) if rw * rh > 0 else 0
        if fill_ratio < 0.4:
            continue
        
        regions.append({
            'x': x, 'y': y, 'w': rw, 'h': rh,
            'area': area, 'aspect': aspect, 'fill': fill_ratio
        })
    
    # If separated mask found too few regions, try using original mask + projection split
    if len(regions) <= 1:
        print(f"  检测到 {len(regions)} 个区域，尝试使用原始掩码+投影分割...")
        regions = _try_projection_split(mask, total_area)
    
    # If still only 1 region, try using eroded mask directly (before dilate)
    if len(regions) <= 1:
        print(f"  尝试使用腐蚀掩码...")
        contours2, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions2 = []
        for cnt in contours2:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            aspect = rw / rh if rh > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue
            fill_ratio = area / (rw * rh) if rw * rh > 0 else 0
            if fill_ratio < 0.3:
                continue
            # Scale eroded bbox back up slightly
            pad = 20
            x = max(0, x - pad)
            y = max(0, y - pad)
            rw = min(w - x, rw + 2 * pad)
            rh = min(h - y, rh + 2 * pad)
            regions2.append({
                'x': x, 'y': y, 'w': rw, 'h': rh,
                'area': area, 'aspect': aspect, 'fill': fill_ratio
            })
        if len(regions2) > len(regions):
            regions = regions2
    
    # Sort by position (top-to-bottom, left-to-right)
    regions.sort(key=lambda r: (r['y'] // (h // 4), r['x']))
    
    return regions, mask


def _try_projection_split(mask, total_area):
    """Try to split a single large region using projection analysis on the mask."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return []
    
    x0, y0, rw, rh = cv2.boundingRect(coords)
    sub_mask = mask[y0:y0+rh, x0:x0+rw]
    
    # Compute mask projections
    proj_x = np.sum(sub_mask > 0, axis=0).astype(float) / rh
    proj_y = np.sum(sub_mask > 0, axis=1).astype(float) / rw
    
    min_size_ratio = 0.20
    
    # Try vertical split
    vsplit = _find_valley(proj_x)
    if vsplit is not None:
        left_w = vsplit
        right_w = rw - vsplit
        if left_w / rw >= min_size_ratio and right_w / rw >= min_size_ratio:
            print(f"  投影分割: 垂直切割于 x={vsplit}")
            return _make_regions([(0, 0, left_w, rh), (vsplit, 0, right_w, rh)], 
                               x0, y0, rw, rh, total_area)
    
    # Try horizontal split
    hsplit = _find_valley(proj_y)
    if hsplit is not None:
        top_h = hsplit
        bot_h = rh - hsplit
        if top_h / rh >= min_size_ratio and bot_h / rh >= min_size_ratio:
            print(f"  投影分割: 水平切割于 y={hsplit}")
            return _make_regions([(0, 0, rw, top_h), (0, hsplit, rw, bot_h)],
                               x0, y0, rw, rh, total_area)
    
    # No split found, return the whole region
    return [{
        'x': x0, 'y': y0, 'w': rw, 'h': rh,
        'area': rw * rh, 'aspect': rw / rh if rh > 0 else 0, 'fill': 0.5
    }]


def _make_regions(sub_regions, x0, y0, rw, rh, total_area):
    """Convert sub-region coordinates to full image coordinates."""
    regions = []
    min_area = total_area * MIN_AREA_RATIO
    for sx, sy, sw, sh in sub_regions:
        ax = x0 + sx
        ay = y0 + sy
        area = sw * sh
        if area < min_area:
            continue
        aspect = sw / sh if sh > 0 else 0
        regions.append({
            'x': ax, 'y': ay, 'w': sw, 'h': sh,
            'area': area, 'aspect': aspect, 'fill': 0.5
        })
    return regions


def _find_valley(proj):
    """Find the deepest valley in a 1D projection."""
    length = len(proj)
    if length < 40:
        return None
    
    kernel_size = max(7, length // 20)
    smoothed = np.convolve(proj, np.ones(kernel_size) / kernel_size, mode='same')
    
    margin = int(length * 0.10)
    center = smoothed[margin:length - margin]
    if len(center) < 10:
        return None
    
    min_idx = np.argmin(center) + margin
    min_val = smoothed[min_idx]
    
    left_max = np.max(smoothed[margin:min_idx]) if min_idx > margin + 5 else np.max(smoothed[:max(1, margin)])
    right_max = np.max(smoothed[min_idx + 1:length - margin]) if min_idx < length - margin - 5 else np.max(smoothed[length - margin:])
    
    reference = min(left_max, right_max)
    depth = reference - min_val
    
    if depth > reference * 0.12 and min_val > 0 and margin + 5 < min_idx < length - margin - 5:
        return int(min_idx)
    
    return None


def main():
    print('=' * 50)
    print('  扫描照片自动切割工具 (AI - U2-Net)')
    print(f'  输入目录: {INPUT_DIR}')
    print(f'  输出目录: {OUTPUT_DIR}')
    print('=' * 50)
    
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                os.unlink(fp)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = sorted([f for f in os.listdir(INPUT_DIR)
                    if f.lower().endswith(('.jpg', '.jpeg'))])
    
    if not files:
        print('未找到 JPG 图片文件。')
        return
    
    print(f'找到 {len(files)} 张扫描图片')
    
    total_crops = 0
    for fname in files:
        fpath = os.path.join(INPUT_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        
        print(f'\n处理: {fname}')
        
        try:
            img = cv2.imread(fpath)
            h, w = img.shape[:2]
            print(f'  原始尺寸: {w}x{h}')
            
            regions, ai_mask = ai_detect_photos(fpath)
            
            if not regions:
                print('  未检测到有效照片区域')
                continue
            
            print(f'  检测到 {len(regions)} 个照片区域')
            
            crop_count = 0
            for i, r in enumerate(regions):
                rx, ry, rw, rh = r['x'], r['y'], r['w'], r['h']
                
                # Apply padding
                cx = max(0, rx - CROP_PADDING)
                cy = max(0, ry - CROP_PADDING)
                cw = min(w - cx, rw + 2 * CROP_PADDING)
                ch = min(h - cy, rh + 2 * CROP_PADDING)
                
                if cw < 100 or ch < 100:
                    continue
                
                crop_count += 1
                cropped = img[cy:cy+ch, cx:cx+cw]
                output_file = os.path.join(OUTPUT_DIR, f"{base_name}_crop{crop_count}.jpg")
                cv2.imwrite(output_file, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                aspect = cw / ch if ch > 0 else 0
                print(f'  [{crop_count}] {cw}x{ch} (比例 {aspect:.2f}) -> {os.path.basename(output_file)}')
            
            # Save AI mask for debugging
            if ai_mask is not None:
                mask_file = os.path.join(OUTPUT_DIR, f"{base_name}_mask.jpg")
                cv2.imwrite(mask_file, ai_mask)
            
            total_crops += crop_count
            print(f'  共切割出 {crop_count} 张照片')
            
        except Exception as e:
            print(f'  处理出错: {e}')
            import traceback
            traceback.print_exc()
    
    print(f'\n{"=" * 50}')
    print(f'处理完成！共切割出 {total_crops} 张照片')
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 50)


if __name__ == '__main__':
    main()
