import cv2
import numpy as np
import os
import sys

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(INPUT_DIR, 'output')
MIN_AREA_RATIO = 0.04       # Minimum 4% of image area for a photo
MAX_AREA_RATIO = 0.95       # Maximum 95% (avoid detecting entire image)
ASPECT_MIN = 0.3            # Min aspect ratio (width/height)
ASPECT_MAX = 3.0            # Max aspect ratio
CROP_PADDING = 15           # Padding around detected region in original pixels

def projection_split(gray_img, mask_img=None, min_size_ratio=0.25):
    """Split a region using mask projection + gradient projection.
    mask_img: binary mask where 255=foreground, 0=background.
    Returns list of sub-regions [(x,y,w,h), ...] or empty."""
    h, w = gray_img.shape
    if w < 50 or h < 50:
        return []

    # --- Mask-based projection ---
    if mask_img is not None:
        mask_proj_x = np.sum(mask_img > 0, axis=0).astype(float) / h
        mask_proj_y = np.sum(mask_img > 0, axis=1).astype(float) / w
    else:
        mask_proj_x = None
        mask_proj_y = None

    # --- Gradient-based projection with multiple kernel sizes ---
    splits = []
    
    for ksize in [3, 7, 11]:
        gx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
        grad = np.sqrt(gx**2 + gy**2)
        grad = cv2.GaussianBlur(grad, (5, 5), 0)
        grad_proj_x = np.mean(grad, axis=0)
        grad_proj_y = np.mean(grad, axis=1)

        # Try vertical split
        vsplit = _find_valley_combined(grad_proj_x, mask_proj_x, w)
        if vsplit is not None:
            left_w = vsplit
            right_w = w - vsplit
            if left_w / w >= min_size_ratio and right_w / w >= min_size_ratio:
                splits.append((0, 0, left_w, h))
                splits.append((vsplit, 0, right_w, h))
                return splits

        # Try horizontal split
        hsplit = _find_valley_combined(grad_proj_y, mask_proj_y, h)
        if hsplit is not None:
            top_h = hsplit
            bot_h = h - hsplit
            if top_h / h >= min_size_ratio and bot_h / h >= min_size_ratio:
                splits.append((0, 0, w, top_h))
                splits.append((0, hsplit, w, bot_h))
                return splits

    return []


def _find_valley_combined(grad_proj, mask_proj, length):
    """Find valley using both gradient and mask projections."""
    # Try mask projection first (usually cleaner signal)
    if mask_proj is not None:
        result = _find_valley(mask_proj, length)
        if result is not None:
            return result
    # Fall back to gradient projection
    return _find_valley(grad_proj, length)


def _find_valley(proj, length):
    """Find the deepest valley using local peak analysis.
    Returns the split position or None."""
    if length < 40:
        return None

    # Smooth projection
    kernel_size = max(7, length // 20)
    smoothed = np.convolve(proj, np.ones(kernel_size)/kernel_size, mode='same')

    # Skip edges (10% on each side)
    margin = int(length * 0.10)
    center = smoothed[margin:length - margin]
    if len(center) < 10:
        return None

    # Find the global minimum in center
    min_idx = np.argmin(center) + margin
    min_val = smoothed[min_idx]

    # Find the maximum to the left of min
    left_max = np.max(smoothed[margin:min_idx]) if min_idx > margin + 5 else np.max(smoothed[:margin])
    # Find the maximum to the right of min
    right_max = np.max(smoothed[min_idx+1:length-margin]) if min_idx < length - margin - 5 else np.max(smoothed[length-margin:])

    # Valley depth = how far below the lower of the two peaks
    reference = min(left_max, right_max)
    depth = reference - min_val

    # The valley must have meaningful depth (at least 15% of reference value)
    if depth > reference * 0.15 and min_val > 0:
        # Also check it's not at the very edge
        if margin + 5 < min_idx < length - margin - 5:
            return int(min_idx)

    return None


def detect_photos(image_path):
    """Detect individual photo regions in a scanned image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  无法读取: {image_path}")
        return []
    
    h, w = img.shape[:2]
    print(f"  原始尺寸: {w}x{h}")
    
    # Resize for analysis (keep reasonable resolution)
    analysis_width = 800
    scale = analysis_width / w
    analysis_h = int(h * scale)
    small = cv2.resize(img, (analysis_width, analysis_h))
    
    # Convert to different color spaces
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    
    # --- Method: Combined saturation + value masking ---
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    
    # Create mask: photo regions have low saturation
    sat_thresh = 35
    val_thresh = 50
    
    mask = np.zeros((analysis_h, analysis_width), dtype=np.uint8)
    mask[(sat < sat_thresh) & (val > val_thresh)] = 255
    
    # --- Morphological operations ---
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.erode(mask, kernel_erode, iterations=2)
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    
    # --- Create a separate mask for projection splitting (stricter threshold = cleaner gaps) ---
    strict_mask = np.zeros((analysis_h, analysis_width), dtype=np.uint8)
    strict_mask[(sat < 20) & (val > 60)] = 255  # Much stricter: only core photo content
    # Light morphology for strict mask
    k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, k_small, iterations=1)
    k_med = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, k_med, iterations=1)
    split_mask = strict_mask
    
    # --- Find contours on the dilated mask ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours
    total_area = analysis_width * analysis_h
    min_area = total_area * MIN_AREA_RATIO
    max_area = total_area * MAX_AREA_RATIO
    
    raw_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        
        x, y, rw, rh = cv2.boundingRect(cnt)
        aspect = rw / rh if rh > 0 else 0
        if aspect < ASPECT_MIN or aspect > ASPECT_MAX:
            continue
        
        fill_ratio = area / (rw * rh) if rw * rh > 0 else 0
        if fill_ratio < 0.5:
            continue
        
        raw_regions.append({
            'x': x, 'y': y, 'w': rw, 'h': rh,
            'area': area, 'fill': fill_ratio, 'aspect': aspect
        })
    
    # Sort by position
    raw_regions.sort(key=lambda r: (r['y'] // (analysis_h // 4), r['x']))
    
    # --- Second pass: try to split large regions using projection ---
    regions = []
    for reg in raw_regions:
        rx, ry, rw, rh = reg['x'], reg['y'], reg['w'], reg['h']
        
        # Extract the sub-region from analysis-size images
        sub_gray = gray[ry:ry+rh, rx:rx+rw]
        sub_mask = split_mask[ry:ry+rh, rx:rx+rw]
        
        # Try projection-based split using both mask and gradient
        sub_splits = projection_split(sub_gray, sub_mask)
        
        if sub_splits:
            # Validate sub-regions: reject if any sub-region has bad aspect ratio
            valid_splits = []
            for sx, sy, sw, sh in sub_splits:
                sub_aspect = sw / sh if sh > 0 else 0
                if 0.4 <= sub_aspect <= 2.5:
                    valid_splits.append((sx, sy, sw, sh))
            
            if len(valid_splits) >= 2:
                print(f"  区域 ({rw}x{rh}) 二次分割为 {len(valid_splits)} 个子区域")
                for sx, sy, sw, sh in valid_splits:
                    orig_x = max(0, int((rx + sx) / scale) - CROP_PADDING)
                    orig_y = max(0, int((ry + sy) / scale) - CROP_PADDING)
                    orig_w = min(w - orig_x, int(sw / scale) + CROP_PADDING * 2)
                    orig_h = min(h - orig_y, int(sh / scale) + CROP_PADDING * 2)
                    sub_aspect = sw / sh if sh > 0 else 0
                    regions.append({
                        'x': orig_x, 'y': orig_y,
                        'w': orig_w, 'h': orig_h,
                        'area': sw * sh, 'aspect': sub_aspect
                    })
            else:
                # Split rejected, use original region
                orig_x = max(0, int(rx / scale) - CROP_PADDING)
                orig_y = max(0, int(ry / scale) - CROP_PADDING)
                orig_w = min(w - orig_x, int(rw / scale) + CROP_PADDING * 2)
                orig_h = min(h - orig_y, int(rh / scale) + CROP_PADDING * 2)
                regions.append({
                    'x': orig_x, 'y': orig_y,
                    'w': orig_w, 'h': orig_h,
                    'area': reg['area'], 'aspect': reg['aspect']
                })
        else:
            # Scale back to original
            orig_x = max(0, int(rx / scale) - CROP_PADDING)
            orig_y = max(0, int(ry / scale) - CROP_PADDING)
            orig_w = min(w - orig_x, int(rw / scale) + CROP_PADDING * 2)
            orig_h = min(h - orig_y, int(rh / scale) + CROP_PADDING * 2)
            regions.append({
                'x': orig_x, 'y': orig_y,
                'w': orig_w, 'h': orig_h,
                'area': reg['area'], 'aspect': reg['aspect']
            })
    
    # Re-sort final regions
    regions.sort(key=lambda r: (r['y'] // (h // 4), r['x']))
    
    return regions


def refine_and_crop(image_path, regions, base_name):
    """Refine detected regions and crop them from the original image."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    crop_count = 0
    for i, r in enumerate(regions):
        x, y, rw, rh = r['x'], r['y'], r['w'], r['h']
        
        if rw < 100 or rh < 100:
            continue
        
        # Try to refine boundary using edge detection on the sub-region
        sub = img[y:y+rh, x:x+rw]
        sub_gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        
        # Use Canny to find edges
        edges = cv2.Canny(sub_gray, 30, 100)
        
        # Dilate edges to connect
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find the largest contour in edges
        edge_contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if edge_contours:
            # Find the bounding box of all significant edge pixels
            all_points = np.vstack(edge_contours)
            ex, ey, ew, eh = cv2.boundingRect(all_points)
            
            # Only use refined bounds if significantly tighter (>10% reduction)
            if ew > rw * 0.3 and eh > rh * 0.3:
                # Add small padding
                pad = 10
                rx = max(0, x + ex - pad)
                ry = max(0, y + ey - pad)
                rrw = min(w - rx, ew + pad * 2)
                rrh = min(h - ry, eh + pad * 2)
                
                if rrw > 100 and rrh > 100:
                    x, y, rw, rh = rx, ry, rrw, rrh
        
        crop_count += 1
        cropped = img[y:y+rh, x:x+rw]
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}_crop{crop_count}.jpg")
        cv2.imwrite(output_file, cropped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        aspect = rw / rh if rh > 0 else 0
        print(f"  [{crop_count}] {rw}x{rh} (比例 {aspect:.2f}) -> {os.path.basename(output_file)}")
    
    return crop_count


def main():
    print('=' * 50)
    print('  扫描照片自动切割工具 (OpenCV)')
    print(f'  输入目录: {INPUT_DIR}')
    print(f'  输出目录: {OUTPUT_DIR}')
    print('=' * 50)
    
    # Clean output dir
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            fp = os.path.join(OUTPUT_DIR, f)
            if os.path.isfile(fp):
                os.unlink(fp)
    
    # Find image files
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
            regions = detect_photos(fpath)
            print(f'  检测到 {len(regions)} 个有效区域')
            
            if regions:
                count = refine_and_crop(fpath, regions, base_name)
                total_crops += count
                print(f'  共切割出 {count} 张照片')
            else:
                # Fallback: try with relaxed parameters
                print('  尝试使用宽松参数重新检测...')
                # Save the whole image as fallback
                img = cv2.imread(fpath)
                out_file = os.path.join(OUTPUT_DIR, f"{base_name}_full.jpg")
                cv2.imwrite(out_file, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f'  已保存完整图像: {os.path.basename(out_file)}')
        except Exception as e:
            print(f'  处理出错: {e}')
    
    print(f'\n{"=" * 50}')
    print(f'处理完成！共切割出 {total_crops} 张照片')
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 50)


if __name__ == '__main__':
    main()
