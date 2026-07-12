const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

const INPUT_DIR = __dirname;
const OUTPUT_DIR = path.join(__dirname, 'output');
const ANALYSIS_WIDTH = 600;
const MIN_PHOTO_RATIO = 0.08; // A single photo must be at least 8% of total area
const GAP_DEPTH_RATIO = 0.4; // Valley must dip below 40% of avg to be a gap
const MIN_GAP_WIDTH = 3; // Min gap width in analysis pixels
const REGION_PADDING = 8;

// --- Compute gradient magnitude ---
function computeGradient(gray, w, h) {
  const grad = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx = -gray[(y-1)*w+x-1] + gray[(y-1)*w+x+1]
               -2*gray[y*w+x-1]   + 2*gray[y*w+x+1]
               -gray[(y+1)*w+x-1] + gray[(y+1)*w+x+1];
      const gy = -gray[(y-1)*w+x-1] - 2*gray[(y-1)*w+x] - gray[(y-1)*w+x+1]
               +gray[(y+1)*w+x-1] + 2*gray[(y+1)*w+x] + gray[(y+1)*w+x+1];
      grad[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return grad;
}

// --- Horizontal projection (sum each column) ---
function projectX(grad, w, h) {
  const proj = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = 0; y < h; y++) sum += grad[y * w + x];
    proj[x] = sum / h;
  }
  return proj;
}

// --- Vertical projection (sum each row) ---
function projectY(grad, w, h) {
  const proj = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = 0; x < w; x++) sum += grad[y * w + x];
    proj[y] = sum / w;
  }
  return proj;
}

// --- Smooth a 1D array with box filter ---
function smooth1D(arr, radius) {
  const n = arr.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0, count = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(n - 1, i + radius); j++) {
      sum += arr[j]; count++;
    }
    out[i] = sum / count;
  }
  return out;
}

// --- Find gap positions in a 1D projection ---
function findGaps(proj, minGapWidth) {
  if (proj.length < 10) return [];

  // Smooth to reduce noise
  const smoothed = smooth1D(proj, 3);

  // Compute adaptive threshold: use a moving window to find local low points
  const windowSize = Math.max(20, Math.floor(proj.length * 0.08));
  const gaps = [];
  let inGap = false;
  let gapStart = 0;

  for (let i = windowSize; i < proj.length - windowSize; i++) {
    // Compute local average in surrounding windows (excluding center)
    let leftSum = 0, rightSum = 0;
    for (let j = i - windowSize; j < i; j++) leftSum += smoothed[j];
    for (let j = i + 1; j <= i + windowSize; j++) rightSum += smoothed[j];
    const localAvg = (leftSum + rightSum) / (2 * windowSize);

    // A gap is where the projection is significantly below the local average
    const isLow = smoothed[i] < localAvg * GAP_DEPTH_RATIO;

    if (isLow && !inGap) {
      gapStart = i;
      inGap = true;
    } else if (!isLow && inGap) {
      if (i - gapStart >= minGapWidth) {
        gaps.push({ start: gapStart, end: i, mid: Math.floor((gapStart + i) / 2) });
      }
      inGap = false;
    }
  }
  // Close any open gap at end
  if (inGap && proj.length - gapStart >= minGapWidth) {
    gaps.push({ start: gapStart, end: proj.length, mid: Math.floor((gapStart + proj.length) / 2) });
  }

  return gaps;
}

// --- Recursive region splitting ---
function splitRegion(grad, fullW, fullH, rx, ry, rw, rh, depth, allRegions) {
  if (depth > 3) {
    allRegions.push({ x: rx, y: ry, w: rw, h: rh });
    return;
  }

  // Sub-gradient for this region
  const subGrad = new Float32Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    for (let x = 0; x < rw; x++) {
      subGrad[y * rw + x] = grad[(ry + y) * fullW + (rx + x)];
    }
  }

  // Try horizontal split first (vertical projection → find column gaps)
  const projX = projectX(subGrad, rw, rh);
  const gapsX = findGaps(projX, MIN_GAP_WIDTH);

  if (gapsX.length > 0) {
    // Split vertically at each gap midpoint
    let cuts = [0];
    for (const gap of gapsX) cuts.push(gap.mid);
    cuts.push(rw);

    for (let i = 0; i < cuts.length - 1; i++) {
      const subX = rx + cuts[i];
      const subW = cuts[i + 1] - cuts[i];
      if (subW > 20) {
        splitRegion(grad, fullW, fullH, subX, ry, subW, rh, depth + 1, allRegions);
      }
    }
    return;
  }

  // Try vertical split (horizontal projection → find row gaps)
  const projY = projectY(subGrad, rw, rh);
  const gapsY = findGaps(projY, MIN_GAP_WIDTH);

  if (gapsY.length > 0) {
    let cuts = [0];
    for (const gap of gapsY) cuts.push(gap.mid);
    cuts.push(rh);

    for (let i = 0; i < cuts.length - 1; i++) {
      const subY = ry + cuts[i];
      const subH = cuts[i + 1] - cuts[i];
      if (subH > 20) {
        splitRegion(grad, fullW, fullH, rx, subY, rw, subH, depth + 1, allRegions);
      }
    }
    return;
  }

  // No gaps found, this is a leaf region (one photo)
  allRegions.push({ x: rx, y: ry, w: rw, h: rh });
}

// --- Main processing ---
async function processImage(imagePath) {
  const baseName = path.basename(imagePath, path.extname(imagePath));
  console.log(`\n处理: ${path.basename(imagePath)}`);

  const metadata = await sharp(imagePath).metadata();
  const origW = metadata.width;
  const origH = metadata.height;
  console.log(`  原始尺寸: ${origW}x${origH}`);

  const scale = ANALYSIS_WIDTH / origW;
  const analysisH = Math.round(origH * scale);

  // Get grayscale data at analysis resolution
  const grayBuffer = await sharp(imagePath)
    .resize(ANALYSIS_WIDTH, analysisH, { fit: 'fill' })
    .grayscale()
    .raw()
    .toBuffer();
  const gray = new Uint8Array(grayBuffer);

  // Compute gradient magnitude
  const grad = computeGradient(gray, ANALYSIS_WIDTH, analysisH);
  console.log(`  边缘检测完成`);

  // Recursive splitting to find photo regions
  const regions = [];
  splitRegion(grad, ANALYSIS_WIDTH, analysisH, 0, 0, ANALYSIS_WIDTH, analysisH, 0, regions);

  // Filter small regions
  const totalArea = ANALYSIS_WIDTH * analysisH;
  const validRegions = regions.filter(r => (r.w * r.h) / totalArea >= MIN_PHOTO_RATIO);

  // Sort by position (top-to-bottom, left-to-right)
  validRegions.sort((a, b) => {
    const rowDiff = Math.abs(a.y - b.y) > analysisH * 0.1 ? a.y - b.y : 0;
    return rowDiff !== 0 ? rowDiff : a.x - b.x;
  });

  console.log(`  检测到 ${validRegions.length} 张照片`);

  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  let cropCount = 0;
  for (let i = 0; i < validRegions.length; i++) {
    const r = validRegions[i];

    // Scale back to original dimensions
    let cropX = Math.max(0, Math.round(r.x / scale) - REGION_PADDING);
    let cropY = Math.max(0, Math.round(r.y / scale) - REGION_PADDING);
    let cropW = Math.min(origW - cropX, Math.round(r.w / scale) + REGION_PADDING * 2);
    let cropH = Math.min(origH - cropY, Math.round(r.h / scale) + REGION_PADDING * 2);

    if (cropW < 100 || cropH < 100) continue;

    cropCount++;
    const outputFile = path.join(OUTPUT_DIR, `${baseName}_crop${cropCount}.jpg`);
    await sharp(imagePath)
      .extract({ left: cropX, top: cropY, width: cropW, height: cropH })
      .jpeg({ quality: 95 })
      .toFile(outputFile);

    const aspectRatio = (cropW / cropH).toFixed(2);
    console.log(`  [${cropCount}] ${cropW}x${cropH} (比例 ${aspectRatio}) → ${path.basename(outputFile)}`);
  }

  console.log(`  共切割出 ${cropCount} 张照片`);
  return cropCount;
}

async function main() {
  console.log('========================================');
  console.log('  扫描照片自动切割工具 v3');
  console.log(`  输入目录: ${INPUT_DIR}`);
  console.log(`  输出目录: ${OUTPUT_DIR}`);
  console.log('========================================');

  // Clean output dir
  if (fs.existsSync(OUTPUT_DIR)) {
    fs.readdirSync(OUTPUT_DIR).forEach(f => fs.unlinkSync(path.join(OUTPUT_DIR, f)));
  }

  const files = fs.readdirSync(INPUT_DIR).filter(f => /\.jpe?g$/i.test(f)).sort();
  if (files.length === 0) {
    console.log('未找到 JPG 图片文件。');
    return;
  }
  console.log(`找到 ${files.length} 张扫描图片`);

  let totalCrops = 0;
  for (const file of files) {
    try {
      totalCrops += await processImage(path.join(INPUT_DIR, file));
    } catch (err) {
      console.error(`处理 ${file} 时出错: ${err.message}`);
    }
  }

  console.log('\n========================================');
  console.log(`处理完成！共切割出 ${totalCrops} 张照片`);
  console.log(`输出目录: ${OUTPUT_DIR}`);
  console.log('========================================');
}

main().catch(console.error);
