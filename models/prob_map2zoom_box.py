import cv2
import numpy as np


def prob_map_to_multi_crops_1(prob_map,
                            rel_thresh=0.1,
                            min_area_ratio=0.0001,
                            pad_ratio=0.2,
                            min_size=336,
                            max_crops=3):
    """
    针对多峰概率分布，提取多个候选 Zoom-in 区域
    参数:
    prob_map: (H, W) float32, [0, 1]
    rel_thresh: 相对阈值，低于 (max_val * rel_thresh) 的区域被忽略
    min_area_ratio: 最小连通域面积占全图比例，小于此比例的噪点被忽略
    pad_ratio: 在检测到的高亮区周围扩充的比例 (Context)
    min_size: 最小 Crop 尺寸 (强制正方形)
    max_crops: 最多返回几个框 (按概率总和排序)
    返回:
    crop_list: List of (xmin, ymin, xmax, ymax)
    """

    H, W = prob_map.shape
    max_val = np.max(prob_map)
    # 1. 极低置信度处理：如果全图概率都很低，返回全图或空
    if max_val < 1e-4:
        return [(0, 0, W, H)]
    # 2. 预处理：高斯模糊 (平滑噪点，连接相近的断点)
    # sigma 设为图像宽度的 1% 左右
    blur_sigma = int(max(W, H) * 0.01) | 1  # 必须是奇数
    smooth_map = cv2.GaussianBlur(prob_map, (blur_sigma, blur_sigma), 0)
    # 3. 二值化 (Thresholding)
    # 提取“核心岛屿”：只保留概率比较高的区域
    thresh_val = max_val * rel_thresh
    binary_map = (smooth_map > thresh_val).astype(np.uint8) * 255
    # 4. 查找连通域 (Contours)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    total_pixels = H * W
    for cnt in contours:
        # 过滤微小噪点
        area = cv2.contourArea(cnt)
        if area < total_pixels * min_area_ratio:
            continue
        # 计算该连通域内的总概率质量 (作为排序依据)
        # 创建 mask 来提取原始 prob_map 在该区域的值
        mask = np.zeros_like(binary_map)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        score = np.sum(prob_map * mask)
        # 获取基础包围框 (x, y, w, h)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w / 2, y + h / 2
        # --- 扩充与正方形化逻辑 (与单峰逻辑类似) ---
        # 1. 确定边长：取最大边长并加上 padding
        side = max(w, h) * (1 + pad_ratio * 2)
        side = max(side, min_size)
        # 2. 计算新坐标
        new_w = new_h = side
        x1 = int(cx - new_w / 2)
        y1 = int(cy - new_h / 2)
        x2 = int(cx + new_w / 2)
        y2 = int(cy + new_h / 2)
        # 3. 边界处理 (平移而非截断)
        if x1 < 0: x2 -= x1; x1 = 0
        if y1 < 0: y2 -= y1; y1 = 0
        if x2 > W: x1 -= (x2 - W); x2 = W
        if y2 > H: y1 -= (y2 - H); y2 = H
        # 硬截断
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        candidates.append({
            'box': (x1, y1, x2, y2),
            'score': score
        })
    # 5. 排序与筛选
    # 按区域内的概率总和从大到小排序
    candidates.sort(key=lambda x: x['score'], reverse=True)
    # 取 Top-K
    final_crops = [c['box'] for c in candidates[:max_crops]]
    # 如果没有找到任何有效区域 (比如所有点都低于阈值)，返回全图或单峰逻辑的结果
    if not final_crops:
        return [(0, 0, W, H)]
    return final_crops



def prob_map_to_multi_crops(prob_map,
                            rel_thresh=0.1,
                            min_area_ratio=0.0001,
                            pad_ratio=0.2,
                            min_size=448,
                            max_crops=3,
                            squareness=0.5):
    """
    针对多峰概率分布，提取多个候选 Zoom-in 区域，支持形状控制系数。

    参数:
    prob_map: (H, W) float32, [0, 1]
    rel_thresh: 相对阈值，低于 (max_val * rel_thresh) 的区域被忽略
    min_area_ratio: 最小连通域面积占全图比例
    pad_ratio: 基础扩充比例
    min_size: 最小 Crop 尺寸
    max_crops: 最多返回几个框
    squareness: [0.0, 1.0] 控制形状系数 (0.0=纯矩形, 1.0=纯正方形)
    """
    H, W = prob_map.shape
    max_val = np.max(prob_map)

    if max_val < 1e-4:
        return [(0, 0, W, H)]

    # 1. 预处理：高斯模糊
    blur_sigma = int(max(W, H) * 0.01) | 1
    smooth_map = cv2.GaussianBlur(prob_map, (blur_sigma, blur_sigma), 0)

    # 2. 二值化提取连通域
    thresh_val = max_val * rel_thresh
    binary_map = (smooth_map > thresh_val).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    total_pixels = H * W

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_pixels * min_area_ratio:
            continue

        # 计算得分（概率质量）
        mask = np.zeros_like(binary_map)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        score = np.sum(prob_map * mask)

        # 获取基础包围框 (x, y, w, h)
        bx, by, bw, bh = cv2.boundingRect(cnt)
        cx, cy = bx + bw / 2, by + bh / 2

        # --- 引入 Squareness 的尺寸计算逻辑 ---

        # A. 计算带 Padding 的基础矩形尺寸
        raw_w = bw * (1 + pad_ratio * 2)
        raw_h = bh * (1 + pad_ratio * 2)

        # B. 计算目标混合尺寸 (线性插值)
        max_side = max(raw_w, raw_h)
        target_w = raw_w + (max_side - raw_w) * squareness
        target_h = raw_h + (max_side - raw_h) * squareness

        # C. 应用最小尺寸约束
        final_w = max(target_w, min_size)
        final_h = max(target_h, min_size)

        # D. 计算坐标
        x1 = int(cx - final_w / 2)
        y1 = int(cy - final_h / 2)
        x2 = int(cx + final_w / 2)
        y2 = int(cy + final_h / 2)

        # 3. 智能平移处理 (保持尺寸，移动中心)
        if x1 < 0: x2 += abs(x1); x1 = 0
        if y1 < 0: y2 += abs(y1); y1 = 0
        if x2 > W: x1 -= (x2 - W); x2 = W
        if y2 > H: y1 -= (y2 - H); y2 = H

        # 硬截断（针对图像尺寸本身小于 min_size 的极端情况）
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        candidates.append({
            'box': (x1, y1, x2, y2),
            'score': score
        })

    # 4. 排序与筛选
    candidates.sort(key=lambda x: x['score'], reverse=True)
    final_crops = [c['box'] for c in candidates[:max_crops]]

    return final_crops if final_crops else [(0, 0, W, H)]

def prob_map_to_zoom_box(prob_map, k=3.0, min_size=224, padding=0.05):
    """
    将概率地图转换为 Zoom-in 矩形区域

    参数:
    prob_map: np.ndarray (H, W), 值域 [0, 1]
    k: 标准差倍数，k越大，zoom区域涵盖的不确定性范围越广 (推荐 2.0-3.0)
    min_size: 最小缩放尺寸（像素），防止缩得太小导致丢失上下文
    padding: 在计算出的框周围额外增加的比例边距 (0.1 = 10% padding)

    返回:
    zoom_box: (xmin, ymin, xmax, ymax) 整数坐标
    """
    H, W = prob_map.shape

    # 1. 计算总能量（用于归一化）
    total_energy = np.sum(prob_map)
    if total_energy == 0:
        return (0, 0, W, H)  # 如果全图没概率，返回原图

    # 2. 计算一阶矩 (质心 / 均值)
    yy, xx = np.mgrid[0:H, 0:W]
    mu_x = np.sum(xx * prob_map) / total_energy
    mu_y = np.sum(yy * prob_map) / total_energy

    # 3. 计算二阶矩 (方差 / 标准差)
    var_x = np.sum(((xx - mu_x) ** 2) * prob_map) / total_energy
    var_y = np.sum(((yy - mu_y) ** 2) * prob_map) / total_energy
    sigma_x = np.sqrt(max(var_x, 1e-6))
    sigma_y = np.sqrt(max(var_y, 1e-6))

    # 4. 确定初始宽度和高度
    # 基于 k * sigma 确定，并考虑 padding
    width = sigma_x * k * 2 * (1 + padding) * 0.7 # 修正系数0.7
    height = sigma_y * k * 2 * (1 + padding) * 1.1 # 修正系数1.1

    # 5. 应用最小尺寸约束 (VLM通常需要一定的上下文才能理解图标)
    width = max(width, min_size)
    height = max(height, min_size)

    # 6. 计算边界并处理溢出
    xmin = int(mu_x - width / 2)
    ymin = int(mu_y - height / 2)
    xmax = int(mu_x + width / 2)
    ymax = int(mu_y + height / 2)

    # 7. 边界对齐与平移（如果一边出界，整体平移而不是直接裁剪，以保持尺寸不变）
    if xmin < 0:
        xmax -= xmin;
        xmin = 0
    if ymin < 0:
        ymax -= ymin;
        ymin = 0
    if xmax > W:
        xmin -= (xmax - W);
        xmax = W
    if ymax > H:
        ymin -= (ymax - H);
        ymax = H

    # 再次检查边界（防止平移后依然出界，如原图尺寸就小于min_size的情况）
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(W, xmax), min(H, ymax)

    return (int(xmin), int(ymin), int(xmax), int(ymax))



def prob_map_to_zoom_box_squ(prob_map, k=3.0, min_size=336, padding=0.1, squareness=0.5):
    """
    将概率地图转换为 Zoom-in 区域，支持形状的平滑过渡。

    参数:
    prob_map: np.ndarray (H, W)
    k: 标准差倍数 (覆盖范围)
    min_size: 最小尺寸 (像素)
    padding: 基础边距比例
    squareness: [0.0, 1.0] 控制形状系数
                0.0 = 保持原始长宽比 (矩形)
                1.0 = 强制正方形 (以最长边为准)
                0.5 = 半正方形 (折中方案)
    """
    H, W = prob_map.shape
    total_energy = np.sum(prob_map)

    if total_energy < 1e-6:
        return (0, 0, W, H)

    # 1. 计算一阶矩 (质心)
    xs = np.arange(W)
    ys = np.arange(H)
    prob_x = np.sum(prob_map, axis=0)
    prob_y = np.sum(prob_map, axis=1)

    mu_x = np.sum(xs * prob_x) / total_energy
    mu_y = np.sum(ys * prob_y) / total_energy

    # 2. 计算二阶矩 (方差)
    var_x = np.sum(((xs - mu_x) ** 2) * prob_x) / total_energy
    var_y = np.sum(((ys - mu_y) ** 2) * prob_y) / total_energy

    sigma_x = np.sqrt(max(var_x, 1e-6))
    sigma_y = np.sqrt(max(var_y, 1e-6))

    # 3. 计算原始矩形尺寸 (Raw Rectangle)
    raw_w = sigma_x * k * 2 * (1 + padding)
    raw_h = sigma_y * k * 2 * (1 + padding)

    # 4. 形状混合逻辑 (Shape Blending)
    # 我们以最长边作为正方形的目标边长，这样只会“补全”短边，而不会“截断”长边
    # 这对 VLM 来说更安全，不会丢失信息
    max_side = max(raw_w, raw_h)

    # 线性插值: Target = Raw + (Max - Raw) * coef
    target_w = raw_w + (max_side - raw_w) * squareness
    target_h = raw_h + (max_side - raw_h) * squareness

    # 5. 应用最小尺寸约束
    final_w = max(target_w, min_size)
    final_h = max(target_h, min_size)

    # 6. 计算坐标并平移 (Sliding Window)
    xmin = int(mu_x - final_w / 2)
    ymin = int(mu_y - final_h / 2)
    xmax = int(mu_x + final_w / 2)
    ymax = int(mu_y + final_h / 2)

    # 智能平移：如果框超出了边界，优先平移而不是直接裁剪
    if xmin < 0:
        xmax += abs(xmin);
        xmin = 0
    if ymin < 0:
        ymax += abs(ymin);
        ymin = 0
    if xmax > W:
        xmin -= (xmax - W);
        xmax = W
    if ymax > H:
        ymin -= (ymax - H);
        ymax = H

    # 7. 硬截断 (最后防线)
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(W, xmax), min(H, ymax)

    if xmax <= xmin or ymax <= ymin:
        return (0, 0, W, H)

    return (int(xmin), int(ymin), int(xmax), int(ymax))