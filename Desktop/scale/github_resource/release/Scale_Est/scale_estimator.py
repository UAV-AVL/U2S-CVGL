import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import json
from typing import Dict, Optional
import cv2  # 新增导入OpenCV用于可视化
import csv

class ScaleEstimator:

    def __init__(self, config=None, data_provider=None):
        # 在配置中添加region_altitude参数
        self.config = {
            'region_altitude': 0.0,  # 新增：区域海拔高度，默认为0
            'min_vehicles_threshold': 3,
            'confidence_threshold': 0.7,  # 置信度阈值
            'average_car_length': 4.5,  # 小型车辆平均宽度（米）
            'average_car_width': 1.8,  # 小型车辆平均宽度（米）
            'average_car_height': 0.8,
            'output_pdf': "height_estimation_results.pdf",
            'log_file': "log.txt",
            'exif_cache_file': "exif_cache.json",  # 新增EXIF缓存文件配置
            'max_images': None,  # 新增：最大处理图像数量
            'visualize_detections': False,  # 新增：是否可视化检测结果
            'visualization_dir': "visualizations",  # 新增：可视化结果保存目录
            'image_params_csv': "image_params.csv",
        }
        if config:
            self.config.update(config)
        self.data_provider = data_provider
        # 结果存储
        self.results = []
        self.overall_stats = {}
        # 初始化EXIF缓存
        self.exif_cache = self._load_exif_cache()
        self._valid_images_cache = {}
        # 创建可视化目录
        if self.config['visualize_detections']:
            os.makedirs(self.config['visualization_dir'], exist_ok=True)

    def _load_exif_cache(self) -> Dict[str, dict]:
        """加载EXIF缓存"""
        if os.path.exists(self.config['exif_cache_file']):
            try:
                with open(self.config['exif_cache_file'], 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load EXIF cache: {e}")
        return {}

    def _save_exif_cache(self):
        """保存EXIF缓存"""
        try:
            # 只有在确实有缓存文件路径时才保存
            if self.config.get('exif_cache_file'):
                with open(self.config['exif_cache_file'], 'w') as f:
                    json.dump(self.exif_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save EXIF cache: {e}")

    def count_valid_test_images(self, image_dir: str, pitch_thresh: float = -60.0) -> int:
        """
        Images 的新定义（并遵循 test_ratio 采样）：
        1) image_dir 下能找到的原始图像文件
        2) 能拿到 uav_data（_get_image_data 返回非 None）
        3) pitch <= pitch_thresh
        4) 再对通过(1)(2)(3)的 img_id 做与 estimate_height 相同的 test_ratio 等间隔采样
        """
        ratio = self.config.get('test_ratio', 1.0)
        cache_key = (image_dir, pitch_thresh, ratio)
        if cache_key in self._valid_images_cache:
            return self._valid_images_cache[cache_key]
        SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        # === DenseUAV fast path: 只用文件名正则判断是否能解析高度 ===
        is_dense = self.data_provider.__class__.__name__.lower().find("dense") != -1
        if is_dense:
            pat = re.compile(r"_H(\d+)")
            valid_img_ids = []
            for fname in os.listdir(image_dir):
                ext = os.path.splitext(fname)[1]
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                if pat.search(fname) is None:
                    continue
                img_id = os.path.splitext(fname)[0]
                valid_img_ids.append(img_id)

            valid_img_ids = sorted(set(valid_img_ids))
            if 0 < ratio < 1.0:
                step = max(1, int(1 / ratio))
                cnt = len(valid_img_ids[::step])
            else:
                cnt = len(valid_img_ids)

            self._valid_images_cache[cache_key] = cnt
            return cnt
        # 1) 收集“原始可用图像”的 img_id（这里 img_id = 文件名去掉扩展名）
        valid_img_ids = []
        for fname in os.listdir(image_dir):
            ext = os.path.splitext(fname)[1]
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            img_id = os.path.splitext(fname)[0]
            img_path = os.path.join(image_dir, fname)

            # uav_data 必须存在
            uav_data = self._get_image_data(img_path)
            if not uav_data:
                continue

            # pitch 必须合格（与你 estimate_height 一致：pitch > -60 跳过）
            if uav_data.get('pitch', 0) > pitch_thresh:
                continue

            valid_img_ids.append(img_id)

        # 2) 按 test_ratio 做等间隔采样（完全对齐 estimate_height 的写法）
        valid_img_ids = sorted(set(valid_img_ids))
        total_count = len(valid_img_ids)

        ratio = self.config.get('test_ratio', 1.0)
        if ratio < 1.0 and ratio > 0:
            step = int(1 / ratio)
            step = max(1, step)
            sampled_ids = valid_img_ids[::step]
            return len(sampled_ids)
        else:
            return total_count
    class Detection:
        def __init__(self, img_id, confidence, points):
            self.img_id = img_id
            self.confidence = confidence
            self.points = points
            self.width_pixels = None
            self.edge_points = None
            self.estimated_height = None
            self.is_inlier = True
            self.rel_error = None  # 新增相对误差字段

        def calculate_error(self, actual_altitude):
            """计算相对误差"""
            if self.estimated_height is not None:
                self.rel_error = (self.estimated_height - actual_altitude) / actual_altitude * 100

        def extract_geometric_features(self):
            """同时计算长边(Length)和短边(Width)及其向量"""
            # points是 [x1, y1, x2, y2, ...] -> [(x,y), ...]
            pts = [(self.points[i], self.points[i + 1]) for i in range(0, 8, 2)]

            # 计算相邻两边的长度和向量 (假设矩形，只算前两边即可区分长宽)
            # 边0: pts[0]->pts[1]
            vec0 = np.array(pts[1]) - np.array(pts[0])
            len0 = np.linalg.norm(vec0)

            # 边1: pts[1]->pts[2]
            vec1 = np.array(pts[2]) - np.array(pts[1])
            len1 = np.linalg.norm(vec1)

            if len0 > len1:
                # 边0是长边，边1是短边
                self.length_pixels = len0
                self.long_edge_vector = vec0

                self.width_pixels = len1
                self.short_edge_vector = vec1
            else:
                # 边1是长边，边0是短边
                self.length_pixels = len1
                self.long_edge_vector = vec1

                self.width_pixels = len0
                self.short_edge_vector = vec0

            return self.length_pixels, self.width_pixels

    def parse_detection_file(self, file_path):
        """解析检测结果文件"""
        detections = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue

                img_id = parts[0]
                confidence = float(parts[1])
                points = list(map(float, parts[2:]))

                if confidence >= self.config['confidence_threshold']:
                    det = self.Detection(img_id, confidence, points)
                    det.extract_geometric_features()
                    detections.append(det)

        return detections

    def find_image_file(self, image_dir, img_id):
        # 支持的图像扩展名列表
        SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        """查找具有不同扩展名的图像文件"""
        for ext in SUPPORTED_EXTENSIONS:
            img_path = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                return img_path
        return None

    def _get_image_data(self, img_path: str) -> Optional[dict]:
        img_key = os.path.basename(img_path)

        # 1. 查缓存
        if img_key in self.exif_cache:
            return self.exif_cache[img_key]

        # 2. 调用适配器获取数据 (解耦核心！)
        uav_data = self.data_provider.get_data(img_path)

        if uav_data:
            # 缓存并返回
            self.exif_cache[img_key] = uav_data
            return uav_data
        return None

    def _plot_results(self, pdf_pages, img_id, img_dets, X, y, ransac, avg_height, std_height, uav_data):
        if pdf_pages is None: return

        inlier_mask = np.array([det.is_inlier for det in img_dets])
        actual_alt = uav_data['rel_alt']

        # 防止除零
        if actual_alt == 0: actual_alt = 1e-5

        # 只计算内点的误差分布
        rel_errors = [(h - actual_alt) / actual_alt * 100 for h in y[inlier_mask]]

        if rel_errors:
            avg_rel_error = np.mean(rel_errors)
        else:
            avg_rel_error = 0

        plt.figure(figsize=(15, 6))

        # === 左图：高度估计散点图 ===
        plt.subplot(1, 2, 1)

        # 1. 绘制散点
        # [修正] 使用 flatten() 将二维数组展平为一维，避免维度警告
        plt.scatter(X[inlier_mask].flatten(), y[inlier_mask], color='green', label='Inliers', alpha=0.7)
        plt.scatter(X[~inlier_mask].flatten(), y[~inlier_mask], color='red', label='Outliers', alpha=0.5)

        # 2. 绘制水平估计线
        plt.axhline(y=avg_height, color='blue', linestyle='-', linewidth=2, label=f'Est Height: {avg_height:.1f}m')

        # 3. 绘制真值线
        plt.axhline(y=actual_alt, color='black', linestyle='--', linewidth=2, label=f'True Height: {actual_alt:.1f}m')

        # 4. 绘制置信区间 (±1 std)
        # [修正] 使用 np.min 和 np.max 获取标量值，修复 ValueError
        if len(X) > 0:
            x_min, x_max = np.min(X), np.max(X)
            plt.fill_between([x_min, x_max], avg_height - std_height, avg_height + std_height, color='blue', alpha=0.1,
                             label='±1 Std Dev')

        plt.xlabel('Vehicle Width (pixels)')
        plt.ylabel('Estimated Height (m)')
        plt.title(f'Image {img_id}: Height Estimation (IQR)\n'
                  f'Est: {avg_height:.1f}m, Diff: {avg_height - actual_alt:.1f}m')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle=':', alpha=0.6)

        # === 右图：误差分布 ===
        plt.subplot(1, 2, 2)
        plt.hist(rel_errors, bins=15, color='purple', alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title(f'Inlier Error Distribution\nMean Error: {avg_rel_error:.1f}%')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()

    def estimate_height(self, detections, image_dir):
        img_groups = {}
        for det in detections:
            if det.img_id not in img_groups:
                img_groups[det.img_id] = []
            img_groups[det.img_id].append(det)

        # =================【修改：按比例等间隔采样】=================
        # 1. 必须先排序！否则“等间隔”没有意义（假设ID包含时间或序列信息）
        # 比如从 ['01', '02', '03'...] 中采样
        all_img_ids = sorted(list(img_groups.keys()))
        total_count = len(all_img_ids)

        # 获取配置中的比例 (默认为 1.0 即全量)
        ratio = self.config.get('test_ratio', 1.0)

        # 如果比例小于 1.0，则进行采样
        if ratio < 1.0 and ratio > 0:
            # 计算步长。例如 ratio=0.2 (20%) -> 步长=5，即每5张取1张
            step = int(1 / ratio)
            # 步长至少为1
            step = max(1, step)

            # 使用列表切片进行等间隔提取 [start:stop:step]
            selected_subset = all_img_ids[::step]
            selected_ids = set(selected_subset)

            print(f"  [Sampling] Mode: Fixed-Interval (Sorted)")
            print(f"  [Sampling] Total: {total_count}, Ratio: {ratio}, Step: {step}, Selected: {len(selected_ids)}")
        else:
            selected_ids = set(all_img_ids)
            print(f"  [Sampling] Use All Images: {total_count}")
        # ========================================================

        # [修改] 只有在路径存在时才初始化 PDF 和 Log
        pdf_pages = None
        if self.config.get('output_pdf'):
            pdf_pages = PdfPages(self.config['output_pdf'])

        log = None
        if self.config.get('log_file'):
            log = open(self.config['log_file'], 'w')

        processed_count = 0

        target_items = [(k, v) for k, v in img_groups.items() if k in selected_ids]
        # 如果需要严格按ID排序处理，取消下面这行的注释：
        target_items.sort(key=lambda x: x[0])

        for i, (img_id, img_dets) in enumerate(tqdm(target_items)):

            if self.config['max_images'] is not None and processed_count >= self.config['max_images']:
                break

            # 检查车辆数量阈值
            if len(img_dets) < self.config['min_vehicles_threshold']:
                if log:
                    log.write(
                        f"跳过图像 {img_id}：车辆数量不足 ({len(img_dets)} < {self.config['min_vehicles_threshold']})\n")
                continue

            # 查找图像
            img_path = self.find_image_file(image_dir, img_id)
            if img_path is None:
                # print(f"Warning: Image {img_id} not found with any supported extension")
                continue

            # 获取相机参数
            uav_data = self._get_image_data(img_path)
            if not uav_data:
                continue
            if uav_data['pitch'] > -60: # 之前注释掉的逻辑
                continue
            # [新增] 检查是否有畸变参数，如果有则进行校正
            has_distortion_info = ('dist_coeffs' in uav_data and 'camera_matrix' in uav_data)
            # 可视化检测结果
            if self.config['visualize_detections']:
                self._visualize_detections(img_path, img_dets, img_id)

            # 为每个检测计算高度估计
            for det in img_dets:
                # [关键修改] 如果有畸变参数，对点进行矫正
                if has_distortion_info:
                    # 备份原始点用于可视化（如果需要）
                    original_points = det.points.copy()

                    # 执行去畸变
                    det.points = self._undistort_points(
                        original_points,
                        uav_data['camera_matrix'],
                        uav_data['dist_coeffs']
                    )

                    # [重要] 重新计算宽度！
                    # 因为点的位置变了，边长也会变 (枕形畸变去畸变后，边缘点会向外扩，宽度变大，高度变低)
                    det.extract_geometric_features()

                det.estimated_height = self._calculate_height(
                    det,
                    img_width=uav_data['width'],
                    img_height=uav_data['height'],
                    focal_length=uav_data['focal_len'],
                    camera_size=uav_data['cam_size'],
                    pitch_angle=uav_data['pitch']  # [重要] 传入 Pitch
                )

            # 保存一次缓存
            self._save_exif_cache()

            # =================【修改开始：使用 IQR 替代 RANSAC 回归】=================
            if len(img_dets) >= 3:
                # 1. 提取所有检测框的像素宽度(X)和估算高度(y)
                X = np.array([det.length_pixels for det in img_dets]).reshape(-1, 1)
                y = np.array([det.estimated_height for det in img_dets])

                # 2. 使用 IQR (四分位距) 进行一维离群值剔除
                # 目的：找到高度分布最密集的区间 (Mode)，剔除过大或过小的高/宽车辆
                Q1 = np.percentile(y, 25)
                Q3 = np.percentile(y, 75)
                IQR = Q3 - Q1

                # 定义保留范围 (通常 k=1.5，若数据较杂可适当收缩至 1.0)
                k = 1.5
                lower_bound = Q1 - k * IQR
                upper_bound = Q3 + k * IQR

                # 生成内点掩码
                inlier_mask = (y >= lower_bound) & (y <= upper_bound)

                # 极端情况保护：如果所有点都被剔除（虽然不太可能），则回退到保留所有点
                if not np.any(inlier_mask):
                    inlier_mask = np.ones(len(y), dtype=bool)

                # 3. 标记内点/外点
                for k_idx, det in enumerate(img_dets):
                    det.is_inlier = inlier_mask[k_idx]

                # 4. 计算统计信息 (仅使用内点，符合高斯分布假设)
                inlier_heights = y[inlier_mask]
                avg_height = np.mean(inlier_heights)
                std_height = np.std(inlier_heights)

                # 记录结果 (注意：_plot_results 不需要传 ransac 对象了，传 None)
                self._log_results(log, img_id, img_dets, avg_height, std_height, uav_data)
                self._plot_results(pdf_pages, img_id, img_dets, X, y, None, avg_height, std_height, uav_data)

                # 保存结果
                self.results.append({
                    'img_id': img_id,
                    'avg_height': avg_height,
                    'std_height': std_height,
                    'actual_altitude': uav_data['rel_alt'],
                    'num_detections': len(img_dets),
                    'num_inliers': int(np.sum(inlier_mask))
                })

                processed_count += 1



        # 计算总体统计信息
        if self.results:
            self._calculate_overall_stats(log)
            self._plot_overall_distribution(pdf_pages)

            # [新增] 保存详细结果到 CSV
            self._save_summary_csv()

        # [修改] 安全关闭
        if pdf_pages:
            pdf_pages.close()
        if log:
            log.close()

    def _save_summary_csv(self):
        """将所有图像的估计结果保存为 CSV 文件"""
        if not self.config['save_root']:
            return

        # 确保目录存在
        os.makedirs(self.config['save_root'], exist_ok=True)
        # 如果是按区域跑的，可能需要区分文件名，或者都叫 summary.csv
        # 这里建议直接保存在 exp_subname 目录下 (由 runner 传入的 config['log_file'] 推断路径)

        # 尝试从 log_file 路径推断保存目录
        if self.config.get('log_file'):
            save_dir = os.path.dirname(self.config['log_file'])
            csv_path = os.path.join(save_dir, "summary.csv")
        else:
            csv_path = os.path.join(self.config['save_root'], "summary.csv")

        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 写入表头
                writer.writerow(
                    ['Image_ID', 'Estimated_Height(m)', 'Actual_Altitude(m)', 'Error(m)', 'Num_Vehicles_Used'])

                # 写入数据
                for res in self.results:
                    writer.writerow([
                        res['img_id'],
                        f"{res['avg_height']:.2f}",
                        f"{res['actual_altitude']:.2f}",
                        f"{res['avg_height'] - res['actual_altitude']:.2f}",
                        res['num_inliers']  # 参与计算的内点数量
                    ])
            print(f"  [Output] Summary saved to: {csv_path}")
        except Exception as e:
            print(f"  [Error] Failed to save CSV: {e}")
    def _visualize_detections(self, img_path, detections, img_id):
        """可视化检测结果并保存为JPG"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return

        for det in detections:
            points = np.array([(det.points[i], det.points[i + 1]) for i in range(0, 8, 2)], dtype=np.int32)
            cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            # p1, p2 = det.edge_points
            # cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 0, 255), 2)
            cv2.putText(img, f"{det.confidence:.2f}", (int(points[0][0]), int(points[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        output_path = os.path.join(self.config['visualization_dir'], f"{img_id}_detections.jpg")
        cv2.imwrite(output_path, img)

    def _undistort_points(self, points, camera_matrix, dist_coeffs):
        """
        对检测框角点进行去畸变
        """
        if camera_matrix is None or dist_coeffs is None:
            return points

        # [修改点]：确保输入转为 numpy array (防止传入的是 list)
        cam_mat_np = np.array(camera_matrix, dtype=np.float32)
        dist_coeffs_np = np.array(dist_coeffs, dtype=np.float32)

        # 1. 重塑为 OpenCV 需要的格式 (N, 1, 2)
        pts_np = np.array(points).reshape(-1, 1, 2).astype(np.float32)

        # 2. 调用 OpenCV 去畸变
        # 这里的 P=cam_mat_np 很重要，保证返回的是像素坐标
        undistorted_pts = cv2.undistortPoints(pts_np, cam_mat_np, dist_coeffs_np, P=cam_mat_np)

        # 3. 展平回 list
        return undistorted_pts.reshape(-1).tolist()

    def _calculate_height(self, det, img_width, img_height, focal_length, camera_size, pitch_angle):
        """
        修正后的高度估计函数：考虑X轴偏移和3D立体投影，含Tuple检查和校准系数
        """
        # 1. 基础参数
        W_car = self.config['average_car_width']
        H_car = self.config['average_car_height']
        L_car = self.config['average_car_length']  # 读取车长
        # 获取统计权重 (如果config没设，默认宽的权重更高，因为车宽方差小)
        k_w = self.config.get('weight_prior_width', 1.0)
        k_l = self.config.get('weight_prior_length', 1.0)  # 车长变化大，权重稍低

        # 2. 相机参数计算 (保持不变)
        if isinstance(camera_size, (list, tuple)):
            sensor_diag = (camera_size[0] ** 2 + camera_size[1] ** 2) ** 0.5
        else:
            sensor_diag = float(camera_size)
        img_diag = (img_width ** 2 + img_height ** 2) ** 0.5
        pixel_size = sensor_diag / img_diag
        f_pix = focal_length / pixel_size
        cx, cy = img_width / 2, img_height / 2

        # 3. 几何中心与视线角 alpha
        pts = np.array(det.points).reshape(-1, 2)
        u_center = np.mean(pts[:, 0])
        v_center = np.mean(pts[:, 1])

        ray_vec = np.array([u_center - cx, v_center - cy, f_pix])
        ray_len = np.linalg.norm(ray_vec)

        pitch_rad = np.radians(pitch_angle)
        up_vec = np.array([0, -np.cos(pitch_rad), -np.sin(pitch_rad)])

        # 计算视线与垂直方向夹角的余弦值 -> 得到 alpha 的正弦值
        # 注意：alpha 是视线与水平面的夹角(俯角)，这里 proj_vertical 是视线与垂直线的投影
        # sin(alpha) = cos(90-alpha) = dot(ray, vertical) / |ray|
        proj_vertical = np.abs(np.dot(ray_vec, up_vec))
        sin_alpha = proj_vertical / ray_len

        if sin_alpha > 1.0: sin_alpha = 1.0
        if sin_alpha < 0.01: return None
        cos_alpha = np.sqrt(1 - sin_alpha ** 2)

        # 4. 计算径向向量
        radial_vec = np.array([u_center - cx, v_center - cy])
        radial_dist = np.linalg.norm(radial_vec)
        if radial_dist < 1e-3: return None  # 避免除零

        # ==================== A. 基于宽度的估计 (Width-based) ====================
        H_width_est = 0
        w_geom_weight = 0

        if det.width_pixels > 0 and det.short_edge_vector is not None:
            # 计算宽边与径向的夹角 gamma_w
            w_vec = det.short_edge_vector
            w_len = np.linalg.norm(w_vec)
            if w_len > 0:
                cos_gamma_w = abs(np.dot(radial_vec, w_vec) / (radial_dist * w_len))
                sin_gamma_w = np.sqrt(1 - cos_gamma_w ** 2)

                # 投影修正公式
                term_rad = (W_car * sin_alpha + H_car * cos_alpha) * cos_gamma_w
                term_tan = W_car * sin_gamma_w
                W_eff = np.sqrt(term_rad ** 2 + term_tan ** 2)

                # 计算高度
                H_width_est = (ray_len * W_eff * sin_alpha) / det.width_pixels


        # ==================== B. 基于长度的估计 (Length-based) ====================
        H_length_est = 0

        if det.length_pixels > 0 and det.long_edge_vector is not None:
            # 计算长边与径向的夹角 gamma_l
            l_vec = det.long_edge_vector
            l_len = np.linalg.norm(l_vec)
            if l_len > 0:
                cos_gamma_l = abs(np.dot(radial_vec, l_vec) / (radial_dist * l_len))
                sin_gamma_l = np.sqrt(1 - cos_gamma_l ** 2)

                # 投影修正公式
                term_rad = (L_car * sin_alpha + H_car * cos_alpha) * cos_gamma_l
                term_tan = L_car * sin_gamma_l
                L_eff = np.sqrt(term_rad ** 2 + term_tan ** 2)

                # 计算高度
                H_length_est = (ray_len * L_eff * sin_alpha) / det.length_pixels
        # ==================== C. 加权融合 (Weighted Fusion) ====================

        # 综合权重 = 统计先验权重 * 几何观测权重
        final_w_width = k_w
        final_w_length = k_l

        total_weight = final_w_width + final_w_length

        if total_weight == 0:
            return None

        H_final = (H_width_est * final_w_width + H_length_est * final_w_length) / total_weight

        # 校准系数 (可根据实验微调)
        calibration_factor = 1.05
        return H_final * calibration_factor

    def _log_results(self, log_file, img_id, detections, avg_height, std_height, uav_data):
        if log_file is None: return  # [修改] 检查 None

        log_file.write(f"Image {img_id}:\n")
        log_file.write(f"  Actual altitude: {uav_data['rel_alt']} m\n")
        log_file.write(f"  Estimated height: {avg_height:.2f} ± {std_height:.2f} m\n")
        log_file.write(f"  Difference: {avg_height - uav_data['rel_alt']:.2f} m\n")
        log_file.write("  Detections:\n")

        for det in detections:
            status = "INLIER" if det.is_inlier else "OUTLIER"
            log_file.write(f"    {status}: width={det.length_pixels:.1f}px, height={det.estimated_height:.1f}m\n")
        log_file.write("\n")

    def _calculate_overall_stats(self, log_file):
        heights = [res['avg_height'] for res in self.results]

        if not heights: return
        # 提取真值列表
        true_alts = [res['actual_altitude'] for res in self.results]
        diffs = [abs(res['avg_height'] - res['actual_altitude']) for res in self.results]
        rel_errors = []
        for res, diff in zip(self.results, diffs):
            alt = res['actual_altitude'] if res['actual_altitude'] != 0 else 1e-5
            rel_errors.append(diff / alt * 100)
        avg_vehicles = np.mean([res['num_inliers'] for res in self.results])
        # [新增] 计算真值统计
        mean_true_alt = np.mean(true_alts)
        min_true_alt = np.min(true_alts)
        max_true_alt = np.max(true_alts)

        # [新增] 计算误差的方差 (基于绝对误差 diffs)
        error_variance = np.var(rel_errors)
        self.overall_stats = {
            'mean_height': np.mean(heights),
            'std_height': np.std(heights),
            'mean_diff': np.mean(diffs),
            'std_diff': np.std(diffs),
            'median_diff': np.median(diffs),
            'min_diff': np.min(diffs),
            'max_diff': np.max(diffs),
            'mean_rel_error': np.mean(rel_errors),
            'median_rel_error': np.median(rel_errors),
            'min_rel_error': np.min(rel_errors),
            'max_rel_error': np.max(rel_errors),
            # [新增] 添加到统计字典中
            'avg_vehicles_used': avg_vehicles,
            # [新增] 存入统计字典
            'mean_true_alt': mean_true_alt,
            'min_true_alt': min_true_alt,
            'max_true_alt': max_true_alt,
            'error_variance': error_variance
        }

        if log_file:  # [修改] 检查 None
            log_file.write("\n=== Overall Statistics ===\n")
            log_file.write(
                f"Average estimated height: {self.overall_stats['mean_height']:.2f} ± {self.overall_stats['std_height']:.2f} m\n")
            log_file.write(
                f"Average difference (Est - Alt): {self.overall_stats['mean_diff']:.2f} ± {self.overall_stats['std_diff']:.2f} m\n")
            log_file.write(f"Median difference: {self.overall_stats['median_diff']:.2f} m\n")
            log_file.write(
                f"Min/Max difference: {self.overall_stats['min_diff']:.2f}/{self.overall_stats['max_diff']:.2f} m\n")
            # ... (原有的日志写入代码保持不变) ...
            log_file.write(
                f"True Altitude: Mean={mean_true_alt:.2f}m, Range=[{min_true_alt:.2f}, {max_true_alt:.2f}]\n")
            log_file.write(f"Error Variance: {error_variance:.4f}\n")
            log_file.write(f"Avg vehicles per image: {avg_vehicles:.1f}\n")  # 可以顺便记录到log
            log_file.write("\n=== Relative Error Analysis ===\n")
            log_file.write(f"Mean relative error: {self.overall_stats['mean_rel_error']:.1f}%\n")
            log_file.write(f"Median relative error: {self.overall_stats['median_rel_error']:.1f}%\n")
            log_file.write(
                f"Error range: {self.overall_stats['min_rel_error']:.1f}% to {self.overall_stats['max_rel_error']:.1f}%\n")

    def _plot_overall_distribution(self, pdf_pages):
        if pdf_pages is None: return  # [修改] 检查 None

        heights = [res['avg_height'] for res in self.results]
        diffs = [res['avg_height'] - res['actual_altitude'] for res in self.results]
        rel_errors = []
        for res, diff in zip(self.results, diffs):
            alt = res['actual_altitude'] if res['actual_altitude'] != 0 else 1e-5
            rel_errors.append(diff / alt * 100)

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.hist(heights, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Estimated height (m)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Estimated Heights')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.hist(diffs, bins=20, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Absolute Error (m)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Absolute Errors')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.hist(rel_errors, bins=20, edgecolor='black', alpha=0.7, color='purple')
        plt.xlabel('Relative Error (%)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Relative Errors')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.boxplot([diffs, rel_errors],
                    labels=['Absolute Error (m)', 'Relative Error (%)'],
                    vert=True,
                    patch_artist=True)
        plt.title('Error Distribution Comparison')
        plt.grid(True)

        plt.tight_layout()
        pdf_pages.savefig()
        plt.close()