import os
import re
import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform
import reference.utils  # 旧的utils，用于兼容其他数据集

class DatasetAdapterFactory:
    @staticmethod
    def get_adapter(dataset_type, config):
        """
        返回一个函数 get_image_data_func(img_path) -> dict
        """
        if dataset_type == 'uav_visloc':
            return UAVVisLocAdapter(config)
        elif dataset_type == 'dense_uav':
            return DenseUAVAdapter(config)
        elif dataset_type == 'any_visloc':
            return AnyVisLocAdapter(config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class UAVVisLocAdapter:
    def __init__(self, config):
        """
        初始化适配器，预加载相机参数和位姿数据，打开DSM句柄
        """
        self.pose_path = config.get('pose_path')
        self.camera_path = config.get('camera_path')
        self.dsm_path = config.get('dsm_path')
        # =================【修改点 1：接收偏移量】=================
        # 如果配置里没有传，默认为 0.0
        self.alt_offset = config.get('altitude_offset', 0.0)
        # =======================================================
        # 1. 加载相机内参
        self.camera_intrinsics = self._load_camera_xml(self.camera_path)

        # 2. 加载位姿 Reference.txt
        self.poses = self._load_reference_txt(self.pose_path)

        # 3. 准备 DSM 读取 (延迟读取或保持句柄)
        self.dsm_src = None
        if os.path.exists(self.dsm_path):
            self.dsm_src = rasterio.open(self.dsm_path)
        else:
            print(f"Warning: DSM file not found at {self.dsm_path}")

    def __del__(self):
        # 关闭 DSM 文件句柄
        if self.dsm_src:
            self.dsm_src.close()

    def _load_camera_xml(self, xml_path):
        """解析 camera.xml 获取内参"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = float(root.find('width').text)
            height = float(root.find('height').text)
            f_pix = float(root.find('f').text)
            cx = float(root.find('cx').text)
            cy = float(root.find('cy').text)

            k1 = float(root.find('k1').text)
            k2 = float(root.find('k2').text)
            k3 = float(root.find('k3').text)
            p1 = float(root.find('p1').text)
            p2 = float(root.find('p2').text)

            # [修改点]：这里不要直接存 np.array，而是存 list，方便 JSON 序列化
            dist_coeffs = [k1, k2, p1, p2, k3]

            abs_cx = width / 2.0 + cx
            abs_cy = height / 2.0 + cy

            # [修改点]：存为嵌套列表
            camera_matrix = [
                [f_pix, 0.0, abs_cx],
                [0.0, f_pix, abs_cy],
                [0.0, 0.0, 1.0]
            ]

            diag_pix = np.sqrt(width ** 2 + height ** 2)

            return {
                'width': width,
                'height': height,
                'focal_len': f_pix,
                'cam_size': diag_pix,
                'dist_coeffs': dist_coeffs,  # list
                'camera_matrix': camera_matrix  # list
            }
        except Exception as e:
            print(f"Error loading camera XML: {e}")
            return None

    def _load_reference_txt(self, txt_path):
        """解析 reference.txt 获取精确位姿"""
        poses = {}
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # 跳过以 # 开头的注释行
            header_found = False
            for line in lines:
                if line.startswith('#'):
                    continue

                parts = line.strip().split()  # 假设是空格或制表符分割
                if len(parts) < 20: continue

                # Label X_est Y_est Z_est Yaw_est Pitch_est Roll_est
                # 对应列索引: Label=0, X_est=15, Y_est=16, Z_est=17, Yaw=18, Pitch=19, Roll=20
                # 注意：你提供的示例数据里 Label在第0列，X_est在第15列(从0开始数)

                img_name = parts[0]

                # 根据你提供的示例数据列进行解析
                # 示例: 03_0001.JPG ... ... 119.896856 32.304672 467.517889 ...
                # 请根据实际 split 后的列表检查索引，这里假设是倒数几列或者固定索引
                # 稳妥起见，这里用负索引，因为 _est 通常在最后
                # X_est(Lon), Y_est(Lat), Z_est(Alt), Yaw, Pitch, Roll

                # 假设格式固定为示例中的列数 (21列)
                # X_est=-6, Y_est=-5, Z_est=-4, Yaw=-3, Pitch=-2, Roll=-1
                try:
                    lon = float(parts[-6])
                    lat = float(parts[-5])
                    alt = float(parts[-4])
                    yaw = float(parts[-3])
                    pitch = -(90 - float(parts[-2]))
                    roll = float(parts[-1])

                    # lon = float(parts[1])
                    # lat = float(parts[2])
                    # alt = float(parts[3])
                    # yaw = float(parts[4])
                    # pitch = -(90 - float(parts[5]))
                    # roll = float(parts[6])

                    poses[img_name] = {
                        'lat': lat, 'lon': lon, 'alt': alt,
                        'yaw': yaw, 'pitch': pitch, 'roll': roll
                    }
                except ValueError:
                    continue
            return poses
        except Exception as e:
            print(f"Error loading reference TXT: {e}")
            return {}

    def _get_ground_elevation(self, lon, lat, radius_meter=200):
        """
        获取机下点周围 200m 范围内的 DSM 中位数高程
        """
        # if self.dsm_src is None:
        #     return 0.0

        # try:
        # 1. 将经纬度 (WGS84) 转换为 DSM 的坐标系 (如果是投影坐标系)
        if self.dsm_src.crs != 'EPSG:4326':
            # 使用 rasterio 的转换功能，这里简化处理，假设 lon, lat 直接可用或已转换
            # 严谨做法是使用 rasterio.warp.transform
            xs, ys = transform('EPSG:4326', self.dsm_src.crs, [lon], [lat])
            cx, cy = xs[0], ys[0]
        else:
            cx, cy = lon, lat

        # 2. 计算 200m 对应的像素窗口大小
        # 获取分辨率 (单位: 米/像素 或 度/像素)
        res_x = self.dsm_src.res[0]

        # 如果是经纬度，200m 大约是 0.0018 度；如果是米，就是 200
        # 简单的判断方法：看分辨率大小
        if res_x < 0.1:  # 猜测是度
            radius_val = 200 / 111320.0  # 粗略转换
        else:  # 猜测是米
            radius_val = 200.0

        # 3. 读取窗口数据
        window = from_bounds(
            cx - radius_val, cy - radius_val,
            cx + radius_val, cy + radius_val,
            self.dsm_src.transform
        )

        data = self.dsm_src.read(1, window=window)

        # 4. 计算中位数 (忽略 NoData 值)
        # 假设 NoData 是 -9999 或其他极小值，先掩膜
        valid_data = data[data > -1000]

        if valid_data.size == 0:
            # return 0.0
            raise ValueError(
            f"[Error] No valid elevation data found at Lon:{lon}, Lat:{lat} (Converted: {cx}, {cy}). Check coordinate system match!")
        return float(np.percentile(valid_data, 5))

        # except Exception as e:
        #     # print(f"DSM read error: {e}")
        #     return 0.0

    def get_data(self, img_path):
        img_name = os.path.basename(img_path)

        # 1. 获取位姿
        pose = self.poses.get(img_name)
        if not pose:
            return None

        # 2. 获取地表高程
        ground_h = self._get_ground_elevation(pose['lon'], pose['lat'], radius_meter=200)

        # =================【修改点 2：应用偏移量】=================
        # 修正后的飞行器海拔 = 原始海拔 + 偏移量 (例如 1800)
        aircraft_alt_corrected = pose['alt'] + self.alt_offset

        # 计算相对航高 = 修正后的海拔 - 地表高程
        rel_alt = float(aircraft_alt_corrected - ground_h)
        # =======================================================

        # 4. 组装数据
        if self.camera_intrinsics:
            data = self.camera_intrinsics.copy()
        else:
            return None

        data.update({
            'pitch': pose['pitch'],  # 真值 Pitch
            'roll': pose['roll'],  # 真值 Roll
            'yaw': pose['yaw'],
            'rel_alt': rel_alt,  # 计算后的相对高度
            'lat': pose['lat'],
            'lon': pose['lon']
        })

        return data

class DenseUAVAdapter:
    def __init__(self, config):
        self.params = config['camera_params']

    def get_data(self, img_path):
        img_name = os.path.basename(img_path)
        try:
            # 从文件名解析高度: 000001_H_80.jpg -> 80
            # 兼容不同格式，这里用正则更稳健
            match = re.search(r'_H(\d+)', img_name)
            if match:
                rel_alt = float(match.group(1))
            else:
                return None

            data = self.params.copy()
            data['rel_alt'] = rel_alt
            return data
        except Exception:
            return None


class AnyVisLocAdapter:
    def __init__(self, config):
        pass  # 可能不需要额外配置

    def get_data(self, img_path):
        # 复用 utils.get_image_data (原 Height_estimate_v4 逻辑)
        try:
            return reference.utils.get_image_data(img_path)
        except Exception:
            return None