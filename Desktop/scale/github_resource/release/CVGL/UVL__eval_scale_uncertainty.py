import os
import re
import sys
import time
import cv2
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utm import from_latlon

from utils import Logger, save_used_code
# 引入项目模块 (请确保路径正确)
from reference.config import UAV_VISLOC_CONFIG
from reference.dataset_adapters import DatasetAdapterFactory
from multi_model.camp.get_camp import get_camp_model

# ================= 配置区域 =================
CONFIG = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    'dataset_root': '/work/documents/UAV_VisLoc_dataset',
    'yaml_dir': 'tif_yamls_UVL_utm',

    # 必须指向包含 40m-150m (或其他范围) 的 HDF5 目录
    'hdf5_dir': 'RESULTS/UVL_utm_h5_DU_rotation_0.8range',

    'CLAHE': False,
    'net_input_size': 384,
    'ckpt_path': r'multi_model/weights/weights_end.pth',

    'error_margin': 0.8,
    'save_base_path': "RESULTS",
    'subfolder_name': "VisLoc_Scale_Uncertainty_utm",
    'run_tag': "DenseUAV_Rotation_0.8"  # 更新Tag
}


# ================= 辅助类 =================
class MultiRegionHDF5Loader:
    def __init__(self, h5_dir, device):
        self.h5_dir = h5_dir
        self.device = device
        self.cache = {}
        self.available_heights = {}

    def load_region(self, region_id):
        if region_id in self.cache: return
        h5_path = os.path.join(self.h5_dir, f"UAV_VisLoc_{region_id}_features.h5")
        if not os.path.exists(h5_path):
            self.cache[region_id] = None
            return

        print(f"  [HDF5] Loading {h5_path}...")
        data_map, heights = {}, []
        try:
            with h5py.File(h5_path, 'r') as f:
                self.utm_sys = f.attrs.get('UTM_SYSTEM', 'Unknown')
                for key in f.keys():
                    if key.startswith('height_'):
                        h = int(key.split('_')[1])
                        feats = torch.from_numpy(f[key]['features'][:])
                        feats = F.normalize(feats, p=2, dim=1)
                        data_map[h] = {
                            'features': feats,
                            'centers': f[key]['centers_utm'][:],
                            'fov_size': float(f[key].attrs['fov_size_meters'])
                        }
                        heights.append(h)
        except Exception as e:
            print(f"  [Error] Load failed: {e}")
            self.cache[region_id] = None
            return

        heights.sort()
        self.cache[region_id] = data_map
        self.available_heights[region_id] = heights

    def get_gallery(self, region_id, height):
        if region_id not in self.cache: self.load_region(region_id)
        r_data = self.cache.get(region_id)
        if not r_data or height not in r_data: return None
        g = r_data[height]
        return {
            'features': g['features'].to(self.device),
            'centers': g['centers'],
            'fov_size': g['fov_size']
        }

    def get_available_heights(self, region_id):
        if region_id not in self.cache: self.load_region(region_id)
        return self.available_heights.get(region_id, [])

    def get_utm_system(self, region_id):
        if region_id not in self.cache: self.load_region(region_id)
        return getattr(self, 'utm_sys', None)

    def clear_cache(self):
        self.cache.clear()
        self.available_heights.clear()
        torch.cuda.empty_cache()


# ================= 辅助函数 =================
def enhance_contrast_clahe(image_rgb, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    使用 LAB 颜色空间的 CLAHE 算法增强图像对比度。
    相比直接对 RGB 操作，这能保持色彩平衡，只增强纹理细节。

    Args:
        image_rgb: 输入的 RGB 图像 (H, W, 3) numpy array
        clip_limit: 对比度限制阈值，值越高对比度越强，但噪声也越大 (建议 2.0 ~ 4.0)
        tile_grid_size: 网格大小，图像被分成多少块进行局部均衡化
    """
    # 1. 转换到 LAB 空间
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    # 2. 分离通道
    l, a, b = cv2.split(lab)

    # 3. 创建 CLAHE 对象并应用到 L (亮度) 通道
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    # 4. 合并通道
    lab_enhanced = cv2.merge((l_enhanced, a, b))

    # 5. 转回 RGB
    image_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return image_enhanced

def calculate_metrics(pred_utm, gt_utm, fov_size):
    dist_error = np.linalg.norm(pred_utm - gt_utm)
    radius = fov_size / 2.0
    return dist_error, radius, 1 if dist_error < 1.6*radius else 0


def get_test_heights(real_h, available_heights, error_margin=0.5):
    low = real_h * (1 - error_margin)
    high = real_h * (1 + error_margin)
    valid = [h for h in available_heights if low <= h <= high]
    if not valid:
        valid = [min(available_heights, key=lambda x: abs(x - real_h))]
    return valid


def generate_and_save_summary(df, save_dir, filename_prefix, title):
    """通用的统计生成与打印函数"""
    if df.empty: return

    # 1. 保存详细数据
    detailed_path = os.path.join(save_dir, f"{filename_prefix}_detailed.csv")
    df.to_csv(detailed_path, index=False)

    # 2. 生成分桶统计
    # 使用 pd.cut 分桶，处理可能的空桶情况
    try:
        df['Rel_Bin'] = pd.cut(df['Rel_H_Error'], bins=10)
        summary = df.groupby('Rel_Bin')[['Pos_Error', 'Success']].agg({
            'Pos_Error': ['mean', 'std', 'count'],
            'Success': 'mean'
        })
        summary.columns = ['Dist_Mean', 'Dist_Std', 'Count', 'Success_Rate']

        print(f"\n=== {title} Summary (by Rel Error) ===")
        # 格式化打印，保留2位小数
        print(summary.to_string(float_format="{:.4f}".format))

        # 保存统计结果
        summary_path = os.path.join(save_dir, f"{filename_prefix}_summary.csv")
        summary.to_csv(summary_path)
    except Exception as e:
        print(f"  [Error] Failed to generate summary for {title}: {e}")


# ================= 主流程 =================
def run():
    # 1. 路径设置
    save_path = os.path.join(CONFIG['save_base_path'], CONFIG['subfolder_name'])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(save_path, f"{timestamp}_{CONFIG['run_tag']}")
    os.makedirs(run_dir, exist_ok=True)

    logger = Logger(os.path.join(run_dir, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger
    save_used_code(run_dir, ignore_patterns=['__pycache__', 'checkpoints'])

    # 2. 模型加载
    print("Loading Model...")
    model = get_camp_model('convnext_base', CONFIG['ckpt_path'], CONFIG['device'])
    model.eval()

    uav_transform = A.Compose([
        A.SmallestMaxSize(max_size=CONFIG['net_input_size']),
        A.CenterCrop(width=CONFIG['net_input_size'], height=CONFIG['net_input_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    h5_loader = MultiRegionHDF5Loader(CONFIG['hdf5_dir'], CONFIG['device'])
    regions = [r for r in UAV_VISLOC_CONFIG['regions'] if int(r) not in UAV_VISLOC_CONFIG['skip_regions']]

    all_results = []  # 总体结果容器

    # 3. 逐区域处理
    for region_id in regions:
        print(f"\n>>> Processing Region {region_id}")
        region_results = []  # 当前区域结果容器

        # --- 初始化 ---
        h5_loader.load_region(region_id)
        avail_heights = h5_loader.get_available_heights(region_id)
        utm_str = h5_loader.get_utm_system(region_id)

        if not avail_heights or not utm_str:
            print(f"Skipping Region {region_id}: Missing HDF5 or Metadata.")
            continue

        try:
            zone_num = int(utm_str[:-1])
        except:
            print(f"Skipping Region {region_id}: Invalid UTM str {utm_str}")
            continue

        adapter_config = {
            'pose_path': os.path.join(CONFIG['dataset_root'], region_id, 'reference.txt'),
            'camera_path': os.path.join(CONFIG['dataset_root'], region_id, 'camera.xml'),
            'dsm_path': os.path.join(CONFIG['dataset_root'], region_id, f"region{region_id}_dsm.tif"),
            'altitude_offset': 0.0
        }
        try:
            adapter = DatasetAdapterFactory.get_adapter('uav_visloc', adapter_config)
        except:
            continue

        img_dir = os.path.join(CONFIG['dataset_root'], region_id, "drone")
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]

        neg_pitch_num = 0
        # --- 遍历图片 ---
        for img_name in tqdm(img_files, desc=f"Reg {region_id}"):
            full_path = os.path.join(img_dir, img_name)
            data = adapter.get_data(full_path)
            if not data: continue

            # try:
            #     e, n, _, _ = from_latlon(data['lat'], data['lon'], force_zone_number=zone_num)
            #     gt_utm = np.array([e, n])
            # except:
            #     continue
            try:
                # 1. 计算 GPS 点的 UTM 坐标 (Nadir Point)
                utm_x_gps, utm_y_gps, _, _ = from_latlon(data['lat'], data['lon'], force_zone_number=zone_num)
                pitch_deg = data['pitch']
                if pitch_deg < -90:
                    neg_pitch_num += 1
                yaw_deg = data['yaw']
                rel_alt = data['rel_alt']

                # 转换为弧度
                pitch_rad = np.deg2rad(pitch_deg)
                yaw_rad = np.deg2rad(yaw_deg)
                # if abs(pitch_deg + 90.0) < 0.1:  # 如果接近垂直向下 (-90)
                #     delta_x = 0.0
                #     delta_y = 0.0
                # else:
                p_calc = -pitch_rad
                y_calc = yaw_rad

                delta_y = rel_alt / np.tan(p_calc) * np.cos(y_calc)
                delta_x = rel_alt / np.tan(p_calc) * np.sin(y_calc)
                # 4. 修正 GT UTM
                gt_utm_x = utm_x_gps + delta_x
                gt_utm_y = utm_y_gps + delta_y
                gt_utm = np.array([gt_utm_x, gt_utm_y])

            except Exception as e:
                # print(f"GT calc error: {e}")
                continue
            real_h = data['rel_alt']

            if real_h < 5: continue

            # compute fov of UAV img
            pitch = data['pitch']
            scale_factor = abs(1 / np.sin(np.deg2rad(pitch)))
            gt_gsd = (real_h) / (data['focal_len']) * scale_factor
            gt_fov = gt_gsd * min(data['width'], data['height'])

            try:
                img_bgr = cv2.imread(full_path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if CONFIG['CLAHE']:
                    img_rgb = enhance_contrast_clahe(img_rgb, clip_limit=3.0, tile_grid_size=(8, 8))

                q_t = uav_transform(image=img_rgb)['image'].unsqueeze(0).to(CONFIG['device'])
                with torch.no_grad():
                    q_feat = F.normalize(model(q_t), p=2, dim=1)
            except:
                continue

            test_heights = get_test_heights(real_h, avail_heights, error_margin=CONFIG['error_margin'])

            for test_h in test_heights:
                gal = h5_loader.get_gallery(region_id, test_h)
                if not gal: continue

                # Max Strategy
                sims = torch.mm(q_feat, gal['features'].T).squeeze()
                max_idx = torch.argmax(sims).item()

                pred_utm = gal['centers'][max_idx]
                fov = gal['fov_size']

                dist_err, radius, success = calculate_metrics(pred_utm, gt_utm, gt_fov)
                rel_h_err = (test_h - real_h) / real_h

                # 记录单条结果
                res_dict = {
                    'Region': region_id,
                    'Image': img_name,
                    'Real_H': real_h,
                    'Test_H': test_h,
                    'Rel_H_Error': rel_h_err,
                    'Pos_Error': dist_err,
                    'Success': success,
                    'FOV_Radius': radius,
                    'satellie_fov_radius': fov/2,
                    'pred_utm': pred_utm,
                    'gt_utm': gt_utm,
                    'pitch': data['pitch']
                }

                region_results.append(res_dict)
                all_results.append(res_dict)
        print(f'neg pitch ratio:{neg_pitch_num/len(img_files)}')
        # --- Region 结束处理 ---
        h5_loader.clear_cache()
        del adapter

        # 立即保存并打印该 Region 的统计信息
        if region_results:
            df_reg = pd.DataFrame(region_results)
            generate_and_save_summary(
                df_reg,
                run_dir,
                filename_prefix=f"region_{region_id}",
                title=f"Region {region_id}"
            )

    # --- 总体统计 ---
    print("\n" + "=" * 50)
    print(" ALL REGIONS PROCESSING COMPLETE ")
    print("=" * 50)

    if all_results:
        df_all = pd.DataFrame(all_results)
        generate_and_save_summary(
            df_all,
            run_dir,
            filename_prefix="overall",
            title="OVERALL (All Regions)"
        )
    else:
        print("No results collected.")


if __name__ == "__main__":
    run()