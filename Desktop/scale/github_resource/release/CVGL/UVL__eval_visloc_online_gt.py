import os
import re
import sys
import time
import cv2
import yaml
import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utm import from_latlon

# 引入项目模块
from utils import Logger, save_used_code
from reference.config import UAV_VISLOC_CONFIG
from reference.dataset_adapters import DatasetAdapterFactory
from multi_model.camp.get_camp import get_camp_model

# ================= 配置区域 =================
CONFIG = {
    'device': 'cuda:2' if torch.cuda.is_available() else 'cpu',

    'dataset_root': '/work/documents/UAV_VisLoc_dataset',
    'yaml_dir': 'tif_yamls_UVL_utm',

    'net_input_size': 384,
    'step_overlap': 0.5,
    'batch_size': 32,
    'ckpt_path': r'multi_model/weights/weights_end.pth',

    'CLAHE': False,
    'save_base_path': "RESULTS",
    'subfolder_name': "VisLoc_GT_Only",
    'run_tag': "DU_rotation_only_GT"
}


# ================= 辅助类 =================
class ConfigLoader:
    def __init__(self, region_id):
        yaml_path = os.path.join(CONFIG['yaml_dir'], f"UAV_VisLoc_Region{region_id}.yaml")
        if not os.path.exists(yaml_path):
            yaml_path = os.path.join(CONFIG['yaml_dir'], f"UAV_VisLoc_{region_id}.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML not found for region {region_id}")
        with open(yaml_path, 'r') as f:
            self.meta = yaml.safe_load(f)


# ================= 在线生成器 =================
class SingleImageOnlineGenerator:
    def __init__(self, meta_config, model, device, transform):
        self.meta = meta_config.meta
        self.model = model
        self.device = device
        self.transform = transform

        ref_path = self.meta['REF_path']
        if not os.path.exists(ref_path):
            print(f"[Critical] Satellite Map not found: {ref_path}")
            sys.exit(-1)  # 移除防御，直接退出

        print(f"  [Load] Satellite Map: {ref_path}")
        self.ref_img = cv2.imread(ref_path)
        if self.ref_img is None:
            print(f"[Critical] Failed to read image content: {ref_path}")
            sys.exit(-1)

        self.ref_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
        self.H_orig, self.W_orig = self.ref_img.shape[:2]

    def generate_gallery(self, gsd, cam_params):
        # 这里的 fov_width 是用于决定卫星图裁剪范围的
        fov_satellite_patch = gsd * min(cam_params['width'], cam_params['height'])

        target_res = fov_satellite_patch / CONFIG['net_input_size']
        scale = self.meta['resolution'] / target_res

        new_w = int(self.W_orig * scale)
        new_h = int(self.H_orig * scale)

        # 保护逻辑：如果缩放后太小无法切片，返回空
        if new_w < CONFIG['net_input_size'] or new_h < CONFIG['net_input_size']: return None
        # 保护逻辑：如果缩放后太大爆内存，返回空 (这是物理限制，非代码逻辑错误)
        if new_w > 35000 or new_h > 35000: return None

        resized_map = cv2.resize(self.ref_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        step = int(CONFIG['net_input_size'] * (1 - CONFIG['step_overlap']))
        if step < 1: step = 1

        batch_imgs, centers_list, feats_list = [], [], []
        ref_x, ref_y, ref_res = self.meta['REF_initialX'], self.meta['REF_initialY'], self.meta['resolution']

        y_range = list(range(0, new_h - CONFIG['net_input_size'] + 1, step))
        if y_range[-1] + CONFIG['net_input_size'] < new_h: y_range.append(new_h - CONFIG['net_input_size'])
        x_range = list(range(0, new_w - CONFIG['net_input_size'] + 1, step))
        if x_range[-1] + CONFIG['net_input_size'] < new_w: x_range.append(new_w - CONFIG['net_input_size'])

        for y in y_range:
            for x in x_range:
                patch = resized_map[y:y + CONFIG['net_input_size'], x:x + CONFIG['net_input_size']]
                batch_imgs.append(patch)  # RGB Numpy

                cx_sc = x + CONFIG['net_input_size'] / 2.0
                cy_sc = y + CONFIG['net_input_size'] / 2.0
                cx_orig = cx_sc / scale
                cy_orig = cy_sc / scale

                ux = ref_x + (cx_orig * ref_res)
                uy = ref_y - (cy_orig * ref_res)
                centers_list.append([ux, uy])

                if len(batch_imgs) >= CONFIG['batch_size']:
                    self._process_batch(batch_imgs, feats_list)
                    batch_imgs = []

        if batch_imgs: self._process_batch(batch_imgs, feats_list)
        if not feats_list: return None

        return {
            'features': torch.cat(feats_list, dim=0),
            'centers': np.array(centers_list),
            'fov_size': fov_satellite_patch
        }

    def _process_batch(self, img_list, feats_list):
        # Albumentations Transform
        tensors = []
        for img in img_list:
            aug = self.transform(image=img)
            tensors.append(aug['image'])
        batch_t = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            out = self.model(batch_t)
            out = F.normalize(out, p=2, dim=1)
            feats_list.append(out)


# ================= 辅助函数 =================
def enhance_contrast_clahe(image_rgb, clip_limit=3.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def calculate_metrics(pred_utm, gt_utm, fov_size):
    dist_error = np.linalg.norm(pred_utm - gt_utm)
    # 半径计算与 Scale Uncertainty 代码保持一致
    radius = fov_size / 2.0
    # Success if error < 1.6 * radius
    is_succ = 1 if dist_error < 1.6 * radius else 0
    return dist_error, radius, is_succ


def generate_summary(df, save_dir, filename_prefix, title):
    if df.empty: return
    # 关键：列名对齐
    # 确保 CSV 包含: Region, Image, Real_H, Test_H, Rel_H_Error, Pos_Error, Success, FOV_Radius
    df.to_csv(os.path.join(save_dir, f"{filename_prefix}_detailed.csv"), index=False)

    # 简单的聚合统计
    mean_dist = df['Pos_Error'].mean()
    med_dist = df['Pos_Error'].median()
    succ_rate = df['Success'].mean() * 100

    print(f"\n=== {title} Summary ===")
    print(f"Mean Error: {mean_dist:.2f} m")
    print(f"Median Error: {med_dist:.2f} m")
    print(f"Success Rate: {succ_rate:.2f} %")

    # 保存 Summary CSV
    sum_df = pd.DataFrame([{
        'Mean_Error': mean_dist, 'Median_Error': med_dist, 'Success_Rate': succ_rate, 'Count': len(df)
    }])
    sum_df.to_csv(os.path.join(save_dir, f"{filename_prefix}_summary.csv"), index=False)


# ================= 主流程 =================
def run():
    save_path = os.path.join(CONFIG['save_base_path'], CONFIG['subfolder_name'])
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(save_path, f"{timestamp}_{CONFIG['run_tag']}")
    os.makedirs(run_dir, exist_ok=True)

    logger = Logger(os.path.join(run_dir, 'log.txt'))
    sys.stdout = logger
    sys.stderr = logger
    save_used_code(run_dir, ignore_patterns=['__pycache__'])

    print("Loading Model...")
    model = get_camp_model('convnext_base', CONFIG['ckpt_path'], CONFIG['device'])
    model.eval()

    # 与 UVL__eval_scale_uncertainty 一致的预处理
    uav_transform = A.Compose([
        A.SmallestMaxSize(max_size=CONFIG['net_input_size']),
        A.CenterCrop(width=CONFIG['net_input_size'], height=CONFIG['net_input_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    regions = [r for r in UAV_VISLOC_CONFIG['regions'] if int(r) not in UAV_VISLOC_CONFIG['skip_regions']]
    all_results = []

    for region_id in regions:
        print(f"\n>>> Processing Region {region_id} (GT Only)")

        # 1. 初始化生成器
        try:
            meta_cfg = ConfigLoader(region_id)
            online_gen = SingleImageOnlineGenerator(meta_cfg, model, CONFIG['device'], uav_transform)
        except Exception as e:
            print(f"[Critical] Generator Init Fail: {e}")
            # sys.exit(-1)  # 移除防御，暴露问题
            continue
        # 2. 获取 Zone Number
        try:
            utm_str = meta_cfg.meta.get('UTM_SYSTEM', '49N')
            zone_num = int(utm_str[:-1])
        except:
            print("[Critical] Lack zone num in YAML")
            sys.exit(-1)

        # 3. Adapter
        adapter_config = {
            'pose_path': os.path.join(CONFIG['dataset_root'], region_id, 'reference.txt'),
            'camera_path': os.path.join(CONFIG['dataset_root'], region_id, 'camera.xml'),
            'dsm_path': os.path.join(CONFIG['dataset_root'], region_id, f"region{region_id}_dsm.tif"),
            'altitude_offset': 0.0
        }
        try:
            adapter = DatasetAdapterFactory.get_adapter('uav_visloc', adapter_config)
        except Exception as e:
            print(f"[Critical] Adapter Init Fail: {e}")
            sys.exit(-1)

        img_dir = os.path.join(CONFIG['dataset_root'], region_id, "drone")
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]

        region_results = []

        for img_name in tqdm(img_files, desc=f"Reg {region_id}"):
            full_path = os.path.join(img_dir, img_name)
            data = adapter.get_data(full_path)
            if not data: continue

            # --- GT Calculation (Pitch/Yaw Corrected) ---
            try:
                utm_x_gps, utm_y_gps, _, _ = from_latlon(data['lat'], data['lon'], force_zone_number=zone_num)

                pitch_rad = np.deg2rad(data['pitch'])
                yaw_rad = np.deg2rad(data['yaw'])
                rel_alt = data['rel_alt']

                p_calc = -pitch_rad
                y_calc = yaw_rad

                delta_y = rel_alt / np.tan(p_calc) * np.cos(y_calc)
                delta_x = rel_alt / np.tan(p_calc) * np.sin(y_calc)

                gt_utm = np.array([utm_x_gps + delta_x, utm_y_gps + delta_y])

            except:
                continue

            real_h = data['rel_alt']
            if real_h < 5: continue

            # --- FOV Calculation (Based on Real Height) ---
            pitch = data['pitch']
            scale_factor = abs(1 / np.sin(np.deg2rad(pitch)))
            gt_gsd = (real_h * data['cam_size']) / (data['focal_len'] *
                                                    math.sqrt(data['width'] ** 2 + data['height'] ** 2)) * scale_factor
            gt_fov = gt_gsd * min(data['width'], data['height'])

            # --- Feature Extraction ---
            try:
                img_bgr = cv2.imread(full_path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if CONFIG['CLAHE']:
                    img_rgb = enhance_contrast_clahe(img_rgb)

                aug = uav_transform(image=img_rgb)
                q_t = aug['image'].unsqueeze(0).to(CONFIG['device'])

                with torch.no_grad():
                    q_feat = F.normalize(model(q_t), p=2, dim=1)
            except:
                continue

            # --- Online Gallery Generation (Based on Real/GT Height) ---
            cam_params = {
                'focal_len': data['focal_len'], 'width': data['width'],
                'height': data['height'], 'cam_size': data['cam_size']
            }

            gal = online_gen.generate_gallery(gt_gsd, cam_params)
            if not gal: continue

            # --- Matching (Max Only) ---
            sims = torch.mm(q_feat, gal['features'].T).squeeze()
            max_idx = torch.argmax(sims).item()

            pred_utm = gal['centers'][max_idx]

            # --- Metric (Using GT FOV) ---
            dist, rad, succ = calculate_metrics(pred_utm, gt_utm, gt_fov)

            # --- Save Result (Aligned Columns) ---
            # Columns: Region, Image, Real_H, Test_H, Rel_H_Error(0), Pos_Error, Success, FOV_Radius
            res = {
                'Region': region_id,
                'Image': img_name,
                'Real_H': real_h,
                'Test_H': real_h,  # Test = Real
                'Rel_H_Error': 0.0,  # Error = 0
                'Pos_Error': dist,
                'Success': succ,
                'FOV_Radius': rad,
                'satellite_fov_radius': gal['fov_size'] / 2.0,
                'pred_utm': pred_utm,
                'gt_utm': gt_utm,
                'pitch': data['pitch']
            }
            region_results.append(res)
            all_results.append(res)

        del online_gen
        del adapter

        # Save Region
        if region_results:
            df_reg = pd.DataFrame(region_results)
            generate_summary(df_reg, run_dir, f"gt_only_region_{region_id}", f"Region {region_id}")

    # Save Overall
    if all_results:
        df_all = pd.DataFrame(all_results)
        generate_summary(df_all, run_dir, "gt_only_overall", "OVERALL")


if __name__ == "__main__":
    run()