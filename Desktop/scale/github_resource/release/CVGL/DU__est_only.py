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

from utils import Logger, save_used_code
from multi_model.camp.get_camp import get_camp_model

# ================= Config Region =================
CONFIG = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'image_dir': "/work/documents/DenseUAV/ALL_test_imgs/",
    'gt_txt_path': "/work/documents/DenseUAV/Dense_GPS_ALL.txt",
    'est_height_csv': "reference/DenseUAV_new/DenseUAV_conf0.5/summary.csv",
    'yaml_path': "tif_yamls/DenseUAV_2024_4_18_L20.yaml",

    'net_input_size': 384,
    'step_overlap': 0.5,
    'batch_size': 16,

    'ckpt_path': r'your_weight_folder/xxx.pth',

    'CLAHE': False,
    'save_base_path': "RESULTS",
    'subfolder_name': "DenseUAV_Est_Only",
    'run_tag': "DU_rotation_TEST_refine_2024"
}


# ================= 辅助类 =================
class ConfigLoader:
    def __init__(self, yaml_path):
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML not found: {yaml_path}")
        with open(yaml_path, 'r') as f:
            self.meta = yaml.safe_load(f)


class EstHeightParser:
    def __init__(self, csv_path):
        self.height_map = {}
        if not os.path.exists(csv_path):
            print(f"[Warn] Est height CSV not found: {csv_path}")
            sys.exit(-1)

        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                # DenseUAV ID 格式通常为 000018_H80
                img_id = str(row['Image_ID']).strip()
                if img_id.endswith('.jpg'): img_id = img_id[:-4]
                try:
                    h = float(row['Estimated_Height(m)'])
                    self.height_map[img_id] = h
                except:
                    continue
            print(f"Loaded {len(self.height_map)} estimated heights.")
        except Exception as e:
            print(f"[Error] Failed to load CSV: {e}")
            sys.exit(-1)

    def get_height(self, filename):
        name_no_ext = os.path.splitext(filename)[0]
        return self.height_map.get(name_no_ext, None)


class DenseUAVGTParser:
    def __init__(self, txt_path):
        self.gt_dict = {}
        if not os.path.exists(txt_path):
            print(f"[Error] GT TXT not found: {txt_path}")
            sys.exit(-1)

        with open(txt_path, 'r') as f:
            lines = f.readlines()
        pattern = re.compile(r'.*/(\d+)/.* E([\d.]+) N([\d.]+)\s+')
        for line in lines:
            match = pattern.search(line)
            if match:
                # DenseUAV ID: 000018 -> (Lon, Lat)
                self.gt_dict[match.group(1)] = {
                    'lon': float(match.group(2)),
                    'lat': float(match.group(3))
                }
        print(f"Loaded {len(self.gt_dict)} GT records.")

    def get_utm_gt(self, img_name, zone_num=50):
        # 000018_H80.jpg -> ID 000018
        match = re.search(r'(\d+)_H', img_name)
        if not match: return None
        img_id = match.group(1)

        if img_id not in self.gt_dict: return None
        info = self.gt_dict[img_id]

        try:
            e, n, _, _ = from_latlon(info['lat'], info['lon'], force_zone_number=zone_num)
            return np.array([e, n])
        except:
            return None


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
            sys.exit(-1)

        print(f"  [Load] Satellite Map: {ref_path}")
        self.ref_img = cv2.imread(ref_path)
        if self.ref_img is None:
            print(f"[Critical] Failed to read image content: {ref_path}")
            sys.exit(-1)

        self.ref_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
        self.H_orig, self.W_orig = self.ref_img.shape[:2]

    def generate_gallery(self, height, cam_params):
        pitch_rad = np.deg2rad(-90.0)
        scale_factor = abs(1 / np.sin(pitch_rad))

        # GSD = H / f_pix * scale
        gsd = (height / cam_params['focal_len']) * scale_factor

        # 【关键对齐】FOV Width 基于中心裁剪 (min dim)
        min_dim = min(cam_params['width'], cam_params['height'])
        fov_width_m = gsd * min_dim

        target_res = fov_width_m / CONFIG['net_input_size']
        scale = self.meta['resolution'] / target_res

        new_w = int(self.W_orig * scale)
        new_h = int(self.H_orig * scale)

        # 物理限制保护
        if new_w < CONFIG['net_input_size'] or new_h < CONFIG['net_input_size']: return None
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
            'fov_size': fov_width_m
        }

    def _process_batch(self, img_list, feats_list):
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


def calculate_gt_fov_radius(height):
    # DenseUAV 相机参数 (固定)
    cam_size = np.sqrt((8.8*4/3)**2 + 8.8**2)
    focal_len = 8.8
    width = 1440
    height_px = 1080

    # 假设垂直向下 Pitch=-90, scale=1.0
    diag_px = np.sqrt(width ** 2 + height_px ** 2)
    gsd = (height * cam_size) / (focal_len * diag_px)

    effective_width_px = min(width, height_px)  # 1080
    fov_width_m = gsd * effective_width_px

    return fov_width_m / 2.0


def calculate_metrics(pred_utm, gt_utm, threshold_radius):
    dist_error = np.linalg.norm(pred_utm - gt_utm)
    # 判定阈值：距离 < sqrt(2) * GT_Radius
    is_success = 1 if dist_error < math.sqrt(2) * threshold_radius else 0
    return dist_error, threshold_radius, is_success


def generate_summary(df, save_dir, filename_prefix, title):
    if df.empty: return

    # Columns Aligned: Image, Real_H, Test_H, Rel_H_Error, Pos_Error, Success, FOV_Radius
    df.to_csv(os.path.join(save_dir, f"{filename_prefix}_detailed.csv"), index=False)

    # Simple Aggregation
    mean_dist = df['Pos_Error'].mean()
    med_dist = df['Pos_Error'].median()
    succ_rate = df['Success'].mean() * 100

    print(f"\n=== {title} Summary ===")
    print(f"Mean Error: {mean_dist:.2f} m")
    print(f"Median Error: {med_dist:.2f} m")
    print(f"Success Rate: {succ_rate:.2f} %")

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

    # Albumentations with CenterCrop
    uav_transform = A.Compose([
        A.SmallestMaxSize(max_size=CONFIG['net_input_size']),
        A.CenterCrop(width=CONFIG['net_input_size'], height=CONFIG['net_input_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 1. Initialize Components
    meta_cfg = ConfigLoader(CONFIG['yaml_path'])
    est_parser = EstHeightParser(CONFIG['est_height_csv'])
    gt_parser = DenseUAVGTParser(CONFIG['gt_txt_path'])
    online_gen = SingleImageOnlineGenerator(meta_cfg, model, CONFIG['device'], uav_transform)

    # Get Zone Number
    try:
        utm_str = meta_cfg.meta['UTM_SYSTEM']
        zone_num = int(utm_str[:-1])
    except KeyError:
        print(f'[Critical] YAML文件缺少必须的 UTM_SYSTEM 字段')
        print(f'请检查 YAML 文件: {CONFIG["yaml_path"]}')
        sys.exit(-1)
    except ValueError:
        print(f'[Critical] UTM_SYSTEM 格式错误: {utm_str}')
        print(f'应该为数字+字母格式，如 "49N"')
        sys.exit(-1)
    except Exception as e:
        print(f'[Critical] 解析 UTM 时发生未知错误: {e}')
        sys.exit(-1)

    img_files = [f for f in os.listdir(CONFIG['image_dir']) if f.lower().endswith('.jpg')]
    all_results = []

    print(f"Start Evaluation on {len(img_files)} images...")

    for img_name in tqdm(img_files):
        # A. Check Est Height
        est_h = est_parser.get_height(img_name)
        if est_h is None: continue  # Skip if no est height

        # B. Get Real Height (from filename)
        match_h = re.search(r'_H(\d+)', img_name)
        if not match_h: continue
        real_h = int(match_h.group(1))

        # C. Get GT UTM
        gt_utm = gt_parser.get_utm_gt(img_name, zone_num)
        if gt_utm is None: continue

        # D. Feature Extraction
        try:
            path = os.path.join(CONFIG['image_dir'], img_name)
            img_bgr = cv2.imread(path)
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

        # E. Online Gallery Generation (Based on Est Height)
        # DenseUAV Cam Params (Fixed)
        cam_params = {'cam_size': np.sqrt((8.8*4/3)**2 + 8.8**2),
                      'focal_len': 8.8,
                      'width': 1440,
                      'height': 1080,
                      'pitch': -90.0}
        # Pixel focal length for GSD calculation:
        # F_px = (F_mm / Sensor_mm) * Diag_px
        diag_px = math.sqrt(cam_params['width'] ** 2 + cam_params['height'] ** 2)
        cam_params['focal_len'] = (cam_params['focal_len'] / cam_params['cam_size']) * diag_px

        gal = online_gen.generate_gallery(est_h, cam_params)
        if not gal: continue

        # F. Matching (Max Only)
        sims = torch.mm(q_feat, gal['features'].T).squeeze()
        max_idx = torch.argmax(sims).item()
        pred_utm = gal['centers'][max_idx]

        # G. Metric (Using GT FOV Radius)
        gt_radius = calculate_gt_fov_radius(real_h)
        dist, rad, succ = calculate_metrics(pred_utm, gt_utm, gt_radius)
        rel_err = (est_h - real_h) / real_h

        # H. Record
        res = {
            'Image': img_name,
            'Real_H': real_h,
            'Test_H': est_h,
            'Rel_H_Error': rel_err,
            'Pos_Error': dist,
            'Success': succ,
            'FOV_Radius': rad,
            'satellite_fov_radius': gal['fov_size'] / 2.0,
            'pred_utm': pred_utm,
            'gt_utm': gt_utm
        }
        all_results.append(res)

    del online_gen

    if all_results:
        df_all = pd.DataFrame(all_results)
        generate_summary(df_all, run_dir, "est_only_overall", "OVERALL (Est Only)")
    else:
        print("No results collected.")


if __name__ == "__main__":
    run()