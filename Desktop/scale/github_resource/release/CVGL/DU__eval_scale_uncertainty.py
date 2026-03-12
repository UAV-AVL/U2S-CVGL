import sys
import os
import re
import time
import h5py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
# 移除 torchvision.transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utm import from_latlon

# 引入项目模块
from utils import save_used_code, Logger
from multi_model.camp.get_camp import get_camp_model

# ================= 配置区域 =================
CONFIG = {
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',

    # 路径配置
    'image_dir': "/work/documents/DenseUAV/all_JPGs/",
    # 'image_dir': "/work/documents/DenseUAV/ALL_test_imgs/",
    'gt_txt_path': "/work/documents/DenseUAV/Dense_GPS_ALL.txt",

    # 指向包含 40m-150m 高度层的 HDF5
    'hdf5_path': "RESULTS/DU_h5__refine_DU_trained_rotation_range0.8/DU_2024_DU_rotation_features.h5",

    # 'ck_path': 'multi_model/weights/DenseUAV_3epochs_e3_0.8992.pth',
    'ck_path': 'multi_model/weights/weights_end.pth',
    # 'ck_path': 'multi_model/weights/AVL-IR-Fusion_weights_e3_0.7557.pth',
    # 'ck_path': 'multi_model/weights/weights_0.9446_for_U1652.pth',

    'uav_input_size': 384,
    'batch_size': 32,
    'save_base_path': "RESULTS",
    'subfolder_name': "Scale_Uncertainty_Exp",
    'run_tag': "DU-train_DU_rptation_refine_0.8range_2024",
    'error_range': 0.8
}


# ================= 核心类 =================
class HDF5Loader:
    def __init__(self, h5_path, device):
        self.h5_path = h5_path
        self.device = device
        self.data = {}
        self.available_heights = []
        self.utm_system_str = None

        print(f"Loading HDF5: {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            # 获取 UTM System
            if 'UTM_SYSTEM' in f.attrs:
                self.utm_system_str = f.attrs['UTM_SYSTEM']
                print(f"  [Info] UTM System from HDF5: {self.utm_system_str}")
            else:
                print(f"  [Warn] UTM_SYSTEM attr not found in HDF5!")

            for key in f.keys():
                if key.startswith('height_'):
                    h = int(key.split('_')[1])
                    feats = torch.from_numpy(f[key]['features'][:]).to(device)
                    feats = F.normalize(feats, p=2, dim=1)

                    self.data[h] = {
                        'features': feats,
                        'centers': f[key]['centers_utm'][:],
                        'fov_size': float(f[key].attrs['fov_size_meters'])
                    }
                    self.available_heights.append(h)
        self.available_heights.sort()
        print(f"  Available Heights: {self.available_heights}")

    def get_gallery(self, height):
        return self.data.get(height, None)

    def get_zone_number(self):
        # 解析 "49N" -> 49
        if self.utm_system_str:
            try:
                # 提取数字部分
                match = re.match(r"(\d+)", self.utm_system_str)
                if match:
                    return int(match.group(1))
            except:
                pass
        return 49  # Default fallback


class DenseUAVGTParser:
    def __init__(self, txt_path):
        self.gt_dict = {}
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            pattern = re.compile(r'.*/(\d+)/.* E([\d.]+) N([\d.]+)\s+')
            for line in lines:
                match = pattern.search(line)
                if match:
                    img_id = match.group(1)
                    if img_id not in self.gt_dict:
                        self.gt_dict[img_id] = {
                            'lon': float(match.group(2)),
                            'lat': float(match.group(3))
                        }
            print(f"Loaded {len(self.gt_dict)} unique scenes from GT file.")

    def get_utm_gt(self, img_name, zone_num, zone_letter='N'):
        match = re.search(r'(\d+)_H', img_name)
        if not match: return None
        img_id = match.group(1)

        if img_id not in self.gt_dict: return None
        info = self.gt_dict[img_id]

        try:
            # 使用动态获取的 Zone Number
            e, n, _, _ = from_latlon(info['lat'], info['lon'],
                                     force_zone_number=zone_num, force_zone_letter=zone_letter)
            return np.array([e, n])
        except:
            return None


# ================= 辅助函数 =================
def calculate_gt_fov_radius(height):
    """
    根据无人机真实高度计算视场半径 (Radius = FOV_Width / 2)
    DenseUAV 相机参数:
    - Sensor Size: 15.864 mm --> 14.667 mm
    - Focal Length: 8.8 mm
    - Image Width: 1440 px, Height: 1080 px
    - Pitch: -90度 (垂直向下)
    """
    # 参数硬编码 (与 config 和 dataset_adapter 一致)
    cam_size = np.sqrt((8.8*4/3)**2 + 8.8**2)  # 13.2 * 8.8 (3:2)  5472*3648(3:2)  4864*3648(4:3) when3:2 -> 15.864
    focal_len = 8.8
    width = 1440
    height_px = 1080  # 1440:1080 = 4:3
    pitch = -90.0

    # 1. 计算对角线像素长度
    diag_px = np.sqrt(width ** 2 + height_px ** 2)

    # 2. 计算 Pitch 带来的缩放因子 (垂直向下时为 1.0)
    # scale_factor = abs(1 / np.sin(np.deg2rad(pitch))) # -90度时 sin(-90)=-1, abs=1
    scale_factor = 1.0

    # 3. 计算 GSD (m/pixel)
    # 公式: GSD = (H * Sensor_Diag) / (Focal * Image_Diag_Px)
    gsd = (height * cam_size) / (focal_len * diag_px) * scale_factor

    # 4. 计算视场物理宽度 (以图像宽度为准，或 min(w,h) 如果做了中心裁剪)
    # 注意：你的预处理代码中如果做了中心裁剪 (384x384)，
    # 那么进入网络的图像对应的是 min(width, height) = 1080 px 的区域。
    # 为了与预存特征库对齐，这里应该计算中心裁剪区域的物理宽度。
    effective_width_px = min(width, height_px)  # 1080

    fov_width_m = gsd * effective_width_px

    return fov_width_m / 2.0

def calculate_metrics(pred_utm, gt_utm, threshold_radius):
    dist_error = np.linalg.norm(pred_utm - gt_utm)
    # radius = fov_size / 2.0
    is_success = 1 if dist_error < 1.6 * threshold_radius else 0
    return dist_error, threshold_radius, is_success


def get_test_heights(real_h, available_heights, error_margin=0.5):
    low = real_h * (1 - error_margin)
    high = real_h * (1 + error_margin)
    valid = [h for h in available_heights if low <= h <= high]
    if not valid:
        nearest = min(available_heights, key=lambda x: abs(x - real_h))
        valid = [nearest]
    return valid


def generate_summary(df, save_dir, prefix, title):
    if df.empty: return

    # Save Detailed
    df.to_csv(os.path.join(save_dir, f"{prefix}_detailed.csv"), index=False)

    # Save Summary
    df['Rel_Bin'] = pd.cut(df['Rel_H_Error'], bins=10)
    summary = df.groupby('Rel_Bin')[['Pos_Error', 'Success']].agg({
        'Pos_Error': ['mean', 'std', 'count'],
        'Success': 'mean'
    })
    summary.columns = ['Dist_Mean', 'Dist_Std', 'Count', 'Success_Rate']

    print(f"\n=== {title} Summary (by Rel Error) ===")
    print(summary.to_string(float_format="{:.2f}".format))
    summary.to_csv(os.path.join(save_dir, f"{prefix}_summary.csv"))


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
    model = get_camp_model('convnext_base', CONFIG['ck_path'], CONFIG['device'])
    model.eval()

    # =========================================================
    # 【修改点】 Albumentations 预处理 (含中心裁剪)
    # 逻辑：
    # 1. SmallestMaxSize(384): 将图像短边缩放到384 (e.g., 1440x1080 -> 512x384)
    # 2. CenterCrop(384, 384): 裁剪中心正方形区域 (e.g., 512x384 -> 384x384)
    # 3. Normalize & ToTensorV2
    # =========================================================
    uav_transform = A.Compose([
        A.SmallestMaxSize(max_size=CONFIG['uav_input_size']),
        A.CenterCrop(width=CONFIG['uav_input_size'], height=CONFIG['uav_input_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    h5_loader = HDF5Loader(CONFIG['hdf5_path'], CONFIG['device'])
    # 从 HDF5 获取 Zone
    zone_num = h5_loader.get_zone_number()
    zone_let = h5_loader.utm_system_str[-1]
    print(f"Using UTM Zone: {h5_loader.utm_system_str}")

    gt_parser = DenseUAVGTParser(CONFIG['gt_txt_path'])
    img_files = [f for f in os.listdir(CONFIG['image_dir']) if f.lower().endswith('.jpg')]

    # 按照真实高度分组处理
    files_by_height = {}
    for f in img_files:
        m = re.search(r'_H(\d+)', f)
        if m:
            h = int(m.group(1))
            if h not in files_by_height: files_by_height[h] = []
            files_by_height[h].append(f)

    sorted_heights = sorted(files_by_height.keys())
    print(f"Found Image Groups (Heights): {sorted_heights}")

    all_results = []

    for real_h in sorted_heights:
        current_files = files_by_height[real_h]
        print(f"\n>>> Processing Height Group: {real_h}m ({len(current_files)} images)")

        group_results = []

        for img_name in tqdm(current_files):
            # A. GT
            gt_utm = gt_parser.get_utm_gt(img_name, zone_num, zone_let)  # 传入 zone_num
            if gt_utm is None: continue

            # B. Feature (Albumentations)
            try:
                path = os.path.join(CONFIG['image_dir'], img_name)
                # OpenCV 读取 (BGR) -> 转 RGB
                img_bgr = cv2.imread(path)
                if img_bgr is None: continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                # Albumentations 调用
                augmented = uav_transform(image=img_rgb)
                t = augmented['image'].unsqueeze(0).to(CONFIG['device'])

                with torch.no_grad():
                    q_feat = F.normalize(model(t), p=2, dim=1)
            except Exception as e:
                print(f"Err processing {img_name}: {e}")
                continue

            # C. Test Heights
            test_heights = get_test_heights(real_h, h5_loader.available_heights, CONFIG['error_range'])

            gt_radius = calculate_gt_fov_radius(real_h)
            # D. Loop
            for test_h in test_heights:
                gal = h5_loader.get_gallery(test_h)
                if not gal: continue

                sims = torch.mm(q_feat, gal['features'].T).squeeze()
                max_idx = torch.argmax(sims).item()

                dist, rad, succ = calculate_metrics(gal['centers'][max_idx], gt_utm, gt_radius)
                rel_err = (test_h - real_h) / real_h

                res = {
                    'Image': img_name, 'Real_H': real_h, 'Test_H': test_h,
                    'Rel_H_Error': rel_err, 'Pos_Error': dist, 'Success': succ, 'FOV_Radius': gt_radius,
                    'satellite_fov_radius': gal['fov_size']/2,
                    'pred_utm': gal['centers'][max_idx],
                    'gt_utm': gt_utm
                }
                group_results.append(res)
                all_results.append(res)

        # 保存该高度组的统计信息
        if group_results:
            df_grp = pd.DataFrame(group_results)
            generate_summary(df_grp, run_dir, f"scale_{real_h}m", f"Height {real_h}m")

    # 总体统计
    print("\n" + "=" * 50)
    print(" ALL HEIGHTS PROCESSING COMPLETE ")
    print("=" * 50)
    if all_results:
        df_all = pd.DataFrame(all_results)
        generate_summary(df_all, run_dir, "scale_overall", "OVERALL")


if __name__ == "__main__":
    run()