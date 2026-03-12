import os
import cv2
import yaml
import h5py
import torch
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

# === 引入项目模块 ===
from reference.config import UAV_VISLOC_CONFIG
from reference.dataset_adapters import DatasetAdapterFactory
from multi_model.camp.get_camp import get_camp_model

# ================= 脚本配置 =================
SCRIPT_CONFIG = {
    'device': 'cuda:1' if torch.cuda.is_available() else 'cpu',
    'yaml_dir': 'tif_yamls_UVL_utm',
    'output_dir': 'RESULTS/UVL_utm_h5_DU_rotation_0.8range',
    'net_input_size': 384,
    'step_overlap': 0.5,
    'batch_size': 24,
    'num_workers': 4,  # 启用多进程加载
    'relative': 0.8,
    'ckpt_path': r'multi_model/weights/weights_end.pth'
}


# ================= 数据集类 (Albumentations优化) =================
class SliceDataset(Dataset):
    def __init__(self, img_array, windows, transform=None):
        """
        img_array: 缩放后的卫星图 (H, W, 3) BGR 格式
        windows: List of (x, y) 坐标
        transform: Albumentations 变换流水线
        """
        self.img = img_array
        self.windows = windows
        self.transform = transform
        self.crop_size = SCRIPT_CONFIG['net_input_size']

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y = self.windows[idx]
        # Crop (OpenCV 格式: y:y+h, x:x+w)
        patch = self.img[y: y + self.crop_size, x: x + self.crop_size]

        # BGR -> RGB (Albumentations 默认期望 RGB)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=patch)
            patch_tensor = augmented['image']
        else:
            # Fallback
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        return patch_tensor, np.array([x, y])


# ================= 辅助函数 =================

def calculate_gsd(height, camera_params):
    pitch_rad = np.deg2rad(-90.0)
    scale_factor = abs(1 / np.sin(pitch_rad))
    diag_px = math.sqrt(camera_params['width'] ** 2 + camera_params['height'] ** 2)
    gsd = (height * camera_params['cam_size']) / (camera_params['focal_len'] * diag_px) * scale_factor
    # a = diag_px / camera_params['height']
    # b = diag_px / camera_params['focal_len']
    return gsd


def get_region_metadata_via_adapter(region_id, global_cfg):
    root_dir = global_cfg['root_dir']
    alt_offset = 0.0
    adapter_config = {
        'pose_path': os.path.join(root_dir, region_id, global_cfg['pose_file_name']),
        'camera_path': os.path.join(root_dir, region_id, global_cfg['camera_file_name']),
        'dsm_path': os.path.join(root_dir, region_id, f"region{region_id}{global_cfg['dsm_suffix']}"),
        'altitude_offset': alt_offset
    }

    if not os.path.exists(adapter_config['pose_path']):
        print(f"[Warn] Pose file missing: {adapter_config['pose_path']}")
        return None, None

    try:
        adapter = DatasetAdapterFactory.get_adapter('uav_visloc', adapter_config)
    except Exception as e:
        print(f"[Error] Failed to init adapter: {e}")
        return None, None

    img_dir = os.path.join(root_dir, region_id, "drone")
    if not os.path.exists(img_dir): return None, None

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_heights = []
    cam_params = None

    print(f"  Scanning {len(img_files)} images in Region {region_id} using Adapter...")

    for img_name in tqdm(img_files, leave=False):
        img_full_path = os.path.join(img_dir, img_name)
        data = adapter.get_data(img_full_path)
        if data:
            all_heights.append(data['rel_alt'])
            if cam_params is None:
                cam_params = {
                    'cam_size': data['cam_size'],
                    'focal_len': data['focal_len'],
                    'width': data['width'],
                    'height': data['height']
                }

    valid_heights = [h for h in all_heights if h > 0]
    if not valid_heights:
        print("  [Error] No valid positive heights found.")
        return None, None

    del adapter
    return valid_heights, cam_params


def generate_height_list(heights):
    if not heights: return []
    min_h = min(heights)
    max_h = max(heights)
    print(f"  [Info] Adapter Rel_Alt Range: {min_h:.2f}m - {max_h:.2f}m")

    relative = SCRIPT_CONFIG['relative']
    lower = min_h * (1-relative)
    upper = max_h * (1+relative)
    start_h = int(np.floor(lower / 5.0) * 5)
    end_h = int(np.ceil(upper / 5.0) * 5)
    if start_h < 5: start_h = 5
    return list(range(start_h, end_h + 1, 5))


# ================= 核心处理流程 =================

def process_region(region_id, model, script_cfg, dataset_cfg):
    print(f"\n{'=' * 50}")
    print(f"Processing Region: {region_id}")

    # 1. Adapter Metadata
    heights_data, cam_params = get_region_metadata_via_adapter(region_id, dataset_cfg)
    if not heights_data or not cam_params:
        print("  [Skip] No valid data from adapter.")
        return

    height_list = generate_height_list(heights_data)
    print(f"  [Plan] Computing {len(height_list)} levels: {height_list}")

    # 2. Load Satellite Map (OpenCV)
    yaml_path = os.path.join(script_cfg['yaml_dir'], f"UAV_VisLoc_Region{region_id}.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(script_cfg['yaml_dir'], f"UAV_VisLoc_{region_id}.yaml")

    if not os.path.exists(yaml_path):
        print(f"  [Error] YAML not found: {yaml_path}")
        return

    with open(yaml_path, 'r') as f:
        meta = yaml.safe_load(f)

    ref_path = meta['REF_path']
    if not os.path.exists(ref_path):
        filename = os.path.basename(ref_path)
        possible_path = os.path.join(dataset_cfg['root_dir'], region_id, filename)
        if os.path.exists(possible_path):
            ref_path = possible_path
        else:
            print(f"  [Error] Image not found: {ref_path}")
            return

    print(f"  [Load] Satellite Map: {ref_path}")
    # 读取 BGR
    sat_img = cv2.imread(ref_path)
    if sat_img is None:
        print("  [Error] Failed to read image.")
        return
    H_orig, W_orig = sat_img.shape[:2]

    # 3. Setup Output & Transform
    os.makedirs(script_cfg['output_dir'], exist_ok=True)
    h5_path = os.path.join(script_cfg['output_dir'], f"UAV_VisLoc_{region_id}_features.h5")

    # Albumentations Transform
    alb_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    with h5py.File(h5_path, 'w') as h5f:
        h5f.attrs['UTM_SYSTEM'] = meta['UTM_SYSTEM']
        h5f.attrs['REF_PATH'] = ref_path
        h5f.attrs['REF_RESOLUTION'] = meta['resolution']

        for height in tqdm(height_list, desc=f"Reg {region_id}"):
            # --- Scale ---
            gsd = calculate_gsd(height, cam_params)
            # FOV = GSD * min(W, H)
            min_size = min(cam_params['width'], cam_params['height'])
            fov_width_m = gsd * min_size

            target_res = fov_width_m / script_cfg['net_input_size']
            scale = meta['resolution'] / target_res

            new_w = int(W_orig * scale)
            new_h = int(H_orig * scale)

            if new_w < script_cfg['net_input_size'] or new_h < script_cfg['net_input_size']:
                continue

            # --- Resize (CPU Efficient) ---
            resized_sat = cv2.resize(sat_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # --- Sliding Windows ---
            step = int(script_cfg['net_input_size'] * (1 - script_cfg['step_overlap']))
            windows = []

            y_range = list(range(0, new_h - script_cfg['net_input_size'] + 1, step))
            if y_range[-1] + script_cfg['net_input_size'] < new_h:
                y_range.append(new_h - script_cfg['net_input_size'])

            x_range = list(range(0, new_w - script_cfg['net_input_size'] + 1, step))
            if x_range[-1] + script_cfg['net_input_size'] < new_w:
                x_range.append(new_w - script_cfg['net_input_size'])

            for y in y_range:
                for x in x_range:
                    windows.append((x, y))

            if not windows: continue

            # --- DataLoader (Parallel Preprocessing) ---
            # 直接传入 Resize 后的 Numpy 数组
            dataset = SliceDataset(resized_sat, windows, transform=alb_transform)
            loader = DataLoader(dataset, batch_size=script_cfg['batch_size'],
                                shuffle=False, num_workers=script_cfg['num_workers'],
                                pin_memory=True)

            feats_all = []
            centers_list = []

            ref_x, ref_y, ref_res = meta['REF_initialX'], meta['REF_initialY'], meta['resolution']

            with torch.no_grad():
                for batch_imgs, batch_coords in loader:
                    batch_imgs = batch_imgs.to(script_cfg['device'])

                    # Extract
                    feats = model(batch_imgs)
                    feats = F.normalize(feats, p=2, dim=1)
                    feats_all.append(feats.cpu().numpy())

                    # Coords
                    coords = batch_coords.numpy()  # [B, 2] (x, y)
                    for (x, y) in coords:
                        cx_sc = x + script_cfg['net_input_size'] / 2.0
                        cy_sc = y + script_cfg['net_input_size'] / 2.0
                        cx_orig = cx_sc / scale
                        cy_orig = cy_sc / scale

                        utm_x = ref_x + (cx_orig * ref_res)
                        utm_y = ref_y - (cy_orig * ref_res)
                        centers_list.append([utm_x, utm_y])

            # --- Save ---
            if feats_all:
                grp = h5f.create_group(f"height_{height}")
                grp.create_dataset("features", data=np.concatenate(feats_all, axis=0))
                grp.create_dataset("centers_utm", data=np.array(centers_list))
                grp.attrs['fov_size_meters'] = float(fov_width_m)
                grp.attrs['scale_factor'] = float(scale)

    print(f"  [Done] Saved: {h5_path}")


def run():
    print("Loading Model...")
    model = get_camp_model('convnext_base', SCRIPT_CONFIG['ckpt_path'], SCRIPT_CONFIG['device'])
    model.eval()

    target_regions = [r for r in UAV_VISLOC_CONFIG['regions'] if int(r) not in UAV_VISLOC_CONFIG['skip_regions']]
    print(f"Target Regions: {target_regions}")

    for region_id in target_regions:
        try:
            process_region(region_id, model, SCRIPT_CONFIG, UAV_VISLOC_CONFIG)
        except Exception as e:
            print(f"[CRITICAL FAIL] Region {region_id}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run()