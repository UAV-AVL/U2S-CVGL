import os
import time
import math
import yaml
import h5py
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader

from multi_model.camp.get_camp import get_camp_model


# ---------------------------------------------------------
# 1. 辅助类与函数 (Albumentations 优化版)
# ---------------------------------------------------------

class SatSliceDataset(Dataset):
    """
    用于PyTorch DataLoader的数据集
    优化：直接处理 Numpy Array，使用 Albumentations
    """

    def __init__(self, satellite_img, crop_windows, transform=None):
        self.img = satellite_img  # 已经是缩放后的整图 (Scaled Satellite, BGR/RGB based on upstream)
        self.windows = crop_windows  # List of (x, y, w, h)
        self.transform = transform

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x, y, w, h = self.windows[idx]
        # Crop: image is (H, W, C)
        # 注意：这里切片出来的是 OpenCV 的 BGR 格式（如果输入是 BGR）
        patch = self.img[y:y + h, x:x + w]

        # 转换为 RGB (Albumentations 默认期望 RGB)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Albumentations 调用方式
            augmented = self.transform(image=patch)
            patch_tensor = augmented['image']
        else:
            # Fallback (手动转 Tensor)
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0

        return patch_tensor, np.array([x, y, w, h])


def calculate_uav_resolution(height, camera_params):
    """根据飞行高度和相机参数计算地面分辨率 (m/pixel)"""
    cam_size = camera_params['cam_size']
    focal_len = camera_params['focal_len']
    width = camera_params['width']
    height_px = camera_params['height']
    pitch = camera_params['pitch']

    ground_res = (height * cam_size) / (focal_len * math.sqrt(width ** 2 + height_px ** 2))
    scale_factor = abs(1 / np.sin(np.deg2rad(pitch)))

    final_res = ground_res * scale_factor
    return final_res


def get_sliding_windows(img_h, img_w, crop_size, step_size):
    """生成滑动窗口坐标"""
    windows = []
    # 保证覆盖边缘
    x_range = list(range(0, img_w - crop_size + 1, step_size))
    if x_range[-1] + crop_size < img_w:
        x_range.append(img_w - crop_size)

    y_range = list(range(0, img_h - crop_size + 1, step_size))
    if y_range[-1] + crop_size < img_h:
        y_range.append(img_h - crop_size)

    for y in y_range:
        for x in x_range:
            windows.append((x, y, crop_size, crop_size))
    return windows


def pixel_to_utm(pixel_x, pixel_y, ref_x, ref_y, ref_res, scale_ratio):
    """坐标反算"""
    orig_px_x = pixel_x * scale_ratio
    orig_px_y = pixel_y * scale_ratio
    cur_utm_x = ref_x + (orig_px_x * ref_res)
    cur_utm_y = ref_y - (orig_px_y * ref_res)
    return cur_utm_x, cur_utm_y


# ---------------------------------------------------------
# 2. 主处理流程
# ---------------------------------------------------------

def precompute_features(opt, config, model, device):
    ref_path = config['REF_path']
    initialX = float(config['REF_initialX'])
    initialY = float(config['REF_initialY'])
    ref_res = float(config['resolution'])
    utm_system = config['UTM_SYSTEM']

    print(f"Loading Reference Map: {ref_path}")
    # OpenCV 读取 (BGR)
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        raise FileNotFoundError(f"Cannot read file: {ref_path}")
    ref_h, ref_w = ref_img.shape[:2]

    # 相机参数
    camera_params = {
        'cam_size': np.sqrt((8.8*4/3)**2 + 8.8**2), 'focal_len': 8.8,
        'width': 1440, 'height': 1080,  # 原始分辨率
        'pitch': -90.0, 'roll': 0.0
    }

    save_file_path = os.path.join(opt.save_subfolder_path, f"{opt.run_tag}_features.h5")
    print(f"Features will be saved to: {save_file_path}")

    # 定义预处理 (Albumentations)
    # 此时卫星图已经被 resize 到目标分辨率了，所以这里不需要 Resize，只需要 Normalize
    net_w, net_h = opt.UAV_size  # 384, 384

    preprocess = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    with h5py.File(save_file_path, 'w') as h5f:
        h5f.attrs['UTM_SYSTEM'] = utm_system
        h5f.attrs['REF_PATH'] = ref_path
        h5f.attrs['REF_RESOLUTION'] = ref_res

        # 计算高度列表 (与之前逻辑一致)
        base_heights = [80, 90, 100]
        relative_error = opt.relative_error
        min_h_bound = min([h * (1 - relative_error) for h in base_heights])
        max_h_bound = max([h * (1 + relative_error) for h in base_heights])
        start_h = int(np.floor(min_h_bound))
        end_h = int(np.ceil(max_h_bound))
        step_interval = 5
        if start_h % step_interval != 0:
            start_h = (start_h // step_interval) * step_interval
        heights_list = list(range(start_h, end_h + 1, step_interval))

        print(f"Processing Heights: {heights_list}")

        for height in heights_list:
            print(f"\n--- Processing Height: {height}m ---")

            uav_res = calculate_uav_resolution(height, camera_params)

            # =========================================================================
            # 【关键修改】适配中心裁剪 (Center Crop)
            # 无人机图像是 1440x1080。在输入网络前，通常的操作是：
            # 1. CenterCrop(min_dim) -> 变成 1080x1080
            # 2. Resize(384, 384)
            # 因此，网络看到的 384x384 图像，实际上对应的是地面上 1080px 宽度的范围，而不是 1440px。
            # 所以在生成卫星底图时，我们也应该以 1080px (min_dim) 对应的物理宽度为准。
            # =========================================================================

            effective_px = min(camera_params['width'], camera_params['height'])  # 1080
            uav_fov_w_m = effective_px * uav_res  # 这是中心正方形区域的物理宽度

            # 计算缩放比例：我们将卫星图缩放，使得 384 个像素正好代表 uav_fov_w_m 米
            target_res = uav_fov_w_m / net_w  # net_w = 384
            scale_factor = ref_res / target_res
            finescale = 1 / scale_factor

            scaled_w = int(ref_w * scale_factor)
            scaled_h = int(ref_h * scale_factor)

            print(f"UAV Res: {uav_res:.4f} m/px | Target Sat Res: {target_res:.4f} m/px")
            print(f"Effective FOV Width (Square): {uav_fov_w_m:.2f} meters")

            # 边界检查
            if scaled_w < net_w or scaled_h < net_h:
                print(f"Skipping H={height}: Scaled image too small.")
                continue

            # Resize 整图 (OpenCV CPU resize is fast)
            interp = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_LINEAR
            scaled_img = cv2.resize(ref_img, (scaled_w, scaled_h), interpolation=interp)

            # 生成滑动窗口
            crop_size = net_w
            step_px = int(crop_size * (1 - opt.step_cover / 100.0))
            if step_px < 1: step_px = 1

            windows = get_sliding_windows(scaled_h, scaled_w, crop_size, step_px)
            print(f"Generated {len(windows)} patches.")

            # DataLoader (使用 Albumentations Dataset)
            dataset = SatSliceDataset(scaled_img, windows, transform=preprocess)
            dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                                    num_workers=opt.num_workers, pin_memory=True)  # pin_memory 加速 GPU 传输

            # HDF5 Group
            grp = h5f.create_group(f"height_{height}")
            grp.attrs['fov_size_meters'] = float(uav_fov_w_m)
            grp.attrs['scale_factor'] = float(scale_factor)

            # 获取特征维度 (动态)
            dummy_input = torch.zeros(1, 3, net_h, net_w).to(device)
            with torch.no_grad():
                dummy_out = model(dummy_input)
                if isinstance(dummy_out, tuple): dummy_out = dummy_out[0]
                feature_dim = dummy_out.shape[1]

            dset_feats = grp.create_dataset("features", (len(windows), feature_dim), dtype='f4')
            dset_centers = grp.create_dataset("centers_utm", (len(windows), 2), dtype='f8')

            current_idx = 0

            # 推理
            with torch.no_grad():
                for batch_imgs, batch_infos in tqdm(dataloader, desc=f"Extracting H={height}"):
                    batch_imgs = batch_imgs.to(device)

                    # Extract
                    features = model(batch_imgs)
                    if isinstance(features, tuple): features = features[0]
                    features = F.normalize(features, p=2, dim=1)
                    features_np = features.cpu().numpy()

                    batch_size = features_np.shape[0]
                    batch_centers = []

                    infos = batch_infos.numpy()
                    for i in range(batch_size):
                        bx, by, bw, bh = infos[i]
                        # 中心点
                        cx_px = bx + bw / 2.0
                        cy_px = by + bh / 2.0
                        # 转换中心点
                        c_utm_x, c_utm_y = pixel_to_utm(cx_px, cy_px, initialX, initialY, ref_res, finescale)
                        batch_centers.append([c_utm_x, c_utm_y])

                    # Save
                    dset_feats[current_idx: current_idx + batch_size] = features_np
                    dset_centers[current_idx: current_idx + batch_size] = batch_centers
                    current_idx += batch_size

    print(f"Processing complete. Saved to {save_file_path}")

if __name__ == "__main__":
    class Options: pass


    opt = Options()
    opt.save_base_path = "RESULTS"
    opt.subfolder_name = "DU_h5__refine_AVL_IR_range0.8"  # 建议改名以区分
    opt.run_tag = "DU_2018_AVL_IR_features"
    opt.UAV_size = [384, 384]
    opt.step_cover = 50
    opt.batch_size = 32
    opt.num_workers = 4
    opt.relative_error = 0.8
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.ck_path = r'multi_model/weights/AVL-IR-Fusion_weights_e3_0.7557.pth'
    yaml_path = r'tif_yamls/DenseUAV_2018_1_1_L20.yaml'

    opt.save_subfolder_path = os.path.join(opt.save_base_path, opt.subfolder_name)
    os.makedirs(opt.save_subfolder_path, exist_ok=True)

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    print("Loading Model...")

    model = get_camp_model('convnext_base', opt.ck_path, opt.device)
    model.eval()

    precompute_features(opt, config_data, model, opt.device)