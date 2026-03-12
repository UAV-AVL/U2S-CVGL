# config.py
import numpy as np

# ================= 通用参数 =================
COMMON_CONFIG = {
    'min_vehicles_threshold': 5,
    'confidence_threshold': 0.5,
    'average_car_width': 1.8,
    'average_car_height': 1.5,

    # 开关控制
    'visualize_detections': False,  # 是否保存检测框图
    'save_log': True,  # 是否保存详细txt日志
    'save_pdf': False,  # 是否保存分析图表
    'test_ratio': 1,  # None 表示跑完所有图片

    # 结果保存根目录
    'save_root': 'Est_v4_calib_1.2_min_car_8_h_1.5_w_1.8_conf_0.5_all_angle_dsm_new'
}

# ================= 数据集路径配置 =================

# 1. UAV VisLoc 数据集配置
UAV_VISLOC_CONFIG = {
    'type': 'uav_visloc',
    'root_dir': '/work/documents/UAV_VisLoc_dataset',  # 数据集根目录
    'detection_root': '/work/mmrotate-1.x/tools/trained_models/VSAI/drone',  # 检测结果根目录
    'regions': [f"{i:02d}" for i in range(1, 12)],  # 01, 02, ..., 11
    # 特定区域跳过列表
    'skip_regions': [7, 10],#[9,10,11], #
    # 'skip_regions': [1,2,3,4,5,6,7,8], #
    # 新增文件名定义
    'pose_file_name': 'reference.txt',
    'camera_file_name': 'camera.xml',
    'dsm_suffix': '_dsm.tif'  # 结合region_id拼接: 04_dsm.tif
}

# 2. DenseUAV 数据集配置
DENSE_UAV_CONFIG = {
    'type': 'dense_uav',
    'image_dir': '/work/documents/DenseUAV/all_JPGs/',
    'detection_file': '/work/mmrotate-1.x/tools/trained_models/DenseUAV/Task1_Results/rotated_rtmdet_l-3x-vsai_rr/Task1_small-vehicle.txt',

    # DenseUAV 相机参数 (固定值)
    'camera_params': {
        'cam_size': np.sqrt((8.8*4/3)**2 + 8.8**2),
        'focal_len': 8.8,
        'width': 1440,
        'height': 1080,
        'pitch': -90.0,
        'roll': 0.0
    }
}

# 3. AnyVisloc数据集配置
ANY_VISLOC_CONFIG = {
    'type': 'any_visloc',
    'image_dir': '/work/D/Spatial_Resolution/VSAI_data/original_data/JPG_Images',
    'detection_file': '/work/D/Spatial_Resolution/Dota_estimate/dota_detect_result/oriented-rcnn-le90_r50_fpn_1x_vsai_rr/Task1_small-vehicle.txt',
    # 如果需要从 utils 读取，这里就不写死参数
}