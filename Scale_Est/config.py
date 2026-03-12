# config.py

# Specify the object detector used for extracting semantic anchors (small vehicles)
MODEL_NAME = 'rtmdet'

# ================= Common Configuration =================
COMMON_CONFIG = {
    # Reliability Filtering: Minimum number of valid semantic anchors (N_min) required to proceed
    'min_vehicles_threshold': 5,
    # Detection confidence threshold (tau_conf) to filter out false positives
    'confidence_threshold': 0.5,

    # Metric Priors for Semantic Anchors (Small Vehicles) in meters
    'average_car_width': 1.9,
    'average_car_height': 1.6,
    'average_car_length': 4.4,

    # I/O and Visualization Control
    'visualize_detections': False,  # Export images with bounding boxes
    'save_log': True,  # Export detailed txt logs
    'save_pdf': False,  # Generate analytical charts
    'test_ratio': 1,  # Set to < 1 to test on a subset, 1 for all images

    # Output directory for scale estimation results
    'save_root': f'results_{MODEL_NAME}_evaluation'
}

# ================= Dataset Adapters Configuration =================

# 1. UAV-VisLoc+ Dataset Configuration
UAV_VISLOC_CONFIG = {
    'type': 'uav_visloc',
    'model': MODEL_NAME,
    'root_dir': '/data/UAVVisLoc/UAVVisLoc_plus',  # Dataset Root
    'detection_root': '/data/UAVVisLoc/Detection',  # Detection result
    'regions': [f"{i:02d}" for i in range(1, 12)],  # Regions 01 to 11
    'skip_regions': [7, 9, 10, 11],  # Regions to skip during evaluation

    # Metadata files for scale-adaptive evaluation
    'pose_file_name': 'reference.txt',
    'camera_file_name': 'camera.xml',
    'dsm_suffix': '_dsm.tif'  # e.g., 04_dsm.tif
}

# 2. DenseUAV+ Dataset Configuration
DENSE_UAV_CONFIG = {
    'type': 'dense_uav',
    'image_dir': '/work2/documents/DenseUAV/ALL_test_imgs/', # Dataset Root
    'detection_file': './data/DenseUAV/small-vehicle.txt', # Detection result

    # Fixed camera intrinsic parameters for DenseUAV dataset
    'camera_params': {
        'cam_size': 14.667,
        'focal_len': 8.8,
        'width': 1440,
        'height': 1080,
        'pitch': -90.0,
        'roll': 0.0
    }
}
