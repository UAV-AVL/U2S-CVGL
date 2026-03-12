import os
import json
from config import COMMON_CONFIG, UAV_VISLOC_CONFIG, DENSE_UAV_CONFIG
from dataset_adapters import DatasetAdapterFactory
from scale_estimator import ScaleEstimator


class UniversalRunner:
    """
    Executes the semantic geometric scale estimation framework across datasets,
    evaluating the accuracy of absolute scale recovery against ground-truth altitude.
    """

    def __init__(self):
        self.base_config = COMMON_CONFIG
        self.table_history = []

    def _print_live_table(self, dataset_name):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"\n=== Scale Estimation Evaluation Report: {dataset_name} ===")
        line_len = 155
        print("-" * line_len)
        print(
            f"{'Subset/Region':<15} | {'Images':<8} | {'Used':<6} | {'Avg Anchors':<11} | "
            f"{'True Alt(m)':<20} | {'Mean Err(%)':<14} | {'Med Err(%)':<14} | {'Err Var':<10} | {'Range(%)':<20}")
        print("-" * line_len)
        for row in self.table_history:
            print(row)
        print("-" * line_len)

    def run_uav_visloc(self):
        """Evaluate Scale Recovery on the UAV-VisLoc+ Dataset"""
        cfg = UAV_VISLOC_CONFIG
        dataset_name = "UAV-VisLoc+"
        self.table_history = []

        for region_id in cfg['regions']:
            if int(region_id) in cfg['skip_regions']:
                continue

            alt_offset = 0.0

            current_config = self.base_config.copy()
            # Altitude is computed relative to DSM in the adapter, so local region altitude is 0.0
            current_config['region_altitude'] = 0.0

            img_dir = f"{cfg['root_dir']}/{region_id}/drone/"

            if cfg['model'] == 'rtmdet':
                det_file = f"{cfg['detection_root']}/{region_id}/Task1_Results/rotated_rtmdet_l-3x-vsai_rr/Task1_small-vehicle.txt"
            else:
                det_file = f"{cfg['detection_root']}/{region_id}/{cfg['model']}_{region_id}/Task1_small-vehicle.txt"

            exp_name = f"VisLoc_{region_id}_conf{current_config['confidence_threshold']}"
            self._setup_paths(current_config, exp_name)

            adapter_config = {
                'pose_path': f"{cfg['root_dir']}/{region_id}/{cfg['pose_file_name']}",
                'camera_path': f"{cfg['root_dir']}/{region_id}/{cfg['camera_file_name']}",
                'dsm_path': f"{cfg['root_dir']}/{region_id}/region{region_id}{cfg['dsm_suffix']}",
                'altitude_offset': alt_offset
            }

            if not os.path.exists(adapter_config['pose_path']):
                print(f"Skipping {region_id}: Reference file not found.")
                continue

            adapter = DatasetAdapterFactory.get_adapter('uav_visloc', adapter_config)

            self._run_single_task(f"Region {region_id}", current_config, adapter, det_file, img_dir)
            self._print_live_table(dataset_name)

    def run_dense_uav(self):
        """Evaluate Scale Recovery on the DenseUAV+ Dataset"""
        cfg = DENSE_UAV_CONFIG
        dataset_name = "DenseUAV+"
        self.table_history = []

        current_config = self.base_config.copy()
        exp_name = f"DenseUAV_conf{current_config['confidence_threshold']}"
        self._setup_paths(current_config, exp_name)

        adapter = DatasetAdapterFactory.get_adapter('dense_uav', cfg)

        self._run_single_task("All Images", current_config, adapter, cfg['detection_file'], cfg['image_dir'])
        self._print_live_table(dataset_name)

    def _run_single_task(self, task_name, config, adapter, det_file, img_dir):
        """Standardized execution and metric tracking for a single dataset subset."""
        estimator = ScaleEstimator(config, data_provider=adapter)
        detections = estimator.parse_detection_file(det_file)

        # Filter strictly based on nadir-equivalent thresholds as discussed in the paper
        total_imgs = estimator.count_valid_test_images(img_dir, pitch_thresh=-60.0)

        # Execute Robust Global Scale Recovery
        estimator.estimate_height(detections, img_dir)

        stats = estimator.overall_stats
        if stats:
            true_alt_str = f"{stats['mean_true_alt']:.1f} ({stats['min_true_alt']:.0f}~{stats['max_true_alt']:.0f})"

            row_str = (
                f"{task_name:<15} | "
                f"{total_imgs:<8} | "
                f"{len(estimator.results):<6} | "
                f"{stats['avg_vehicles_used']:<11.1f} | "
                f"{true_alt_str:<20} | "
                f"{stats['mean_rel_error']:>13.2f}% | "
                f"{stats['median_rel_error']:>13.2f}% | "
                f"{stats['error_variance']:<10.2f} | "
                f"{stats['min_rel_error']:>6.1f}~{stats['max_rel_error']:<6.1f}"
            )
            self.table_history.append(row_str)
        else:
            self.table_history.append(f"{task_name:<15} | NO DATA")

    def _setup_paths(self, config, exp_subname):
        """Generate output directories for evaluation logs and visualizations."""
        root = config['save_root']
        exp_dir = os.path.join(root, exp_subname)

        config['output_pdf'] = f"{exp_dir}/result.pdf" if config['save_pdf'] else None
        config['log_file'] = f"{exp_dir}/log.txt" if config['save_log'] else None
        config['visualization_dir'] = f"{exp_dir}/viz"
        config['exif_cache_file'] = f"{exp_dir}/cache.json"

        os.makedirs(exp_dir, exist_ok=True)
        if config['visualization_dir']:
            os.makedirs(config['visualization_dir'], exist_ok=True)


if __name__ == "__main__":
    runner = UniversalRunner()

    # Run evaluation pipelines
    runner.run_uav_visloc()
    # runner.run_dense_uav()
