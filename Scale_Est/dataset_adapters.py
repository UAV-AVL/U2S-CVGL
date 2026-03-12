import os
import re
import numpy as np
import xml.etree.ElementTree as ET
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform


class DatasetAdapterFactory:
    @staticmethod
    def get_adapter(dataset_type, config):
        """
        Factory method to instantiate the appropriate dataset adapter.
        """
        if dataset_type == 'uav_visloc':
            return UAVVisLocAdapter(config)
        elif dataset_type == 'dense_uav':
            return DenseUAVAdapter(config)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


class UAVVisLocAdapter:
    """
    Adapter for the UAV-VisLoc+ dataset.
    Parses camera intrinsics, poses, and samples the Digital Surface Model (DSM)
    to compute the ground-truth relative flight altitude (H_uav) for evaluation.
    """

    def __init__(self, config):
        self.pose_path = config.get('pose_path')
        self.camera_path = config.get('camera_path')
        self.dsm_path = config.get('dsm_path')
        self.alt_offset = config.get('altitude_offset', 0.0)

        # Load camera intrinsics and absolute flight poses
        self.camera_intrinsics = self._load_camera_xml(self.camera_path)
        self.poses = self._load_reference_txt(self.pose_path)

        # Prepare DSM handle to compute nadir ground elevation
        self.dsm_src = None
        if os.path.exists(self.dsm_path):
            self.dsm_src = rasterio.open(self.dsm_path)
        else:
            print(f"Warning: DSM file not found at {self.dsm_path}")

    def __del__(self):
        if self.dsm_src:
            self.dsm_src.close()

    def _load_camera_xml(self, xml_path):
        """Parse camera.xml to extract intrinsic parameters and distortion coefficients."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = float(root.find('width').text)
            height = float(root.find('height').text)
            f_pix = float(root.find('f').text)
            cx = float(root.find('cx').text)
            cy = float(root.find('cy').text)

            k1, k2, k3 = [float(root.find(k).text) for k in ('k1', 'k2', 'k3')]
            p1, p2 = [float(root.find(p).text) for p in ('p1', 'p2')]

            dist_coeffs = [k1, k2, p1, p2, k3]
            abs_cx = width / 2.0 + cx
            abs_cy = height / 2.0 + cy

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
                'dist_coeffs': dist_coeffs,
                'camera_matrix': camera_matrix
            }
        except Exception as e:
            print(f"Error loading camera XML: {e}")
            return None

    def _load_reference_txt(self, txt_path):
        """Parse reference.txt to extract absolute flight pose (GNSS coordinates and IMU angles)."""
        poses = {}
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith('#'):
                    continue

                parts = line.strip().split()
                if len(parts) < 10: continue

                img_name = parts[0]
                try:
                    lon, lat = float(parts[-6]), float(parts[-5])
                    alt = float(parts[-4])
                    yaw = float(parts[-3])
                    # Adjust pitch to follow standard coordinate conventions
                    pitch = -(90 - abs(float(parts[-2])))
                    roll = float(parts[-1])

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
        Sample the median elevation from the DSM within a given radius around the nadir point.
        This isolates the local ground elevation to compute accurate relative altitude.
        """
        if self.dsm_src is None:
            return 0.0

        try:
            if self.dsm_src.crs != 'EPSG:4326':
                xs, ys = transform('EPSG:4326', self.dsm_src.crs, [lon], [lat])
                cx, cy = xs[0], ys[0]
            else:
                cx, cy = lon, lat

            res_x = self.dsm_src.res[0]
            # Convert radius to degrees if the CRS is geographic
            radius_val = 200 / 111320.0 if res_x < 0.1 else 200.0

            window = from_bounds(
                cx - radius_val, cy - radius_val,
                cx + radius_val, cy + radius_val,
                self.dsm_src.transform
            )

            data = self.dsm_src.read(1, window=window)
            valid_data = data[data > -1000]  # Mask out NoData values

            if valid_data.size == 0:
                return 0.0

            # Use 5th percentile to robustly approximate ground level ignoring buildings/trees
            return float(np.percentile(valid_data, 5))

        except Exception as e:
            return 0.0

    def get_data(self, img_path):
        img_name = os.path.basename(img_path)

        pose = self.poses.get(img_name)
        if not pose:
            return None

        # Calculate relative altitude: (Absolute Altitude + Offset) - Ground Elevation
        ground_h = self._get_ground_elevation(pose['lon'], pose['lat'], radius_meter=200)
        aircraft_alt_corrected = pose['alt'] + self.alt_offset
        rel_alt = float(aircraft_alt_corrected - ground_h)

        if not self.camera_intrinsics:
            return None

        data = self.camera_intrinsics.copy()
        data.update({
            'pitch': pose['pitch'],
            'roll': pose['roll'],
            'yaw': pose['yaw'],
            'rel_alt': rel_alt,  # Ground-truth relative altitude for scale evaluation
            'lat': pose['lat'],
            'lon': pose['lon']
        })

        return data


class DenseUAVAdapter:
    """
    Adapter for the DenseUAV+ dataset.
    Extracts the ground-truth relative altitude directly from the image filenames.
    """

    def __init__(self, config):
        self.params = config['camera_params']

    def get_data(self, img_path):
        img_name = os.path.basename(img_path)
        try:
            # Parse relative altitude from filename (e.g., 000001_H_80.jpg -> 80)
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
