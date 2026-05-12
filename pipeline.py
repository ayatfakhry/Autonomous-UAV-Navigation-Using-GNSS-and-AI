"""
=============================================================================
sensors/models.py  –  Physics-Based Sensor Simulation Models
=============================================================================
Simulates realistic sensor outputs with noise, bias, drift, and failure modes:
  - GNSS/RTK:     Position, velocity, signal degradation, multipath
  - IMU:          Accelerometer, gyroscope, vibration, thermal drift
  - Barometer:    Altitude, temperature compensation
  - Magnetometer: Hard/soft iron distortion
  - LiDAR:        360° point cloud with beam dropout
  - Camera:       Synthetic image features for CV pipeline
=============================================================================
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from sensors.fusion import (
    GNSSMeasurement, IMUMeasurement, BaroMeasurement, MagMeasurement
)
from utils.math_utils import (
    euler_to_rotation_matrix, body_to_world, normalize_vector
)
from utils.logger import get_logger

log = get_logger("SENSORS")


# ──────────────────────────────────────────────────────────────────────────────
# GNSS / RTK Sensor
# ──────────────────────────────────────────────────────────────────────────────

class GNSSSensor:
    """
    Simulates GNSS receiver with:
      - Gaussian position / velocity noise
      - RTK mode (sub-cm accuracy)
      - Signal dropout (buildings, trees)
      - Multipath error
      - HDOP/PDOP degradation model
    """

    def __init__(self, config: dict):
        self._noise_m   = config.get("position_noise_m", 0.5)
        self._vel_noise = config.get("velocity_noise_ms", 0.05)
        self._rtk       = config.get("rtk_enabled", True)
        self._rtk_noise = config.get("rtk_noise_m", 0.02)
        self._dropout_p = config.get("dropout_probability", 0.01)
        self._dropout_t = config.get("dropout_duration_s", 2.0)
        self._rate_hz   = config.get("update_rate_hz", 10)

        self._in_dropout  = False
        self._dropout_rem = 0.0
        self._t           = 0.0
        self._last_meas: Optional[GNSSMeasurement] = None

    def update(self, true_pos: np.ndarray, true_vel: np.ndarray,
               dt: float, environment_type: str = "open") -> Optional[GNSSMeasurement]:
        self._t += dt

        # Trigger / continue dropout
        if self._in_dropout:
            self._dropout_rem -= dt
            if self._dropout_rem <= 0:
                self._in_dropout = False
                log.info("GNSS signal restored")
            return None  # No measurement during dropout

        if random.random() < self._dropout_p * dt:
            self._in_dropout = True
            self._dropout_rem = self._dropout_t + random.uniform(-0.5, 0.5)
            log.warning(f"GNSS dropout started ({self._dropout_rem:.1f}s)")
            return None

        # Environment multipath
        multipath_scale = {"urban": 2.0, "forest": 1.5, "mountain": 1.2, "open_field": 0.5}
        mp = multipath_scale.get(environment_type, 1.0)

        # Noise model
        noise_m = self._rtk_noise if self._rtk else self._noise_m
        noise_m *= mp

        pos_noise = np.random.normal(0, noise_m, 3)
        pos_noise[2] *= 1.5  # Vertical accuracy ≈ 1.5× horizontal

        vel_noise = np.random.normal(0, self._vel_noise, 3)

        # Correlated multipath error (low-frequency)
        mp_error = noise_m * 0.3 * np.sin(self._t * 0.1 * np.array([1.0, 1.3, 0.7]))

        fix_type = 3 if self._rtk else 2
        accuracy = noise_m

        meas = GNSSMeasurement(
            position=true_pos + pos_noise + mp_error,
            velocity=true_vel + vel_noise,
            accuracy=accuracy,
            fix_type=fix_type,
        )
        self._last_meas = meas
        return meas


# ──────────────────────────────────────────────────────────────────────────────
# IMU Sensor
# ──────────────────────────────────────────────────────────────────────────────

class IMUSensor:
    """
    6-DOF IMU model:
      - White noise on accel & gyro
      - Bias instability (random walk)
      - Temperature drift
      - Vibration noise (motor harmonics)
    """

    def __init__(self, config: dict):
        self._a_noise = config.get("accel_noise_ms2", 0.01)
        self._g_noise = config.get("gyro_noise_rads", 0.001)
        self._a_bias_max = config.get("accel_bias_ms2", 0.005)
        self._g_bias_max = config.get("gyro_bias_rads", 0.0005)
        self._vib_noise  = config.get("vibration_noise", 0.002)
        self._rate_hz    = config.get("update_rate_hz", 200)

        # Biases (random walk)
        self._accel_bias = np.random.uniform(-self._a_bias_max, self._a_bias_max, 3)
        self._gyro_bias  = np.random.uniform(-self._g_bias_max, self._g_bias_max, 3)
        self._t = 0.0

    def update(self, true_accel: np.ndarray, true_gyro: np.ndarray,
               dt: float, motor_rpm: float = 5000.0) -> IMUMeasurement:
        self._t += dt

        # Bias random walk (Gauss-Markov process)
        tau_a, tau_g = 300.0, 600.0   # Correlation times [s]
        self._accel_bias += (-self._accel_bias/tau_a + np.random.normal(0, self._a_bias_max*0.001, 3)) * dt
        self._gyro_bias  += (-self._gyro_bias /tau_g + np.random.normal(0, self._g_bias_max*0.001, 3)) * dt

        # Motor vibration (fundamental + harmonics)
        f1 = motor_rpm / 60.0
        vib = self._vib_noise * np.array([
            math.sin(2*math.pi*f1*self._t) + 0.3*math.sin(4*math.pi*f1*self._t),
            math.cos(2*math.pi*f1*self._t) + 0.2*math.cos(6*math.pi*f1*self._t),
            0.5*math.sin(2*math.pi*f1*self._t + 0.3),
        ])

        accel_meas = (true_accel + self._accel_bias
                      + np.random.normal(0, self._a_noise, 3) + vib)
        gyro_meas  = (true_gyro  + self._gyro_bias
                      + np.random.normal(0, self._g_noise, 3))

        return IMUMeasurement(accel=accel_meas, gyro=gyro_meas, dt=dt)


# ──────────────────────────────────────────────────────────────────────────────
# Barometer
# ──────────────────────────────────────────────────────────────────────────────

class BarometerSensor:
    def __init__(self, config: dict):
        self._noise_m  = config.get("altitude_noise_m", 0.1)
        self._temp_drift = config.get("temperature_drift", 0.01)
        self._t = 0.0

    def update(self, true_alt: float, dt: float) -> BaroMeasurement:
        self._t += dt
        drift = self._temp_drift * math.sin(self._t / 300.0)  # 5-min thermal cycle
        noise = np.random.normal(0, self._noise_m)
        return BaroMeasurement(altitude=true_alt + noise + drift, noise_m=self._noise_m)


# ──────────────────────────────────────────────────────────────────────────────
# Magnetometer
# ──────────────────────────────────────────────────────────────────────────────

class MagnetometerSensor:
    """Magnetometer with hard-iron bias and soft-iron distortion."""

    EARTH_FIELD = np.array([0.2, 0.0, -0.45])  # Gauss (mid-latitude)

    def __init__(self, config: dict):
        self._noise   = config.get("noise_gauss", 0.005)
        self._hi_bias = np.array(config.get("hard_iron_bias", [0.1, -0.05, 0.02]))
        # Soft-iron matrix (near-identity)
        self._si_matrix = np.eye(3) + np.random.uniform(-0.02, 0.02, (3, 3))

    def update(self, roll: float, pitch: float, yaw: float) -> MagMeasurement:
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        B_world = self.EARTH_FIELD
        B_body = R.T @ B_world  # Rotate to body frame

        # Apply distortions
        B_distorted = self._si_matrix @ B_body + self._hi_bias
        B_noisy = B_distorted + np.random.normal(0, self._noise, 3)
        return MagMeasurement(field=B_noisy)


# ──────────────────────────────────────────────────────────────────────────────
# LiDAR
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LiDARScan:
    ranges:  np.ndarray   # [n_beams] distances [m]
    angles:  np.ndarray   # [n_beams] azimuth angles [rad]
    valid:   np.ndarray   # [n_beams] bool mask
    min_range: float
    max_range: float


class LiDARSensor:
    """
    2D + elevation LiDAR:
      - 360° azimuth scan
      - Obstacle range measurement using ray-casting
      - Gaussian range noise
      - Beam dropout simulation
    """

    def __init__(self, config: dict):
        self._range_max  = config.get("range_m", 100.0)
        self._range_min  = 0.15
        self._range_noise= config.get("range_noise_m", 0.05)
        self._n_beams    = config.get("num_beams", 360)
        self._fov        = math.radians(config.get("fov_deg", 360))
        self._rate_hz    = config.get("update_rate_hz", 20)
        self._angles     = np.linspace(-self._fov/2, self._fov/2, self._n_beams, endpoint=False)

    def update(self, pos: np.ndarray, obstacles: List[dict], yaw: float) -> LiDARScan:
        ranges = np.full(self._n_beams, self._range_max)
        valid  = np.ones(self._n_beams, dtype=bool)

        for obs in obstacles:
            obs_pos = np.array(obs["position"])
            obs_r   = obs.get("radius", 1.0)
            rel     = obs_pos[:2] - pos[:2]
            dist    = np.linalg.norm(rel)
            if dist > self._range_max + obs_r:
                continue
            angle_to_obs = math.atan2(rel[1], rel[0]) - yaw
            angle_to_obs = (angle_to_obs + math.pi) % (2*math.pi) - math.pi

            for i, ang in enumerate(self._angles):
                angle_diff = abs((ang - angle_to_obs + math.pi) % (2*math.pi) - math.pi)
                half_angle = math.asin(min(1.0, obs_r / max(dist, obs_r))) if dist > obs_r else math.pi/2
                if angle_diff < half_angle:
                    measured = max(self._range_min, dist - obs_r)
                    if measured < ranges[i]:
                        ranges[i] = measured

        # Add noise
        noise = np.random.normal(0, self._range_noise, self._n_beams)
        ranges = np.clip(ranges + noise, self._range_min, self._range_max)

        # Random beam dropout (5% probability)
        dropout_mask = np.random.random(self._n_beams) < 0.05
        valid[dropout_mask] = False
        ranges[dropout_mask] = self._range_max

        return LiDARScan(
            ranges=ranges,
            angles=self._angles + yaw,
            valid=valid,
            min_range=self._range_min,
            max_range=self._range_max,
        )

    def get_obstacle_map(self, scan: LiDARScan, pos: np.ndarray) -> np.ndarray:
        """Convert LiDAR scan to 2D obstacle point cloud [N, 2]."""
        mask = scan.valid & (scan.ranges < self._range_max * 0.95)
        xs = pos[0] + scan.ranges[mask] * np.cos(scan.angles[mask])
        ys = pos[1] + scan.ranges[mask] * np.sin(scan.angles[mask])
        return np.column_stack([xs, ys])


# ──────────────────────────────────────────────────────────────────────────────
# Camera (Synthetic Feature Extractor)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CameraFrame:
    detections: List[Dict]   # [{label, bbox, confidence, distance}]
    landing_zones: List[Dict]  # [{center, area, flatness}]
    optical_flow: np.ndarray   # [2] velocity estimate from optical flow


class CameraSensor:
    """
    Simulates RGB camera with:
      - Object detection output (bounding boxes)
      - Landing zone detection
      - Optical flow velocity estimate
    Used as interface layer for YOLOv8 / CV pipeline.
    """

    OBJECT_CLASSES = ["person", "vehicle", "tree", "building", "unknown_obstacle"]

    def __init__(self, config: dict):
        self._fps       = config.get("update_rate_hz", 30)
        self._res       = config.get("resolution", [640, 480])
        self._fov       = math.radians(config.get("fov_deg", 90))
        self._noise_std = config.get("noise_std", 5.0)

    def update(self, pos: np.ndarray, vel: np.ndarray, att: np.ndarray,
               obstacles: List[dict]) -> CameraFrame:
        detections = []
        roll, pitch, yaw = att

        for obs in obstacles:
            obs_pos = np.array(obs["position"])
            rel_world = obs_pos - pos
            dist = np.linalg.norm(rel_world)
            if dist > 50.0:
                continue

            # Check if in camera FOV
            angle = math.atan2(rel_world[1], rel_world[0]) - yaw
            if abs(angle) > self._fov / 2:
                continue

            # Synthetic bounding box (pixel space)
            apparent_size = max(10, int(500 / max(dist, 1)))
            cx = int(self._res[0]/2 + self._res[0] * math.tan(angle) / math.tan(self._fov/2))
            cy = int(self._res[1]/2 - rel_world[2] * 100 / max(dist, 1))
            cx += int(np.random.normal(0, self._noise_std))
            cy += int(np.random.normal(0, self._noise_std))

            detections.append({
                "label": obs.get("type", "unknown_obstacle"),
                "bbox": [cx - apparent_size//2, cy - apparent_size//2,
                         apparent_size, apparent_size],
                "confidence": max(0.3, 0.95 - 0.01*dist + np.random.normal(0, 0.05)),
                "distance_m": dist + np.random.normal(0, 0.5),
            })

        # Landing zone (flat terrain below)
        lz = []
        if pos[2] < 30.0 and abs(att[0]) < 0.1 and abs(att[1]) < 0.1:
            lz.append({
                "center": (self._res[0]//2, self._res[1]//2),
                "area_px2": self._res[0] * self._res[1] * 0.1,
                "flatness": max(0, 1.0 - 0.1*pos[2]),
            })

        # Optical flow velocity estimate
        flow = vel[:2] / max(pos[2], 0.5) * 100 + np.random.normal(0, 2.0, 2)

        return CameraFrame(
            detections=detections,
            landing_zones=lz,
            optical_flow=flow,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Sensor Suite (Aggregator)
# ──────────────────────────────────────────────────────────────────────────────

class SensorSuite:
    """
    Aggregates all sensors. One call per simulation step.
    Returns all measurements; some may be None (rate mismatch / dropout).
    """

    def __init__(self, config: dict):
        sensor_cfg = config.get("sensors", {})
        self.gnss  = GNSSSensor(sensor_cfg.get("gnss", {}))
        self.imu   = IMUSensor(sensor_cfg.get("imu", {}))
        self.baro  = BarometerSensor(sensor_cfg.get("barometer", {}))
        self.mag   = MagnetometerSensor(sensor_cfg.get("magnetometer", {}))
        self.lidar = LiDARSensor(sensor_cfg.get("lidar", {}))
        self.camera= CameraSensor(sensor_cfg.get("camera", {}))

        self._t       = 0.0
        self._env_type = config.get("environment", {}).get("type", "urban")

    def update(
        self,
        pos: np.ndarray, vel: np.ndarray, att: np.ndarray,
        accel_body: np.ndarray, omega_body: np.ndarray,
        obstacles: List[dict], dt: float,
    ) -> Dict:
        self._t += dt
        roll, pitch, yaw = att

        imu_meas   = self.imu.update(accel_body, omega_body, dt)
        gnss_meas  = self.gnss.update(pos, vel, dt, self._env_type)
        baro_meas  = self.baro.update(pos[2], dt)
        mag_meas   = self.mag.update(roll, pitch, yaw)
        lidar_scan = self.lidar.update(pos, obstacles, yaw)
        camera_frame = self.camera.update(pos, vel, att, obstacles)

        return {
            "imu":    imu_meas,
            "gnss":   gnss_meas,
            "baro":   baro_meas,
            "mag":    mag_meas,
            "lidar":  lidar_scan,
            "camera": camera_frame,
        }
