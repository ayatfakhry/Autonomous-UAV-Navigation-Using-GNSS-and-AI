"""
=============================================================================
utils/math_utils.py  –  Aerospace Mathematics Library
=============================================================================
SO(3) rotations, coordinate frames, geodetic conversions, quaternion algebra,
Euler angles, and aerospace-standard math helpers used across all modules.
=============================================================================
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Rotation Matrices (SO3)
# ──────────────────────────────────────────────────────────────────────────────

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    ZYX (aerospace) Euler angles → 3×3 rotation matrix R_body_to_world.
    Args:
        roll  (φ): rotation about X [rad]
        pitch (θ): rotation about Y [rad]
        yaw   (ψ): rotation about Z [rad]
    Returns:
        R: 3×3 ndarray
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ])
    return R


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    3×3 rotation matrix → (roll, pitch, yaw) in radians.
    Handles gimbal-lock singularity at pitch = ±90°.
    """
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = math.atan2(R[1, 0], R[0, 0])
    else:  # Gimbal lock
        roll  = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw   = 0.0
    return roll, pitch, yaw


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion Algebra
# ──────────────────────────────────────────────────────────────────────────────

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX Euler → unit quaternion [w, x, y, z]."""
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([w, x, y, z])


def quaternion_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """Unit quaternion [w, x, y, z] → (roll, pitch, yaw) radians."""
    w, x, y, z = q
    # Roll
    sinr_cosp = 2.0*(w*x + y*z)
    cosr_cosp = 1.0 - 2.0*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # Pitch
    sinp = 2.0*(w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # Yaw
    siny_cosp = 2.0*(w*z + x*y)
    cosy_cosp = 1.0 - 2.0*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-10 else np.array([1.0, 0.0, 0.0, 0.0])


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate Frame Transforms
# ──────────────────────────────────────────────────────────────────────────────

def ned_to_enu(v_ned: np.ndarray) -> np.ndarray:
    """NED (North-East-Down) → ENU (East-North-Up) frame."""
    n, e, d = v_ned
    return np.array([e, n, -d])


def enu_to_ned(v_enu: np.ndarray) -> np.ndarray:
    """ENU (East-North-Up) → NED (North-East-Down) frame."""
    e, n, u = v_enu
    return np.array([n, e, -u])


def body_to_world(v_body: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotate a vector from body frame to world (ENU) frame."""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return R @ v_body


def world_to_body(v_world: np.ndarray, roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotate a vector from world (ENU) frame to body frame."""
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    return R.T @ v_world


# ──────────────────────────────────────────────────────────────────────────────
# Geodetic Utilities
# ──────────────────────────────────────────────────────────────────────────────

_EARTH_RADIUS_M = 6_371_000.0  # Mean Earth radius [m]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two WGS-84 points [meters].
    Args: lat/lon in decimal degrees.
    """
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δφ = math.radians(lat2 - lat1)
    Δλ = math.radians(lon2 - lon1)
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def lla_to_ned(lat: float, lon: float, alt: float,
               lat0: float, lon0: float, alt0: float) -> np.ndarray:
    """
    LLA (lat/lon/alt deg, deg, m) → NED offset [m] relative to reference.
    Uses flat-Earth approximation (valid < 50 km).
    """
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    north = dlat * _EARTH_RADIUS_M
    east  = dlon * _EARTH_RADIUS_M * math.cos(math.radians(lat0))
    down  = -(alt - alt0)
    return np.array([north, east, down])


# ──────────────────────────────────────────────────────────────────────────────
# Scalar Helpers
# ──────────────────────────────────────────────────────────────────────────────

def wrap_angle(angle: float) -> float:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def wrap_angle_360(angle_deg: float) -> float:
    """Wrap angle to [0, 360)."""
    return angle_deg % 360.0


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, val))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Return unit vector; returns zero vector if norm ≈ 0."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.zeros_like(v)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix for cross-product: skew(a)·b = a×b."""
    return np.array([
        [   0.0, -v[2],  v[1]],
        [ v[2],    0.0, -v[0]],
        [-v[1],   v[0],   0.0],
    ])


def rms(values: np.ndarray) -> float:
    """Root-mean-square of array."""
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average filter."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="full")[:len(data)]
