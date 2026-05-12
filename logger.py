"""
=============================================================================
sensors/fusion.py  –  Extended & Unscented Kalman Filter Sensor Fusion
=============================================================================
15-state EKF/UKF fusing:
  - GNSS position & velocity
  - IMU accelerometer & gyroscope
  - Barometer altitude
  - Magnetometer heading

State vector x ∈ ℝ¹⁵:
  [pos_e, pos_n, pos_u,      (0:3)  ENU position [m]
   vel_e, vel_n, vel_u,      (3:6)  ENU velocity [m/s]
   roll, pitch, yaw,         (6:9)  Euler angles [rad]
   b_gx, b_gy, b_gz,        (9:12) Gyro bias [rad/s]
   b_ax, b_ay, b_az]        (12:15) Accel bias [m/s²]
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from utils.math_utils import (
    euler_to_rotation_matrix,
    rotation_matrix_to_euler,
    skew_symmetric,
    wrap_angle,
)
from utils.logger import get_logger

log = get_logger("FUSION")


# ──────────────────────────────────────────────────────────────────────────────
# Measurement Data Classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IMUMeasurement:
    accel: np.ndarray   # [ax, ay, az]  body frame [m/s²]
    gyro:  np.ndarray   # [gx, gy, gz]  body frame [rad/s]
    dt:    float        # integration period [s]

@dataclass
class GNSSMeasurement:
    position: np.ndarray    # ENU [m] (relative to home)
    velocity: Optional[np.ndarray] = None  # ENU [m/s]
    accuracy: float = 1.0   # 1σ position accuracy [m]
    fix_type: int = 3       # 0=no fix,1=2D,2=3D,3=RTK

@dataclass
class BaroMeasurement:
    altitude: float     # MSL altitude [m]
    noise_m:  float = 0.1

@dataclass
class MagMeasurement:
    field: np.ndarray   # Magnetic field vector [gauss] body frame

@dataclass
class FusedState:
    """Output of the sensor fusion pipeline."""
    position:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    attitude:  np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll,pitch,yaw
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_bias:np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance:np.ndarray = field(default_factory=lambda: np.eye(15) * 0.01)
    gnss_quality: float = 1.0   # 0..1
    timestamp: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Extended Kalman Filter (EKF)
# ──────────────────────────────────────────────────────────────────────────────

class ExtendedKalmanFilter:
    """
    15-state Strapdown INS EKF.

    Prediction: IMU mechanization + error-state linearization.
    Updates:
      - GNSS position (3-state)
      - GNSS velocity (3-state)
      - Barometer altitude (1-state)
      - Magnetometer heading (1-state)

    Implements:
      - Chi-squared innovation gating (outlier rejection)
      - Adaptive process noise scaling
      - Covariance symmetrization
    """

    STATE_DIM = 15
    G = np.array([0.0, 0.0, -9.81])  # Gravity in ENU [m/s²]

    def __init__(
        self,
        process_noise_scale: float = 1.0,
        innovation_gate: float = 9.21,   # χ²(3, 99%)
        adaptive: bool = True,
    ):
        self._scale = process_noise_scale
        self._gate  = innovation_gate
        self._adaptive = adaptive

        # State & covariance
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 1.0

        # Process noise covariance Q
        self.Q = self._build_process_noise()

        # Measurement noise matrices
        self.R_gnss_pos = np.eye(3) * 0.25   # 0.5m 1σ
        self.R_gnss_vel = np.eye(3) * 0.0025 # 0.05 m/s
        self.R_baro     = np.array([[0.01]])  # 0.1m altitude
        self.R_mag      = np.array([[0.001]]) # heading noise

        self._initialized = False
        self._n_updates = 0

    # ── Initialisation ────────────────────────────────────────────────────────

    def initialize(self, pos: np.ndarray, vel: np.ndarray, att: np.ndarray) -> None:
        self.x[:3]  = pos
        self.x[3:6] = vel
        self.x[6:9] = att
        self.P = np.diag([
            1.0, 1.0, 1.0,          # position uncertainty 1m
            0.1, 0.1, 0.1,          # velocity 0.1 m/s
            0.01, 0.01, 0.01,       # attitude 0.01 rad (~0.6°)
            1e-4, 1e-4, 1e-4,       # gyro bias
            1e-3, 1e-3, 1e-3,       # accel bias
        ])
        self._initialized = True
        log.info("EKF initialized | pos={} att={}".format(
            np.round(pos, 2), np.round(np.degrees(att), 1)
        ))

    # ── Predict Step (IMU mechanization) ─────────────────────────────────────

    def predict(self, imu: IMUMeasurement) -> None:
        if not self._initialized:
            return

        x = self.x.copy()
        dt = imu.dt

        pos = x[0:3];  vel = x[3:6]; att = x[6:9]
        b_g = x[9:12]; b_a = x[12:15]

        roll, pitch, yaw = att
        R = euler_to_rotation_matrix(roll, pitch, yaw)

        # Correct measurements for biases
        a_corr = imu.accel - b_a
        g_corr = imu.gyro  - b_g

        # Mechanization – Euler integration (RK4 optional)
        a_world = R @ a_corr + self.G
        new_vel = vel + a_world * dt
        new_pos = pos + vel * dt + 0.5 * a_world * dt**2

        # Attitude update via Euler angle rates
        # ω_body → Euler rates (via T matrix)
        T = self._euler_rate_matrix(roll, pitch)
        att_dot = T @ g_corr
        new_att = att + att_dot * dt
        new_att[2] = wrap_angle(new_att[2])

        # Build new state
        x_new = x.copy()
        x_new[0:3] = new_pos
        x_new[3:6] = new_vel
        x_new[6:9] = new_att
        self.x = x_new

        # Linearised state transition matrix F (15×15)
        F = self._compute_F(a_corr, R, roll, pitch, yaw, dt)

        # Covariance propagation: P = F P Fᵀ + Q
        self.P = F @ self.P @ F.T + self.Q * self._scale
        self.P = 0.5 * (self.P + self.P.T)  # Enforce symmetry

    # ── Measurement Updates ───────────────────────────────────────────────────

    def update_gnss_position(self, meas: GNSSMeasurement) -> bool:
        """3-state GNSS position update."""
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, 0:3] = np.eye(3)

        R_scale = (meas.accuracy ** 2)
        R = self.R_gnss_pos * R_scale

        innovation = meas.position - self.x[0:3]
        return self._update(H, innovation, R, tag="GNSS_POS")

    def update_gnss_velocity(self, meas: GNSSMeasurement) -> bool:
        if meas.velocity is None:
            return False
        H = np.zeros((3, self.STATE_DIM))
        H[0:3, 3:6] = np.eye(3)
        innovation = meas.velocity - self.x[3:6]
        return self._update(H, innovation, self.R_gnss_vel, tag="GNSS_VEL")

    def update_barometer(self, meas: BaroMeasurement, home_alt: float = 0.0) -> bool:
        H = np.zeros((1, self.STATE_DIM))
        H[0, 2] = 1.0   # Altitude component (ENU Up)
        R = np.array([[meas.noise_m ** 2]])
        innovation = np.array([meas.altitude - home_alt - self.x[2]])
        return self._update(H, innovation, R, tag="BARO")

    def update_magnetometer(self, meas: MagMeasurement) -> bool:
        """Pseudo-measurement: yaw from magnetometer heading."""
        roll, pitch, yaw = self.x[6:9]
        # Project to heading
        B = meas.field
        Bh_x = B[0]*np.cos(pitch) + B[2]*np.sin(pitch)
        Bh_y = B[0]*np.sin(roll)*np.sin(pitch) + B[1]*np.cos(roll) - B[2]*np.sin(roll)*np.cos(pitch)
        mag_yaw = np.arctan2(-Bh_y, Bh_x)

        H = np.zeros((1, self.STATE_DIM))
        H[0, 8] = 1.0  # yaw component
        innovation = np.array([wrap_angle(mag_yaw - yaw)])
        return self._update(H, innovation, self.R_mag, tag="MAG")

    # ── Core KF Update ────────────────────────────────────────────────────────

    def _update(self, H: np.ndarray, innovation: np.ndarray,
                R: np.ndarray, tag: str = "") -> bool:
        """Standard KF measurement update with chi-squared gating."""
        S = H @ self.P @ H.T + R
        S_inv = np.linalg.inv(S)

        # Innovation gating
        maha2 = float(innovation.T @ S_inv @ innovation)
        if maha2 > self._gate and self._n_updates > 50:
            log.warning(f"EKF {tag}: innovation gated χ²={maha2:.2f} > {self._gate:.2f}")
            return False

        K = self.P @ H.T @ S_inv        # Kalman gain
        self.x = self.x + K @ innovation
        self.x[6:9] = np.array([wrap_angle(a) for a in self.x[6:9]])

        # Joseph form for numerical stability
        I_KH = np.eye(self.STATE_DIM) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self._n_updates += 1

        if self._adaptive and maha2 < 1.0:
            self.Q *= 0.99   # Reduce process noise if filter is converging

        return True

    # ── Jacobians ─────────────────────────────────────────────────────────────

    def _compute_F(self, a_corr, R, roll, pitch, yaw, dt) -> np.ndarray:
        """Linearised state transition matrix (error-state formulation)."""
        F = np.eye(self.STATE_DIM)

        # ∂pos/∂vel
        F[0:3, 3:6] = np.eye(3) * dt

        # ∂vel/∂att  (via cross product with specific force)
        a_world = R @ a_corr
        F[3:6, 6:9] = -skew_symmetric(a_world) @ self._euler_rate_matrix(roll, pitch) * dt

        # ∂vel/∂accel_bias
        F[3:6, 12:15] = -R * dt

        # ∂att/∂gyro_bias
        T = self._euler_rate_matrix(roll, pitch)
        F[6:9, 9:12] = -T * dt

        return F

    @staticmethod
    def _euler_rate_matrix(roll: float, pitch: float) -> np.ndarray:
        """T matrix: ω_body → Euler angle rates (roll, pitch, yaw)."""
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        tp = np.tan(pitch)
        return np.array([
            [1.0, sr*tp,  cr*tp],
            [0.0, cr,     -sr  ],
            [0.0, sr/cp,  cr/cp],
        ])

    def _build_process_noise(self) -> np.ndarray:
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        # Position noise (from velocity integration)
        Q[0:3, 0:3]   = np.eye(3) * 1e-6
        # Velocity noise (from accelerometer)
        Q[3:6, 3:6]   = np.eye(3) * 0.0001
        # Attitude noise (from gyroscope)
        Q[6:9, 6:9]   = np.eye(3) * 0.000001
        # Gyro bias walk
        Q[9:12, 9:12] = np.eye(3) * 1e-9
        # Accel bias walk
        Q[12:15, 12:15] = np.eye(3) * 1e-8
        return Q

    # ── State Access ──────────────────────────────────────────────────────────

    @property
    def state(self) -> FusedState:
        return FusedState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            attitude=self.x[6:9].copy(),
            gyro_bias=self.x[9:12].copy(),
            accel_bias=self.x[12:15].copy(),
            covariance=self.P.copy(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Unscented Kalman Filter (UKF)
# ──────────────────────────────────────────────────────────────────────────────

class UnscentedKalmanFilter:
    """
    Sigma-point UKF for nonlinear sensor fusion.
    Uses scaled unscented transform (alpha, beta, kappa parameters).
    More accurate than EKF for highly nonlinear attitude dynamics.
    """

    STATE_DIM = 15

    def __init__(self, alpha: float = 0.001, beta: float = 2.0, kappa: float = 0.0):
        n = self.STATE_DIM
        self._alpha = alpha
        self._beta  = beta
        self._kappa = kappa
        self._lam   = alpha**2 * (n + kappa) - n
        self._n     = n

        # Weights
        self.Wm, self.Wc = self._compute_weights()

        # State
        self.x = np.zeros(n)
        self.P = np.eye(n)
        self.Q = np.eye(n) * 1e-6
        self._initialized = False

    def _compute_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self._n
        lam = self._lam
        N_sigma = 2*n + 1

        Wm = np.full(N_sigma, 1.0 / (2*(n + lam)))
        Wc = Wm.copy()
        Wm[0] = lam / (n + lam)
        Wc[0] = Wm[0] + (1 - self._alpha**2 + self._beta)

        return Wm, Wc

    def _sigma_points(self) -> np.ndarray:
        n = self._n
        sigma = np.zeros((2*n + 1, n))
        sigma[0] = self.x
        try:
            L = np.linalg.cholesky((n + self._lam) * self.P)
        except np.linalg.LinAlgError:
            self.P = 0.5*(self.P + self.P.T) + np.eye(n)*1e-9
            L = np.linalg.cholesky((n + self._lam) * self.P)
        for i in range(n):
            sigma[i+1]   = self.x + L[i]
            sigma[n+i+1] = self.x - L[i]
        return sigma

    def initialize(self, pos, vel, att) -> None:
        self.x[:3]  = pos
        self.x[3:6] = vel
        self.x[6:9] = att
        self.P = np.eye(self._n) * 0.1
        self._initialized = True
        log.info("UKF initialized")

    def predict(self, imu: IMUMeasurement) -> None:
        if not self._initialized:
            return
        sigmas = self._sigma_points()
        sigmas_pred = np.array([self._process_model(s, imu) for s in sigmas])

        # Predicted mean
        self.x = np.einsum("i,ij->j", self.Wm, sigmas_pred)
        self.x[6:9] = np.array([wrap_angle(a) for a in self.x[6:9]])

        # Predicted covariance
        diff = sigmas_pred - self.x
        self.P = np.einsum("i,ij,ik->jk", self.Wc, diff, diff) + self.Q
        self.P = 0.5*(self.P + self.P.T)

    def _process_model(self, x: np.ndarray, imu: IMUMeasurement) -> np.ndarray:
        """Nonlinear state transition (full mechanization)."""
        pos = x[0:3]; vel = x[3:6]; att = x[6:9]
        b_g = x[9:12]; b_a = x[12:15]
        dt = imu.dt

        R = euler_to_rotation_matrix(*att)
        a = R @ (imu.accel - b_a) + np.array([0, 0, -9.81])
        g = imu.gyro - b_g

        new_pos = pos + vel*dt + 0.5*a*dt**2
        new_vel = vel + a*dt

        sr, cr = np.sin(att[0]), np.cos(att[0])
        sp, cp = np.sin(att[1]), np.cos(att[1])
        tp = np.tan(att[1])
        T = np.array([
            [1, sr*tp, cr*tp],
            [0, cr,    -sr   ],
            [0, sr/cp, cr/cp ],
        ])
        new_att = att + T @ g * dt
        new_att[2] = wrap_angle(new_att[2])

        x_new = x.copy()
        x_new[0:3] = new_pos
        x_new[3:6] = new_vel
        x_new[6:9] = new_att
        return x_new

    def update_gnss(self, meas: GNSSMeasurement) -> None:
        if not self._initialized:
            return
        H_fn = lambda x: x[0:3]
        z = meas.position
        R = np.eye(3) * (meas.accuracy**2)
        self._ukf_update(H_fn, z, R)

    def _ukf_update(self, h_fn, z: np.ndarray, R: np.ndarray) -> None:
        sigmas = self._sigma_points()
        z_sigmas = np.array([h_fn(s) for s in sigmas])
        z_pred = np.einsum("i,ij->j", self.Wm, z_sigmas)

        Pzz = np.einsum("i,ij,ik->jk", self.Wc, z_sigmas - z_pred, z_sigmas - z_pred) + R
        Pxz = np.einsum("i,ij,ik->jk", self.Wc, sigmas - self.x, z_sigmas - z_pred)

        K = Pxz @ np.linalg.inv(Pzz)
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pzz @ K.T
        self.P = 0.5*(self.P + self.P.T)

    @property
    def state(self) -> FusedState:
        return FusedState(
            position=self.x[0:3].copy(),
            velocity=self.x[3:6].copy(),
            attitude=self.x[6:9].copy(),
            covariance=self.P.copy(),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Fusion Manager
# ──────────────────────────────────────────────────────────────────────────────

class SensorFusionManager:
    """
    Orchestrates EKF + UKF: runs both and blends based on GNSS quality.
    Falls back to UKF during high-dynamic manoeuvres.
    """

    def __init__(self, algorithm: str = "EKF"):
        self._algo = algorithm.upper()
        self.ekf = ExtendedKalmanFilter(adaptive=True)
        self.ukf = UnscentedKalmanFilter()
        self._gnss_quality = 1.0
        self._last_state = FusedState()

    def initialize(self, pos, vel, att) -> None:
        self.ekf.initialize(np.array(pos, dtype=float),
                            np.array(vel, dtype=float),
                            np.array(att, dtype=float))
        self.ukf.initialize(np.array(pos, dtype=float),
                            np.array(vel, dtype=float),
                            np.array(att, dtype=float))

    def step(self, imu: IMUMeasurement,
             gnss: Optional[GNSSMeasurement] = None,
             baro: Optional[BaroMeasurement] = None,
             mag:  Optional[MagMeasurement]  = None) -> FusedState:

        # Predict
        self.ekf.predict(imu)
        if self._algo in ("UKF", "hybrid"):
            self.ukf.predict(imu)

        # Update
        if gnss is not None:
            self._gnss_quality = min(1.0, gnss.fix_type / 3.0)
            self.ekf.update_gnss_position(gnss)
            self.ekf.update_gnss_velocity(gnss)
            if self._algo in ("UKF", "hybrid"):
                self.ukf.update_gnss(gnss)

        if baro is not None:
            self.ekf.update_barometer(baro)

        if mag is not None:
            self.ekf.update_magnetometer(mag)

        # Select output
        if self._algo == "hybrid":
            # Blend: EKF for low dynamics, UKF for high
            accel_mag = float(np.linalg.norm(imu.accel))
            high_dynamic = accel_mag > 15.0
            state = self.ukf.state if high_dynamic else self.ekf.state
        elif self._algo == "UKF":
            state = self.ukf.state
        else:
            state = self.ekf.state

        state.gnss_quality = self._gnss_quality
        self._last_state = state
        return state

    @property
    def state(self) -> FusedState:
        return self._last_state
