"""
=============================================================================
navigation/controller.py  –  Cascaded UAV Flight Controller
=============================================================================
Implements a cascaded PID + feedforward flight controller:
  Outer loop: Position → Velocity setpoint
  Middle loop: Velocity → Attitude setpoint
  Inner loop:  Attitude → Motor commands

Also provides:
  - Waypoint sequencer
  - Altitude hold
  - Heading controller
  - Emergency recovery logic
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from utils.math_utils import clamp, wrap_angle, euler_to_rotation_matrix
from utils.logger import get_logger

log = get_logger("NAV")


# ──────────────────────────────────────────────────────────────────────────────
# PID Controller (generic, vectorizable)
# ──────────────────────────────────────────────────────────────────────────────

class PIDController:
    """
    Anti-windup PID with derivative filtering and output clamping.

    Features:
      - Integral anti-windup (clamping + back-calculation)
      - Low-pass filtered derivative (τ_d parameter)
      - Feedforward term
      - Rate-mode option (no integral)
    """

    def __init__(
        self,
        kp: float, ki: float, kd: float,
        output_min: float = -float("inf"),
        output_max: float =  float("inf"),
        integral_limit: float = 100.0,
        derivative_filter_tau: float = 0.02,
    ):
        self.kp = kp; self.ki = ki; self.kd = kd
        self._out_min = output_min
        self._out_max = output_max
        self._i_lim   = integral_limit
        self._tau_d   = derivative_filter_tau

        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0
        self._prev_measurement = 0.0

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
        self._deriv_filtered = 0.0

    def update(self, error: float, dt: float, feedforward: float = 0.0,
               measurement: Optional[float] = None) -> float:
        if dt <= 0:
            return 0.0

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = clamp(self._integral, -self._i_lim, self._i_lim)
        i_term = self.ki * self._integral

        # Derivative on measurement (avoids derivative kick on setpoint changes)
        if measurement is not None:
            raw_deriv = -(measurement - self._prev_measurement) / dt
            self._prev_measurement = measurement
        else:
            raw_deriv = (error - self._prev_error) / dt

        # First-order low-pass filter on derivative
        alpha = dt / (self._tau_d + dt)
        self._deriv_filtered = (1 - alpha) * self._deriv_filtered + alpha * raw_deriv
        d_term = self.kd * self._deriv_filtered
        self._prev_error = error

        output = clamp(p_term + i_term + d_term + feedforward,
                       self._out_min, self._out_max)

        # Back-calculation anti-windup
        clamped_output = clamp(output, self._out_min, self._out_max)
        if self.ki > 0:
            self._integral += (clamped_output - output) / self.ki * dt * 0.5

        return clamped_output


class PIDVector3:
    """Three independent PID controllers for 3D vector control (x, y, z)."""

    def __init__(self, kp, ki, kd, out_min=-float("inf"), out_max=float("inf")):
        params = dict(output_min=out_min, output_max=out_max)
        if isinstance(kp, (list, tuple)):
            self._pids = [PIDController(kp[i], ki[i], kd[i], **params) for i in range(3)]
        else:
            self._pids = [PIDController(kp, ki, kd, **params) for _ in range(3)]

    def reset(self) -> None:
        for pid in self._pids: pid.reset()

    def update(self, error: np.ndarray, dt: float,
               feedforward: np.ndarray = None) -> np.ndarray:
        ff = feedforward if feedforward is not None else np.zeros(3)
        return np.array([
            self._pids[i].update(float(error[i]), dt, float(ff[i]))
            for i in range(3)
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Motor Mixer
# ──────────────────────────────────────────────────────────────────────────────

class MotorMixer:
    """
    Quadrotor X-configuration motor mixing.
    Maps [thrust, roll_torque, pitch_torque, yaw_torque] → 4 motor speeds.

    Motor layout (top view, X-config):
        M2(CCW) --- M1(CW)
           |    X    |
        M3(CW)  --- M4(CCW)
    """

    def __init__(self, arm_length: float = 0.23, k_thrust: float = 1.0,
                 k_torque: float = 0.016):
        l = arm_length
        k = k_torque
        # Mixing matrix columns: [T, τ_roll, τ_pitch, τ_yaw]
        self._mix = np.array([
            [1,  1,  1,  1],   # M1 (front-right, CW)
            [1, -1,  1, -1],   # M2 (front-left,  CCW)
            [1, -1, -1,  1],   # M3 (rear-left,   CW)
            [1,  1, -1, -1],   # M4 (rear-right,  CCW)
        ], dtype=float)
        self._mix_inv = np.linalg.pinv(self._mix)

    def mix(self, thrust: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Returns normalized motor commands [0, 1]."""
        cmd = np.array([thrust, roll, pitch, yaw])
        motor_cmds = self._mix @ cmd
        motor_cmds = np.clip(motor_cmds, 0.0, 1.0)
        return motor_cmds


# ──────────────────────────────────────────────────────────────────────────────
# Cascaded Flight Controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ControlOutput:
    """Full controller output for one timestep."""
    motor_commands: np.ndarray = field(default_factory=lambda: np.zeros(4))
    thrust:         float = 0.0
    roll_cmd:       float = 0.0
    pitch_cmd:      float = 0.0
    yaw_rate_cmd:   float = 0.0
    velocity_cmd:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_error: np.ndarray = field(default_factory=lambda: np.zeros(3))


class CascadedController:
    """
    Full cascaded PID flight controller for multirotor UAV.

    Architecture:
      Level 3: Position controller     → velocity setpoint
      Level 2: Velocity controller     → attitude + thrust setpoint
      Level 1: Attitude rate controller→ motor torque commands
    """

    def __init__(self, config: dict):
        uav = config.get("uav", {})
        self._mass    = uav.get("mass_kg", 1.5)
        self._max_vel = uav.get("max_velocity_ms", 15.0)
        self._max_acc = uav.get("max_acceleration_ms2", 8.0)
        self._max_tilt= math.radians(35.0)   # Max tilt angle [rad]
        self._g       = 9.81
        self._arm_len = uav.get("arm_length_m", 0.23)

        # ── Position PID (outer loop) ─────────────────────────────────
        self._pos_pid = PIDVector3(
            kp=[1.2, 1.2, 1.5],
            ki=[0.05, 0.05, 0.1],
            kd=[0.4, 0.4, 0.5],
            out_min=-self._max_vel,
            out_max= self._max_vel,
        )

        # ── Velocity PID (middle loop) ────────────────────────────────
        self._vel_pid = PIDVector3(
            kp=[2.5, 2.5, 3.0],
            ki=[0.1, 0.1, 0.2],
            kd=[0.3, 0.3, 0.4],
            out_min=-self._max_acc * self._mass,
            out_max= self._max_acc * self._mass,
        )

        # ── Attitude Rate PID (inner loop) ────────────────────────────
        self._roll_rate_pid  = PIDController(kp=5.0, ki=0.1, kd=0.3,
                                              output_min=-1.0, output_max=1.0)
        self._pitch_rate_pid = PIDController(kp=5.0, ki=0.1, kd=0.3,
                                              output_min=-1.0, output_max=1.0)
        self._yaw_rate_pid   = PIDController(kp=3.0, ki=0.05, kd=0.1,
                                              output_min=-1.0, output_max=1.0)

        # ── Heading controller ────────────────────────────────────────
        self._yaw_pid = PIDController(kp=1.5, ki=0.01, kd=0.2,
                                       output_min=-math.radians(90),
                                       output_max= math.radians(90))

        self._mixer = MotorMixer(arm_length=self._arm_len)
        self._mode  = "POSITION"   # POSITION | VELOCITY | ATTITUDE | STABILIZE

    def reset(self) -> None:
        self._pos_pid.reset()
        self._vel_pid.reset()
        self._roll_rate_pid.reset()
        self._pitch_rate_pid.reset()
        self._yaw_rate_pid.reset()
        self._yaw_pid.reset()

    def set_mode(self, mode: str) -> None:
        valid = {"POSITION", "VELOCITY", "ATTITUDE", "STABILIZE"}
        if mode in valid:
            self._mode = mode
        else:
            log.warning(f"Unknown controller mode: {mode}")

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        pos:     np.ndarray,   # Current ENU position [m]
        vel:     np.ndarray,   # Current ENU velocity [m/s]
        att:     np.ndarray,   # Current attitude [roll, pitch, yaw] rad
        omega:   np.ndarray,   # Current angular rates [rad/s]
        pos_sp:  np.ndarray,   # Position setpoint [m]
        vel_sp:  Optional[np.ndarray] = None,  # Optional velocity feedforward
        yaw_sp:  Optional[float]      = None,
        dt:      float = 0.02,
    ) -> ControlOutput:

        pos_err = pos_sp - pos
        roll, pitch, yaw = att

        # ── Level 3: Position → Velocity ─────────────────────────────
        vel_from_pos = self._pos_pid.update(pos_err, dt)

        # Combine with feedforward velocity
        if vel_sp is not None:
            vel_cmd = vel_from_pos + 0.5 * vel_sp
        else:
            vel_cmd = vel_from_pos

        vel_cmd = np.clip(vel_cmd, -self._max_vel, self._max_vel)

        # ── Level 2: Velocity → Thrust + Tilt ────────────────────────
        vel_err  = vel_cmd - vel
        force_sp = self._vel_pid.update(vel_err, dt,
                                         feedforward=self._mass * np.array([0,0,self._g]))

        # Gravity compensation
        thrust_z = force_sp[2]
        thrust    = clamp(thrust_z / (self._mass * self._g), 0.1, 1.0)

        # Desired horizontal accelerations → roll/pitch
        # In body frame: ax = g·tan(pitch), ay = -g·tan(roll)
        ax_des = force_sp[0] / self._mass
        ay_des = force_sp[1] / self._mass

        # Decouple via yaw rotation
        cy, sy = math.cos(yaw), math.sin(yaw)
        ax_body =  cy * ax_des + sy * ay_des
        ay_body = -sy * ax_des + cy * ay_des

        pitch_des = clamp(math.atan2(ax_body, self._g), -self._max_tilt, self._max_tilt)
        roll_des  = clamp(math.atan2(-ay_body, self._g), -self._max_tilt, self._max_tilt)

        # ── Level 1: Attitude → Angular Rates ────────────────────────
        roll_err  = roll_des  - roll
        pitch_err = pitch_des - pitch

        # Yaw control
        if yaw_sp is not None:
            yaw_err     = wrap_angle(yaw_sp - yaw)
            yaw_rate_sp = self._yaw_pid.update(yaw_err, dt)
        else:
            yaw_rate_sp = 0.0

        roll_torque  = self._roll_rate_pid.update(roll_err - omega[0], dt)
        pitch_torque = self._pitch_rate_pid.update(pitch_err - omega[1], dt)
        yaw_torque   = self._yaw_rate_pid.update(yaw_rate_sp - omega[2], dt)

        motor_cmds = self._mixer.mix(thrust, roll_torque, pitch_torque, yaw_torque)

        return ControlOutput(
            motor_commands=motor_cmds,
            thrust=thrust,
            roll_cmd=roll_des,
            pitch_cmd=pitch_des,
            yaw_rate_cmd=yaw_rate_sp,
            velocity_cmd=vel_cmd,
            position_error=pos_err,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint Sequencer
# ──────────────────────────────────────────────────────────────────────────────

class WaypointSequencer:
    """
    Manages sequential waypoint following with:
      - Radius-based waypoint capture
      - Loiter support
      - Auto-land / RTH (return to home)
      - Progress callbacks
    """

    def __init__(self, waypoint_radius: float = 2.0):
        self._radius    = waypoint_radius
        self._waypoints: List = []
        self._idx       = 0
        self._loiter_remaining = 0.0
        self._home_pos: Optional[np.ndarray] = None
        self._mission_complete = False
        self._reached_count   = 0

    def load_waypoints(self, waypoints: List) -> None:
        self._waypoints = list(waypoints)
        self._idx       = 0
        self._mission_complete = False
        self._reached_count   = 0
        log.info(f"Sequencer: Loaded {len(waypoints)} waypoints")

    def set_home(self, pos: np.ndarray) -> None:
        self._home_pos = pos.copy()

    def update(self, current_pos: np.ndarray, dt: float) -> Tuple[Optional[np.ndarray], bool]:
        """
        Returns (target_position, mission_complete).
        target_position is None when mission is complete.
        """
        if self._mission_complete or self._idx >= len(self._waypoints):
            self._mission_complete = True
            return None, True

        wp = self._waypoints[self._idx]
        wp_pos = wp.position if hasattr(wp, "position") else np.array(wp)
        dist   = np.linalg.norm(current_pos - wp_pos)

        # Loiter handling
        if self._loiter_remaining > 0:
            self._loiter_remaining -= dt
            return wp_pos, False

        # Waypoint capture
        if dist < self._radius:
            loiter = getattr(wp, "loiter_s", 0.0)
            if loiter > 0:
                self._loiter_remaining = loiter
                log.info(f"WP {self._idx}: loitering {loiter:.1f}s at {np.round(wp_pos,1)}")

            self._reached_count += 1
            log.flight_event = None  # placeholder
            log.info(f"WP {self._idx} reached | dist={dist:.2f}m | remaining={len(self._waypoints)-self._idx-1}")
            self._idx += 1

            if self._idx >= len(self._waypoints):
                self._mission_complete = True
                log.info("Mission complete!")
                return None, True

        wp = self._waypoints[self._idx]
        wp_pos = wp.position if hasattr(wp, "position") else np.array(wp)
        return wp_pos, False

    @property
    def current_wp_index(self) -> int:
        return self._idx

    @property
    def progress(self) -> float:
        if not self._waypoints:
            return 0.0
        return self._reached_count / len(self._waypoints)

    @property
    def mission_complete(self) -> bool:
        return self._mission_complete
