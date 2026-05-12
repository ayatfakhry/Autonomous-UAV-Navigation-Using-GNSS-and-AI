# System Architecture

## Overview

The Autonomous UAV Navigation System uses a layered, modular architecture designed for extensibility, testability, and real-time performance.

```
┌─────────────────────────────────────────────────────────┐
│                  MISSION CONTROL DASHBOARD               │
│         (Plotly Dash — Real-time WebSocket UI)           │
└───────────────────────┬─────────────────────────────────┘
                        │ Telemetry
┌───────────────────────▼─────────────────────────────────┐
│                    MISSION PLANNER                        │
│         Waypoint Sequencer │ Path Planner                 │
│         A* │ RRT* │ Hybrid │ Dynamic Avoidance            │
└───────────────────────┬─────────────────────────────────┘
                        │ Target Position
┌───────────────────────▼─────────────────────────────────┐
│                   AI CONTROLLER LAYER                     │
│                                                           │
│  ┌─────────────────┐      ┌──────────────────────────┐  │
│  │  RL Policy       │      │   Cascaded PID           │  │
│  │  SAC / PPO / DQN │◄────►│   Pos→Vel→Att→Motor     │  │
│  │  Transformer     │      │   Anti-windup, Feedfwd   │  │
│  └────────┬────────┘      └──────────┬───────────────┘  │
│           └─────────Hybrid blend─────┘                    │
└───────────────────────┬─────────────────────────────────┘
                        │ Motor Commands [4]
┌───────────────────────▼─────────────────────────────────┐
│                   SIMULATION ENGINE                       │
│   6-DOF Newton-Euler Dynamics │ Wind (Dryden turbulence) │
│   Battery model │ Ground effect │ Obstacle environment    │
└───────────────────────┬─────────────────────────────────┘
                        │ True State
┌───────────────────────▼─────────────────────────────────┐
│                   SENSOR LAYER                            │
│                                                           │
│  GNSS/RTK  IMU    Baro   Mag   LiDAR   Camera            │
│  (10Hz)  (200Hz) (50Hz) (50Hz) (20Hz)  (30Hz)           │
│                                                           │
│  Dropout │ Noise │ Bias │ Drift │ Vibration models        │
└───────────────────────┬─────────────────────────────────┘
                        │ Noisy Measurements
┌───────────────────────▼─────────────────────────────────┐
│                   SENSOR FUSION                           │
│                                                           │
│  ┌─────────────────────┐   ┌──────────────────────────┐ │
│  │  Extended KF (EKF)  │   │  Unscented KF (UKF)      │ │
│  │  15-state INS       │   │  Sigma-point transform    │ │
│  │  Chi² gating        │   │  High-dynamic manoeuvres  │ │
│  │  Adaptive Q         │   │                           │ │
│  └──────────┬──────────┘   └──────────┬───────────────┘ │
│             └────────Hybrid Fusion─────┘                  │
└───────────────────────┬─────────────────────────────────┘
                        │ Fused State (pos, vel, att)
┌───────────────────────▼─────────────────────────────────┐
│                   VISION PIPELINE                         │
│                                                           │
│  YOLOv8 Detection │ SORT Tracking │ Landing Zone          │
│  Terrain Classify │ Optical Flow │ Anomaly Detection      │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

### Control Loop (50 Hz / 20ms)
1. **Read sensors** → IMU at 200Hz, GNSS at 10Hz, LiDAR at 20Hz
2. **Sensor fusion** → EKF/UKF integrates IMU (predict) + GNSS/Baro/Mag (update)
3. **Get fused state** → position, velocity, attitude, biases
4. **AI decision** → RL policy or PID produces velocity command
5. **Path target** → Mission planner provides immediate target position
6. **Dynamics step** → 6-DOF Euler integration
7. **Dashboard push** → Telemetry to Dash store

### Planning Loop (on demand)
1. Waypoint received → Occupancy map built from obstacle list
2. A* or RRT* plans path through free space
3. Path smoothed (gradient descent)
4. Waypoint sequencer tracks progress

## State Space

### EKF State Vector (15-dim)
```
x = [pos_E, pos_N, pos_U,     # ENU position [m]
     vel_E, vel_N, vel_U,     # ENU velocity [m/s]
     φ, θ, ψ,                 # Roll, Pitch, Yaw [rad]
     b_gx, b_gy, b_gz,        # Gyro bias [rad/s]
     b_ax, b_ay, b_az]        # Accel bias [m/s²]
```

### RL Observation (48-dim)
```
obs = [pos_norm(3), vel_norm(3), att(3), omega_norm(3),  # 12
       goal_dir(3), goal_dist(1), battery(1), gnss_q(1),  # 6
       lidar_sectors(6), wind_est(3), prev_action(4),      # 13 → pad to 18
       nearest_obs_offsets(18)]                            # 18  → total 48
```

## Threading Model

```
Main Thread                Dashboard Thread
    │                           │
    ├── SimulationRunner         ├── Dash App (Flask/Werkzeug)
    │   └── step() @ 50Hz        │   └── Callbacks @ 200ms polling
    │       └── push telemetry   │       └── read from TelemetryStore
    │           to TelemetryStore│
    │                           │
    └── Blocks on duration_s     └── daemon=True, auto-killed
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| EKF over UKF as default | 5× faster; sufficient accuracy for ≤15m/s dynamics |
| Hybrid EKF/UKF | UKF activated only when accel > 15 m/s² |
| SAC over PPO | Continuous action space; off-policy, sample-efficient |
| A\* + RRT\* | A\* fast in dense maps; RRT\* optimal in open spaces |
| 48-dim observation | Covers all safety-relevant state without overloading policy |
| Motor Mixer (X-config) | Standard quadrotor; easily adapted to hexarotor |
| Plotly Dash | Zero-install web dashboard; works over SSH tunnel |
