# ============================================================
# Autonomous UAV Navigation System - Master Configuration
# ============================================================

system:
  name: "AutonomousUAV-NavSystem"
  version: "1.0.0"
  debug: false
  log_level: "INFO"
  seed: 42

# ── UAV Physical Parameters ──────────────────────────────────
uav:
  mass_kg: 1.5
  arm_length_m: 0.23
  max_thrust_N: 25.0
  max_velocity_ms: 15.0
  max_acceleration_ms2: 8.0
  max_angular_rate_rads: 3.14
  battery_capacity_mAh: 5000
  hover_current_A: 8.5
  motor_count: 4
  drag_coefficient: 0.1
  inertia:
    Ixx: 0.0123
    Iyy: 0.0123
    Izz: 0.0246

# ── Environment Settings ─────────────────────────────────────
environment:
  type: "urban"           # urban | forest | mountain | open_field
  grid_size: [200, 200, 100]  # x, y, z meters
  time_step_s: 0.02       # 50 Hz simulation
  gravity_ms2: 9.81
  air_density_kgm3: 1.225
  wind:
    enabled: true
    base_speed_ms: 3.0
    turbulence_intensity: 0.4
    gust_probability: 0.05
    gust_max_ms: 8.0
  obstacles:
    static_count: 50
    dynamic_count: 5
    dynamic_speed_ms: 2.0
  terrain:
    resolution_m: 1.0
    max_elevation_m: 50.0

# ── Sensor Configuration ─────────────────────────────────────
sensors:
  gnss:
    enabled: true
    update_rate_hz: 10
    position_noise_m: 0.5
    velocity_noise_ms: 0.05
    rtk_enabled: true
    rtk_noise_m: 0.02
    dropout_probability: 0.01
    dropout_duration_s: 2.0
  imu:
    enabled: true
    update_rate_hz: 200
    accel_noise_ms2: 0.01
    gyro_noise_rads: 0.001
    accel_bias_ms2: 0.005
    gyro_bias_rads: 0.0005
    vibration_noise: 0.002
  barometer:
    enabled: true
    update_rate_hz: 50
    altitude_noise_m: 0.1
    temperature_drift: 0.01
  magnetometer:
    enabled: true
    update_rate_hz: 50
    noise_gauss: 0.005
    hard_iron_bias: [0.1, -0.05, 0.02]
  lidar:
    enabled: true
    update_rate_hz: 20
    range_m: 100.0
    range_noise_m: 0.05
    fov_deg: 360
    num_beams: 360
  camera:
    enabled: true
    update_rate_hz: 30
    resolution: [640, 480]
    fov_deg: 90
    noise_std: 5.0

# ── Sensor Fusion (EKF/UKF) ──────────────────────────────────
fusion:
  algorithm: "EKF"        # EKF | UKF | hybrid
  state_dimension: 15     # [pos(3), vel(3), att(3), gyro_bias(3), accel_bias(3)]
  process_noise_scale: 1.0
  measurement_noise_scale: 1.0
  ekf:
    adaptive: true
    innovation_gate: 9.21  # chi-squared 99% for 3DOF
  ukf:
    alpha: 0.001
    beta: 2.0
    kappa: 0.0

# ── Navigation & Path Planning ────────────────────────────────
navigation:
  algorithm: "hybrid"     # astar | rrt_star | hybrid | rl_only
  waypoint_radius_m: 2.0
  lookahead_distance_m: 5.0
  safety_margin_m: 2.5
  astar:
    heuristic: "euclidean"
    grid_resolution_m: 1.0
    diagonal_movement: true
  rrt_star:
    max_iterations: 5000
    step_size_m: 2.0
    search_radius_m: 10.0
    rewire: true
  dynamic_avoidance:
    enabled: true
    prediction_horizon_s: 3.0
    reaction_distance_m: 10.0

# ── Reinforcement Learning ────────────────────────────────────
rl:
  algorithm: "SAC"        # PPO | SAC | DQN
  policy: "MlpPolicy"
  observation_dim: 48
  action_dim: 4           # [vx, vy, vz, yaw_rate]
  action_bounds: [-1.0, 1.0]
  gamma: 0.99
  learning_rate: 3.0e-4
  batch_size: 256
  buffer_size: 1000000
  tau: 0.005
  training:
    total_timesteps: 2000000
    eval_freq: 10000
    n_eval_episodes: 10
    save_freq: 50000
    checkpoint_dir: "models/checkpoints"
  rewards:
    waypoint_reach: 100.0
    collision_penalty: -200.0
    step_penalty: -0.1
    velocity_efficiency: 1.0
    altitude_violation: -50.0
    energy_efficiency: 0.5
    smoothness_reward: 0.3

# ── Computer Vision ───────────────────────────────────────────
vision:
  detector: "yolov8"      # yolov8 | cnn | vit
  confidence_threshold: 0.5
  nms_threshold: 0.4
  landing_zone:
    min_area_m2: 4.0
    flatness_threshold: 0.1
    obstacle_clearance_m: 3.0
  tracking:
    max_distance_m: 50.0
    max_disappeared_frames: 30
    use_kalman: true

# ── Mission Control Dashboard ─────────────────────────────────
dashboard:
  host: "0.0.0.0"
  port: 8050
  update_interval_ms: 100
  map_style: "dark"
  theme: "dark_aerospace"
  max_trajectory_points: 1000
  enable_3d_view: true

# ── Swarm Configuration ───────────────────────────────────────
swarm:
  enabled: false
  num_agents: 5
  formation: "v_formation"   # v_formation | line | circle | grid
  separation_distance_m: 8.0
  cohesion_weight: 0.3
  separation_weight: 0.5
  alignment_weight: 0.2
  communication_range_m: 50.0
