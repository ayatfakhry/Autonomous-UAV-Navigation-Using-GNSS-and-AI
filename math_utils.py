"""
=============================================================================
scripts/run_simulation.py  –  Full System Simulation Runner
=============================================================================
Runs the complete autonomous UAV system:
  - Initialises all subsystems
  - Runs real-time simulation loop
  - Feeds live data to the Dash dashboard
  - Saves flight data to disk

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --env mountain --model models/final_model
    python scripts/run_simulation.py --swarm --n-drones 5
    python scripts/run_simulation.py --no-dashboard  (headless)
=============================================================================
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from utils.data_recorder import FlightDataRecorder
from envs.uav_env import UAVNavigationEnv
from agents.swarm import SwarmCoordinator
from vision.pipeline import VisionPipeline
from dashboard.app import get_store, run_dashboard

log = get_logger("MISSION")


# ──────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Autonomous UAV Navigation Simulation")
    p.add_argument("--env",          default="urban",
                   choices=["urban", "forest", "mountain", "open_field"])
    p.add_argument("--model",        default=None,          help="Trained RL model path")
    p.add_argument("--algo",         default="SAC",         choices=["SAC", "PPO", "DQN"])
    p.add_argument("--swarm",        action="store_true",   help="Enable swarm mode")
    p.add_argument("--n-drones",     default=5,             type=int)
    p.add_argument("--no-dashboard", action="store_true",   help="Run headless")
    p.add_argument("--config",       default="configs/config.yaml")
    p.add_argument("--duration",     default=300,           type=float, help="Sim duration (s)")
    p.add_argument("--record",       action="store_true",   help="Save flight data")
    p.add_argument("--seed",         default=42,            type=int)
    p.add_argument("--port",         default=8050,          type=int)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Simulation Loop
# ──────────────────────────────────────────────────────────────────────────────

class SimulationRunner:
    """
    Main simulation loop integrating all UAV subsystems.
    Runs at configurable Hz and pushes telemetry to the dashboard store.
    """

    def __init__(self, config: dict, model=None,
                 swarm: bool = False, record: bool = False, seed: int = 42):
        self._config  = config
        self._model   = model
        self._swarm   = swarm
        self._record  = record
        self._seed    = seed
        self._running = False
        self._store   = get_store()

        random.seed(seed)
        np.random.seed(seed)

        # Environment
        self._env = UAVNavigationEnv(config, record=record)

        # Vision pipeline
        self._vision = VisionPipeline(config)

        # Swarm
        self._swarm_ctrl: Optional[SwarmCoordinator] = None
        if swarm:
            self._swarm_ctrl = SwarmCoordinator(config)

        # Data recorder
        self._recorder: Optional[FlightDataRecorder] = None
        if record:
            self._recorder = FlightDataRecorder(
                output_dir="logs/flights",
                session_id=f"sim_{int(time.time())}"
            )

        # Simulation timing
        env_cfg   = config.get("environment", {})
        self._dt  = env_cfg.get("time_step_s", 0.02)
        self._t   = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, duration_s: float = 300.0) -> None:
        """Run simulation for duration_s seconds (blocking)."""
        self._running = True
        obs, _ = self._env.reset(seed=self._seed)

        # Init swarm
        if self._swarm_ctrl:
            n = self._config.get("swarm", {}).get("num_agents", 5)
            for i in range(n):
                pos = self._env._dynamics.pos + np.array([
                    i * 8.0, 0.0, 0.0
                ])
                self._swarm_ctrl.register_agent(i, pos, is_leader=(i == 0))
            self._swarm_ctrl.set_goal(self._env._goal_pos)

        self._store.push_event("MISSION_START",
                                f"env={self._config.get('environment',{}).get('type','urban')}")
        log.info(f"Simulation started | dt={self._dt}s | duration={duration_s}s")

        step_count = 0
        real_t0    = time.perf_counter()

        while self._running and self._t < duration_s:
            t_step_start = time.perf_counter()

            # ── Get action from RL model or PID ────────────────────────
            if self._model is not None:
                action, _ = self._model.predict(obs, deterministic=True)
            else:
                action = self._default_action(obs)

            # ── Step environment ────────────────────────────────────────
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._t += self._dt
            step_count += 1

            done = terminated or truncated

            # ── Vision pipeline ─────────────────────────────────────────
            pos  = self._env._dynamics.pos
            alt  = float(pos[2])
            try:
                cam_frame  = self._env._sensors.camera.update(
                    pos, self._env._dynamics.vel,
                    self._env._dynamics.att,
                    self._env._obs_gen.all_obstacles,
                )
                lidar_scan = self._env._sensors.lidar.update(
                    pos, self._env._obs_gen.all_obstacles,
                    self._env._dynamics.att[2],
                )
                vis_out = self._vision.process(
                    cam_frame, lidar_scan, pos, alt,
                    self._env._obs_gen.all_obstacles, obs
                )
            except Exception:
                lidar_scan = None
                vis_out    = None

            # ── Swarm update ─────────────────────────────────────────────
            if self._swarm_ctrl:
                self._swarm_ctrl.step(self._dt)

            # ── Build telemetry for dashboard ────────────────────────────
            telem = self._build_telemetry(obs, reward, info, lidar_scan, vis_out, action)
            self._store.push(telem)

            # ── Log events ────────────────────────────────────────────────
            if info.get("collision"):
                self._store.push_event("COLLISION", f"pos={np.round(pos,1).tolist()}")
            if info.get("distance_to_goal", 999) < 3.0 and terminated:
                self._store.push_event("GOAL_REACHED", f"t={self._t:.1f}s")

            # ── Record ────────────────────────────────────────────────────
            if self._recorder:
                self._recorder.record(
                    t=self._t, pos=pos, vel=self._env._dynamics.vel,
                    att=self._env._dynamics.att,
                    battery_pct=self._env._dynamics.battery_pct,
                    gnss_quality=self._env._fusion.state.gnss_quality,
                    n_obstacles=len(self._env._obs_gen.all_obstacles),
                    action=action, reward=reward, mode="RL" if self._model else "PID",
                )

            # ── Reset on episode end ──────────────────────────────────────
            if done:
                reason = "GOAL" if terminated and info.get("distance_to_goal",999) < 3.0 \
                         else "COLLISION" if info.get("collision") else "TIMEOUT"
                self._store.push_event("EPISODE_END", f"reason={reason} t={self._t:.1f}s")
                log.info(f"Episode ended: {reason} at t={self._t:.1f}s, "
                         f"step={step_count}, reward={reward:.1f}")
                obs, _ = self._env.reset(seed=self._seed + step_count)

            # ── Real-time pacing ─────────────────────────────────────────
            elapsed = time.perf_counter() - t_step_start
            sleep_t = max(0, self._dt * 0.5 - elapsed)   # Run at 2× real-time
            if sleep_t > 0:
                time.sleep(sleep_t)

        self._running = False
        log.info(f"Simulation complete | steps={step_count} | t={self._t:.1f}s")
        self._store.push_event("MISSION_END", f"steps={step_count}")

        if self._recorder:
            paths = self._recorder.save()
            summary = self._recorder.summary()
            log.info(f"Flight data saved: {paths}")
            log.info(f"Flight summary: {summary}")

    def stop(self) -> None:
        self._running = False

    # ── Telemetry Builder ─────────────────────────────────────────────────────

    def _build_telemetry(self, obs, reward, info, lidar_scan, vis_out, action) -> dict:
        dyn     = self._env._dynamics
        pos     = dyn.pos
        vel     = dyn.vel
        att     = dyn.att
        telem   = self._env.telemetry

        telem.update({
            "t":           self._t,
            "reward":      float(reward),
            "action":      action.tolist(),
            "attitude":    att.tolist(),
            "angular_rates": dyn.omega.tolist(),
            "obstacles":   self._env._obs_gen.all_obstacles[:20],  # cap for JSON
            "lidar": {
                "ranges": lidar_scan.ranges.tolist() if lidar_scan is not None else [],
                "angles": lidar_scan.angles.tolist() if lidar_scan is not None else [],
            } if lidar_scan is not None else None,
            "vision": {
                "n_detections": len(vis_out.detections) if vis_out else 0,
                "n_landing_zones": len(vis_out.landing_zones) if vis_out else 0,
                "terrain": vis_out.terrain_class if vis_out else "unknown",
                "anomaly_score": vis_out.anomaly_score if vis_out else 0.0,
                "fps": vis_out.fps if vis_out else 0.0,
            },
            "swarm": self._swarm_ctrl.get_status() if self._swarm_ctrl else None,
        })
        return telem

    def _default_action(self, obs: np.ndarray) -> np.ndarray:
        """Simple proportional navigation when no RL model is loaded."""
        pos      = self._env._dynamics.pos
        goal     = self._env._goal_pos
        to_goal  = goal - pos
        dist     = np.linalg.norm(to_goal)
        max_vel  = self._env._max_vel

        # Normalise direction, scale by distance (slow down near goal)
        speed_factor = min(1.0, dist / 20.0)
        vel_cmd  = (to_goal / max(dist, 0.1)) * speed_factor
        vel_cmd  = np.clip(vel_cmd, -1, 1)

        # Yaw towards goal
        yaw_sp   = math.atan2(to_goal[1], to_goal[0])
        yaw_err  = yaw_sp - self._env._dynamics.att[2]
        yaw_cmd  = np.clip(yaw_err / math.pi, -1, 1)

        # Simple LiDAR obstacle avoidance
        lidar_sectors = obs[18:24]   # 6 sectors normalized
        min_sec_idx   = int(np.argmin(lidar_sectors))
        if lidar_sectors[min_sec_idx] < 0.15:
            # Steer away from closest obstacle
            avoid_angles = np.linspace(-math.pi, math.pi, 6)
            avoid_dir    = avoid_angles[min_sec_idx] + math.pi  # opposite direction
            vel_cmd[0]  += 0.5 * math.cos(avoid_dir)
            vel_cmd[1]  += 0.5 * math.sin(avoid_dir)
            vel_cmd      = np.clip(vel_cmd, -1, 1)

        return np.array([vel_cmd[0], vel_cmd[1], vel_cmd[2], yaw_cmd], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load & patch config
    cfg_loader = ConfigLoader(args.config)
    cfg_loader.patch("environment.type", args.env)
    cfg_loader.patch("swarm.enabled",    args.swarm)
    cfg_loader.patch("swarm.num_agents", args.n_drones)
    config = cfg_loader.raw

    log.info("=" * 60)
    log.info("  AUTONOMOUS UAV NAVIGATION SYSTEM")
    log.info("  Research-Grade Aerospace Simulation")
    log.info("=" * 60)

    # Load RL model if provided
    model = None
    if args.model:
        try:
            from agents.rl_agents import RLTrainingManager
            env_tmp = UAVNavigationEnv(config)
            trainer = RLTrainingManager(config)
            model   = trainer.load(args.model, env_tmp)
            log.info(f"RL model loaded: {args.model}")
        except Exception as e:
            log.warning(f"Could not load model ({e}); using PID navigation")

    # Create simulation runner
    runner = SimulationRunner(
        config   = config,
        model    = model,
        swarm    = args.swarm,
        record   = args.record,
        seed     = args.seed,
    )

    # Start dashboard in background thread (if not headless)
    if not args.no_dashboard:
        dash_thread = threading.Thread(
            target=run_dashboard,
            kwargs=dict(config=config, port=args.port, debug=False),
            daemon=True,
        )
        dash_thread.start()
        log.info(f"Dashboard: http://localhost:{args.port}")
        time.sleep(2.0)   # Give Dash time to start

    # Run simulation (blocking)
    try:
        runner.start(duration_s=args.duration)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
        runner.stop()


if __name__ == "__main__":
    main()
