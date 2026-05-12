"""
=============================================================================
utils/timer.py & data_recorder.py – Performance Profiling & Flight Recording
=============================================================================
"""

from __future__ import annotations

import time
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# PerformanceTimer
# ══════════════════════════════════════════════════════════════════════════════

class PerformanceTimer:
    """
    Context-manager and decorator for profiling code blocks.

    Usage:
        timer = PerformanceTimer()
        with timer("sensor_fusion"):
            fuse()
        print(timer.report())
    """

    def __init__(self):
        self._records: Dict[str, List[float]] = defaultdict(list)
        self._start: Dict[str, float] = {}
        self._current_tag: Optional[str] = None

    def __call__(self, tag: str) -> "PerformanceTimer":
        self._current_tag = tag
        return self

    def __enter__(self):
        self._start[self._current_tag] = time.perf_counter()
        return self

    def __exit__(self, *_):
        tag = self._current_tag
        elapsed = time.perf_counter() - self._start[tag]
        self._records[tag].append(elapsed * 1e3)  # store in ms

    def record(self, tag: str, elapsed_s: float) -> None:
        self._records[tag].append(elapsed_s * 1e3)

    def mean_ms(self, tag: str) -> float:
        data = self._records.get(tag, [])
        return float(np.mean(data)) if data else 0.0

    def max_ms(self, tag: str) -> float:
        data = self._records.get(tag, [])
        return float(np.max(data)) if data else 0.0

    def report(self) -> str:
        lines = ["── Performance Report ─────────────────"]
        for tag, vals in sorted(self._records.items()):
            arr = np.array(vals)
            lines.append(
                f"  {tag:<30s} mean={arr.mean():.2f}ms  "
                f"max={arr.max():.2f}ms  n={len(vals)}"
            )
        lines.append("────────────────────────────────────────")
        return "\n".join(lines)

    def reset(self) -> None:
        self._records.clear()
        self._start.clear()


# ══════════════════════════════════════════════════════════════════════════════
# FlightDataRecorder
# ══════════════════════════════════════════════════════════════════════════════

class FlightDataRecorder:
    """
    High-frequency black-box recorder for flight data.

    Writes telemetry to:
      - CSV  (full numeric log, pandas-friendly)
      - JSON (structured events / anomalies)
      - NPZ  (numpy arrays for post-processing / plotting)
    """

    DEFAULT_FIELDS = [
        "timestamp", "t",
        "pos_x", "pos_y", "pos_z",
        "vel_x", "vel_y", "vel_z",
        "roll", "pitch", "yaw",
        "battery_pct",
        "gnss_quality",
        "n_obstacles",
        "action_vx", "action_vy", "action_vz", "action_yaw",
        "reward",
        "mode",
    ]

    def __init__(self, output_dir: str = "logs/flights", session_id: Optional[str] = None):
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._session = session_id or f"flight_{int(time.time())}"
        self._csv_path  = self._dir / f"{self._session}.csv"
        self._json_path = self._dir / f"{self._session}_events.json"
        self._npz_path  = self._dir / f"{self._session}_arrays.npz"

        self._rows: List[Dict] = []
        self._events: List[Dict] = []
        self._arrays: Dict[str, List] = defaultdict(list)

        # Open CSV writer
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.DEFAULT_FIELDS)
        self._csv_writer.writeheader()

    # ── Recording API ─────────────────────────────────────────────────────────

    def record(self, t: float, pos, vel, att, battery_pct: float,
               gnss_quality: float = 1.0, n_obstacles: int = 0,
               action=None, reward: float = 0.0, mode: str = "AUTO") -> None:
        action = action if action is not None else [0, 0, 0, 0]
        row = {
            "timestamp": time.time(),
            "t": round(t, 4),
            "pos_x": round(float(pos[0]), 4),
            "pos_y": round(float(pos[1]), 4),
            "pos_z": round(float(pos[2]), 4),
            "vel_x": round(float(vel[0]), 4),
            "vel_y": round(float(vel[1]), 4),
            "vel_z": round(float(vel[2]), 4),
            "roll":  round(float(att[0]), 4),
            "pitch": round(float(att[1]), 4),
            "yaw":   round(float(att[2]), 4),
            "battery_pct": round(battery_pct, 2),
            "gnss_quality": round(gnss_quality, 3),
            "n_obstacles": n_obstacles,
            "action_vx":  round(float(action[0]), 4),
            "action_vy":  round(float(action[1]), 4),
            "action_vz":  round(float(action[2]), 4),
            "action_yaw": round(float(action[3]), 4),
            "reward": round(reward, 4),
            "mode": mode,
        }
        self._csv_writer.writerow(row)
        self._rows.append(row)
        # Accumulate arrays
        for key in ["pos_x","pos_y","pos_z","vel_x","vel_y","vel_z","reward"]:
            self._arrays[key].append(row[key])

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        self._events.append({
            "type": event_type,
            "timestamp": time.time(),
            **data
        })

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> Dict[str, str]:
        self._csv_file.flush()
        self._csv_file.close()

        with open(self._json_path, "w") as f:
            json.dump(self._events, f, indent=2)

        arrays = {k: np.array(v) for k, v in self._arrays.items()}
        np.savez_compressed(self._npz_path, **arrays)

        return {
            "csv":  str(self._csv_path),
            "json": str(self._json_path),
            "npz":  str(self._npz_path),
        }

    def summary(self) -> Dict[str, Any]:
        """Compute flight statistics from recorded data."""
        if not self._rows:
            return {}
        pos = np.array([[r["pos_x"], r["pos_y"], r["pos_z"]] for r in self._rows])
        rewards = np.array([r["reward"] for r in self._rows])
        batt = [r["battery_pct"] for r in self._rows]
        total_dist = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
        return {
            "total_steps":    len(self._rows),
            "flight_time_s":  self._rows[-1]["t"] if self._rows else 0,
            "total_distance_m": round(total_dist, 2),
            "cumulative_reward": round(float(rewards.sum()), 2),
            "mean_reward":     round(float(rewards.mean()), 4),
            "battery_used_pct": round(batt[0] - batt[-1], 2),
            "n_events":       len(self._events),
            "session_id":     self._session,
        }
