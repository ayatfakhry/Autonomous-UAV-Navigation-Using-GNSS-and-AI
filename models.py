"""
=============================================================================
navigation/path_planning.py  –  Advanced 3D Path Planning Algorithms
=============================================================================
Implements:
  1. A* (3D grid-based, with diagonal movement and height penalties)
  2. RRT* (Rapidly-exploring Random Trees with rewiring)
  3. Hybrid A*-RRT* with smoothing
  4. Dynamic obstacle avoidance (velocity obstacles / social force)
=============================================================================
"""

from __future__ import annotations

import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.math_utils import normalize_vector
from utils.logger import get_logger

log = get_logger("NAV")


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Waypoint:
    position: np.ndarray       # [x, y, z] in meters (ENU)
    heading:  Optional[float] = None  # desired yaw [rad]
    speed_ms: float = 5.0      # approach speed
    loiter_s: float = 0.0      # hold time [s]
    label:    str   = ""

    def distance_to(self, other: "Waypoint") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def __repr__(self):
        p = np.round(self.position, 1)
        return f"WP[{p[0]},{p[1]},{p[2]}]"


@dataclass
class Path:
    waypoints: List[Waypoint] = field(default_factory=list)
    total_length_m: float = 0.0
    planning_time_s: float = 0.0
    algorithm: str = ""

    def smooth(self, weight_data: float = 0.5, weight_smooth: float = 0.3,
               iterations: int = 100) -> "Path":
        """Gradient-descent path smoothing (preserves endpoints)."""
        if len(self.waypoints) < 3:
            return self
        pts = np.array([w.position for w in self.waypoints])
        original = pts.copy()
        for _ in range(iterations):
            for i in range(1, len(pts) - 1):
                pts[i] += weight_data * (original[i] - pts[i])
                pts[i] += weight_smooth * (pts[i-1] + pts[i+1] - 2*pts[i])
        smoothed_wps = []
        for i, wp in enumerate(self.waypoints):
            new_wp = Waypoint(
                position=pts[i].copy(),
                heading=wp.heading,
                speed_ms=wp.speed_ms,
                loiter_s=wp.loiter_s,
                label=wp.label,
            )
            smoothed_wps.append(new_wp)
        new_path = Path(waypoints=smoothed_wps, algorithm=self.algorithm + "_smoothed")
        new_path.total_length_m = sum(
            float(np.linalg.norm(smoothed_wps[i+1].position - smoothed_wps[i].position))
            for i in range(len(smoothed_wps)-1)
        )
        return new_path

    def __len__(self) -> int:
        return len(self.waypoints)


# ──────────────────────────────────────────────────────────────────────────────
# Occupancy Map
# ──────────────────────────────────────────────────────────────────────────────

class OccupancyMap3D:
    """
    Efficient 3D occupancy grid for path planning.
    Inflates obstacles by safety margin.
    """

    def __init__(self, bounds: Tuple[float,...], resolution: float = 1.0,
                 safety_margin: float = 2.5):
        self._res = resolution
        self._margin = safety_margin
        self._bounds = bounds  # (x_min, x_max, y_min, y_max, z_min, z_max)

        x_range = int((bounds[1] - bounds[0]) / resolution) + 1
        y_range = int((bounds[3] - bounds[2]) / resolution) + 1
        z_range = int((bounds[5] - bounds[4]) / resolution) + 1
        self._grid = np.zeros((x_range, y_range, z_range), dtype=bool)
        self._shape = (x_range, y_range, z_range)

    def _to_idx(self, pos: np.ndarray) -> Tuple[int,int,int]:
        ix = int((pos[0] - self._bounds[0]) / self._res)
        iy = int((pos[1] - self._bounds[2]) / self._res)
        iz = int((pos[2] - self._bounds[4]) / self._res)
        return (
            max(0, min(ix, self._shape[0]-1)),
            max(0, min(iy, self._shape[1]-1)),
            max(0, min(iz, self._shape[2]-1)),
        )

    def _to_pos(self, idx: Tuple[int,int,int]) -> np.ndarray:
        return np.array([
            idx[0] * self._res + self._bounds[0],
            idx[1] * self._res + self._bounds[2],
            idx[2] * self._res + self._bounds[4],
        ])

    def add_obstacle(self, center: np.ndarray, radius: float) -> None:
        """Mark obstacle cells (inflated by safety margin)."""
        inflated_r = radius + self._margin
        r_cells = int(math.ceil(inflated_r / self._res))
        cx, cy, cz = self._to_idx(center)
        for dx in range(-r_cells, r_cells+1):
            for dy in range(-r_cells, r_cells+1):
                for dz in range(-r_cells, r_cells+1):
                    ix, iy, iz = cx+dx, cy+dy, cz+dz
                    if not (0 <= ix < self._shape[0] and
                            0 <= iy < self._shape[1] and
                            0 <= iz < self._shape[2]):
                        continue
                    dist = math.sqrt((dx**2 + dy**2 + dz**2)) * self._res
                    if dist <= inflated_r:
                        self._grid[ix, iy, iz] = True

    def is_free(self, pos: np.ndarray) -> bool:
        idx = self._to_idx(pos)
        return not self._grid[idx[0], idx[1], idx[2]]

    def is_collision_free(self, p1: np.ndarray, p2: np.ndarray,
                           n_checks: int = 10) -> bool:
        """Line-segment collision check."""
        for t in np.linspace(0, 1, n_checks):
            pt = p1 + t * (p2 - p1)
            if not self.is_free(pt):
                return False
        return True

    def build_from_obstacles(self, obstacles: List[dict]) -> None:
        self._grid[:] = False
        for obs in obstacles:
            self.add_obstacle(
                np.array(obs["position"]),
                obs.get("radius", 1.0)
            )
        log.info(f"OccupancyMap: {self._grid.sum()} occupied cells / {self._grid.size} total")


# ──────────────────────────────────────────────────────────────────────────────
# A* Path Planner
# ──────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    3D grid A* with:
      - 26-connected graph (full diagonal in 3D)
      - Weighted heuristic (ε-A* for speed/optimality tradeoff)
      - Altitude penalty (prefer lower, safer altitudes)
    """

    def __init__(self, occ_map: OccupancyMap3D, epsilon: float = 1.2):
        self._map = occ_map
        self._eps = epsilon

    def plan(self, start: np.ndarray, goal: np.ndarray,
             timeout_s: float = 5.0) -> Optional[Path]:
        t0 = time.perf_counter()

        s_idx = self._map._to_idx(start)
        g_idx = self._map._to_idx(goal)

        if not self._map.is_free(goal):
            log.warning("A*: Goal position is occupied!")
            return None

        # Priority queue: (f, g, idx)
        open_set: List[Tuple] = []
        heapq.heappush(open_set, (0.0, 0.0, s_idx))

        came_from: Dict[Tuple, Tuple] = {}
        g_score: Dict[Tuple, float] = {s_idx: 0.0}
        closed_set = set()

        # 26-connected neighbors in 3D
        directions = [
            (dx, dy, dz)
            for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        while open_set and (time.perf_counter() - t0) < timeout_s:
            _, g_curr, curr = heapq.heappop(open_set)
            if curr in closed_set:
                continue
            closed_set.add(curr)

            if curr == g_idx:
                path_indices = self._reconstruct(came_from, curr)
                t_plan = time.perf_counter() - t0
                waypoints = [
                    Waypoint(position=self._map._to_pos(idx)) for idx in path_indices
                ]
                p = Path(waypoints=waypoints, algorithm="A*", planning_time_s=t_plan)
                p.total_length_m = sum(
                    float(np.linalg.norm(waypoints[i+1].position - waypoints[i].position))
                    for i in range(len(waypoints)-1)
                )
                log.info(f"A*: Found path | {len(waypoints)} WPs | {p.total_length_m:.1f}m | {t_plan*1000:.0f}ms")
                return p.smooth()

            for d in directions:
                nb = (curr[0]+d[0], curr[1]+d[1], curr[2]+d[2])
                if not (0 <= nb[0] < self._map._shape[0] and
                        0 <= nb[1] < self._map._shape[1] and
                        0 <= nb[2] < self._map._shape[2]):
                    continue
                if nb in closed_set:
                    continue
                pos_nb = self._map._to_pos(nb)
                if not self._map.is_free(pos_nb):
                    continue

                step = math.sqrt(sum(x**2 for x in d)) * self._map._res
                altitude_cost = 0.01 * pos_nb[2]  # Slight preference for lower altitude
                tentative_g = g_score[curr] + step + altitude_cost

                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = curr
                    g_score[nb]   = tentative_g
                    h = self._heuristic(nb, g_idx)
                    heapq.heappush(open_set, (tentative_g + self._eps*h, tentative_g, nb))

        log.warning(f"A*: No path found (timeout={timeout_s}s, closed={len(closed_set)} nodes)")
        return None

    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        """Octile distance heuristic (admissible for 26-connected grid)."""
        dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1]); dz = abs(a[2]-b[2])
        s  = sorted([dx, dy, dz])
        return self._map._res * (s[2] + (math.sqrt(3)-1)*s[1] + (math.sqrt(2)-math.sqrt(3))*s[0])

    def _reconstruct(self, came_from: dict, current: tuple) -> List[tuple]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# ──────────────────────────────────────────────────────────────────────────────
# RRT* Path Planner
# ──────────────────────────────────────────────────────────────────────────────

class RRTStarPlanner:
    """
    Asymptotically-optimal RRT* in 3D continuous space.
    Features: rewiring, goal biasing, near-neighbour radius decay.
    """

    @dataclass
    class Node:
        pos:    np.ndarray
        parent: Optional[int] = None
        cost:   float = 0.0
        idx:    int = 0

    def __init__(self, occ_map: OccupancyMap3D, config: dict):
        self._map         = occ_map
        self._max_iter    = config.get("max_iterations", 5000)
        self._step        = config.get("step_size_m", 2.0)
        self._radius      = config.get("search_radius_m", 10.0)
        self._goal_bias   = 0.10   # 10% goal sampling
        self._bounds      = occ_map._bounds
        self._nodes: List[RRTStarPlanner.Node] = []

    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[Path]:
        t0 = time.perf_counter()
        self._nodes.clear()

        root = self.Node(pos=start.copy(), parent=None, cost=0.0, idx=0)
        self._nodes.append(root)

        best_goal_idx = None
        best_goal_cost = float("inf")

        for i in range(self._max_iter):
            # Sample
            if random.random() < self._goal_bias:
                q_rand = goal.copy()
            else:
                q_rand = self._sample_random()

            # Nearest
            near_idx = self._nearest(q_rand)
            q_near   = self._nodes[near_idx].pos

            # Steer
            q_new = self._steer(q_near, q_rand)

            if not self._map.is_collision_free(q_near, q_new):
                continue

            # Near neighbours
            near_indices = self._near(q_new)
            best_parent  = near_idx
            best_cost    = self._nodes[near_idx].cost + np.linalg.norm(q_new - q_near)

            for ni in near_indices:
                c = self._nodes[ni].cost + np.linalg.norm(q_new - self._nodes[ni].pos)
                if c < best_cost and self._map.is_collision_free(self._nodes[ni].pos, q_new):
                    best_cost   = c
                    best_parent = ni

            new_idx = len(self._nodes)
            new_node = self.Node(pos=q_new, parent=best_parent, cost=best_cost, idx=new_idx)
            self._nodes.append(new_node)

            # Rewire
            for ni in near_indices:
                c_through = best_cost + np.linalg.norm(self._nodes[ni].pos - q_new)
                if c_through < self._nodes[ni].cost and \
                   self._map.is_collision_free(q_new, self._nodes[ni].pos):
                    self._nodes[ni].parent = new_idx
                    self._nodes[ni].cost   = c_through

            # Check goal
            if np.linalg.norm(q_new - goal) < self._step:
                if best_cost < best_goal_cost and self._map.is_collision_free(q_new, goal):
                    best_goal_cost = best_cost
                    best_goal_idx  = new_idx

        if best_goal_idx is None:
            log.warning("RRT*: No path found")
            return None

        path_positions = self._extract_path(best_goal_idx, goal)
        t_plan = time.perf_counter() - t0
        waypoints = [Waypoint(position=p) for p in path_positions]
        path = Path(waypoints=waypoints, algorithm="RRT*", planning_time_s=t_plan)
        path.total_length_m = best_goal_cost
        log.info(f"RRT*: Path found | {len(waypoints)} WPs | {path.total_length_m:.1f}m | {t_plan*1000:.0f}ms")
        return path.smooth()

    def _sample_random(self) -> np.ndarray:
        b = self._bounds
        return np.array([
            random.uniform(b[0], b[1]),
            random.uniform(b[2], b[3]),
            random.uniform(b[4], b[5]),
        ])

    def _nearest(self, q: np.ndarray) -> int:
        dists = [np.linalg.norm(n.pos - q) for n in self._nodes]
        return int(np.argmin(dists))

    def _near(self, q: np.ndarray) -> List[int]:
        n = len(self._nodes)
        r = min(self._radius, self._radius * math.sqrt(math.log(max(n, 1)) / max(n, 1)) * 5)
        return [i for i, node in enumerate(self._nodes) if np.linalg.norm(node.pos - q) < r]

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        d = q_rand - q_near
        dist = np.linalg.norm(d)
        if dist <= self._step:
            return q_rand.copy()
        return q_near + (d / dist) * self._step

    def _extract_path(self, goal_node_idx: int, goal_pos: np.ndarray) -> List[np.ndarray]:
        path = [goal_pos]
        idx = goal_node_idx
        while idx is not None:
            path.append(self._nodes[idx].pos.copy())
            idx = self._nodes[idx].parent
        path.reverse()
        return path


# ──────────────────────────────────────────────────────────────────────────────
# Dynamic Obstacle Avoidance (Velocity Obstacles)
# ──────────────────────────────────────────────────────────────────────────────

class DynamicObstacleAvoider:
    """
    Velocity Obstacle (VO) / ORCA-inspired reactive collision avoidance.
    Modifies the UAV's commanded velocity to avoid moving obstacles.
    """

    def __init__(self, safety_margin: float = 2.5, horizon_s: float = 3.0):
        self._margin  = safety_margin
        self._horizon = horizon_s

    def compute_safe_velocity(
        self,
        uav_pos: np.ndarray,
        uav_vel: np.ndarray,
        desired_vel: np.ndarray,
        dynamic_obstacles: List[dict],
    ) -> np.ndarray:
        """
        Returns modified velocity command that avoids VO cones.
        Falls back to desired_vel if no conflicts.
        """
        if not dynamic_obstacles:
            return desired_vel.copy()

        avoidance = np.zeros(3)
        any_conflict = False

        for obs in dynamic_obstacles:
            obs_pos = np.array(obs["position"])
            obs_vel = np.array(obs.get("velocity", [0, 0, 0]))
            obs_r   = obs.get("radius", 1.0) + self._margin

            rel_pos = obs_pos - uav_pos
            rel_vel = uav_vel - obs_vel  # Velocity of UAV relative to obstacle

            dist = np.linalg.norm(rel_pos)
            if dist > self._horizon * np.linalg.norm(rel_vel) + obs_r:
                continue

            # Time to closest approach
            t_ca = -np.dot(rel_pos, rel_vel) / max(np.dot(rel_vel, rel_vel), 1e-6)
            t_ca = max(0.0, min(t_ca, self._horizon))

            closest = rel_pos + rel_vel * t_ca
            closest_dist = np.linalg.norm(closest)

            if closest_dist < obs_r:
                any_conflict = True
                # Repulsion perpendicular to the VO cone
                perp = rel_pos - normalize_vector(rel_vel) * np.dot(rel_pos, normalize_vector(rel_vel))
                if np.linalg.norm(perp) > 1e-6:
                    avoidance += normalize_vector(perp) * (obs_r - closest_dist) / max(t_ca, 0.1)

        if any_conflict:
            modified = desired_vel + avoidance * 2.0
            # Scale to original speed
            des_speed = np.linalg.norm(desired_vel)
            mod_speed = np.linalg.norm(modified)
            if mod_speed > 1e-6:
                modified = modified / mod_speed * min(des_speed, mod_speed)
            return modified

        return desired_vel.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Mission Planner (High-Level)
# ──────────────────────────────────────────────────────────────────────────────

class MissionPlanner:
    """
    High-level mission manager:
      - Holds a list of mission waypoints
      - Replans if path is blocked
      - Switches between A* and RRT* based on environment
    """

    def __init__(self, config: dict):
        nav_cfg = config.get("navigation", {})
        self._wp_radius = nav_cfg.get("waypoint_radius_m", 2.0)
        self._lookahead = nav_cfg.get("lookahead_distance_m", 5.0)
        self._safety_m  = nav_cfg.get("safety_margin_m", 2.5)

        env_cfg  = config.get("environment", {})
        grid     = env_cfg.get("grid_size", [200, 200, 100])
        bounds   = (0, grid[0], 0, grid[1], 0, grid[2])
        self._occ_map = OccupancyMap3D(bounds, resolution=1.0, safety_margin=self._safety_m)

        self._astar  = AStarPlanner(self._occ_map)
        self._rrt    = RRTStarPlanner(self._occ_map, nav_cfg.get("rrt_star", {}))
        self._dyn_av = DynamicObstacleAvoider(self._safety_m)

        self._mission_wps: List[Waypoint] = []
        self._current_path: Optional[Path] = None
        self._path_wp_idx: int = 0
        self._algo  = nav_cfg.get("algorithm", "hybrid")

    def set_mission(self, waypoints: List[Waypoint]) -> None:
        self._mission_wps = waypoints
        log.info(f"Mission set: {len(waypoints)} waypoints")

    def update_obstacles(self, static_obstacles: List[dict]) -> None:
        self._occ_map.build_from_obstacles(static_obstacles)

    def plan_to_next(self, current_pos: np.ndarray, goal_wp: Waypoint) -> Optional[Path]:
        """Plan a path from current_pos to goal_wp."""
        if self._algo in ("astar", "hybrid"):
            path = self._astar.plan(current_pos, goal_wp.position)
        elif self._algo == "rrt_star":
            path = self._rrt.plan(current_pos, goal_wp.position)
        else:
            path = self._astar.plan(current_pos, goal_wp.position)
            if path is None:
                path = self._rrt.plan(current_pos, goal_wp.position)

        self._current_path = path
        self._path_wp_idx  = 0
        return path

    def get_next_target(self, current_pos: np.ndarray) -> Optional[np.ndarray]:
        """Return the immediate target position from the current path."""
        if self._current_path is None or self._path_wp_idx >= len(self._current_path):
            return None

        target_wp = self._current_path.waypoints[self._path_wp_idx]

        # Advance waypoint index if within radius
        if np.linalg.norm(current_pos - target_wp.position) < self._wp_radius:
            self._path_wp_idx += 1
            if self._path_wp_idx >= len(self._current_path):
                return None
            target_wp = self._current_path.waypoints[self._path_wp_idx]

        return target_wp.position

    def compute_safe_velocity(
        self, pos, vel, desired_vel, dynamic_obstacles
    ) -> np.ndarray:
        return self._dyn_av.compute_safe_velocity(pos, vel, desired_vel, dynamic_obstacles)
