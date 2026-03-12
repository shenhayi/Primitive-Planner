#!/usr/bin/env python3
"""
Primitive-Planner offline scenario generator
===========================================
Features:
  1. Randomly generate a static cylinder obstacle environment
  2. Randomly sample collision-free start and goal points for multiple drones
  3. Use the Primitive-Planner core logic to solve collision-free paths
  4. Export position, velocity, and acceleration at each time step
  5. Save everything to JSON for Isaac Sim replay

Usage examples:
  # Default parameters (5 drones, 40 obstacles)
  conda run -n silent python3 gen_scenario.py

  # Custom parameters
  conda run -n silent python3 gen_scenario.py --n-drones 10 --n-obstacles 60 --map-x 30 --map-y 30 --seed 42

  # Specify the primitive library path and output file
  conda run -n silent python3 gen_scenario.py --primitive-lib /path/to/primitive_library --output my_scene.json

Output JSON schema:
  {
    "metadata": { "n_drones", "dt", "max_vel", ... },
    "obstacles": [ {"type", "x", "y", "z", "radius", "height", "base_z"}, ... ],
    "drones": [
      {
        "id": 0, "start": [x,y,z], "goal": [x,y,z],
        "trajectory": {
          "duration": float,
          "timestamps": [...],
          "positions": [[x,y,z], ...],      # 100 Hz
          "velocities": [[vx,vy,vz], ...],
          "accelerations": [[ax,ay,az], ...]
        }
      }, ...
    ]
  }
"""

from __future__ import annotations

import numpy as np
import json
import os
import math
import argparse
import time
import struct
from typing import Optional
from pathlib import Path

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:
    uniform_filter1d = None

# ================================================================
# Default configuration
# ================================================================
DEFAULT_CONFIG = {
    # Drone parameters
    "n_drones":         5,
    "drone_radius":     0.15,    # m, drone body radius
    "max_vel":          1.0,     # m/s, must match the primitive library
    "flight_height":    0.5,     # m, fixed flight height

    # Map parameters (centered at the origin)
    "map_x":            30.0,    # m, total X size (-map_x/2 ~ +map_x/2)
    "map_y":            30.0,    # m, total Y size
    "map_z":            10.0,    # m, Z height (0 ~ map_z)
    "ground_z":         0.0,     # m

    # Cylinder obstacle parameters (center region only), matching primitive_planner_data_gen
    "n_obstacles":      40,
    "obs_radius_min":   0.4,     # m
    "obs_radius_max":   0.8,     # m
    "obs_height_min":   2.0,     # m
    "obs_height_max":   10.0,    # m
    "min_obs_spacing":  2.0,     # m, minimum cylinder-center spacing
    "obs_inner_ratio":  0.60,    # obstacles are placed in the center region only
    "voxel_size":       0.1,     # m, occupied voxel size for obstacle export/correspondence

    # Start/goal parameters
    "safe_margin":      0.5,     # m, minimum clearance from obstacle surfaces
    "traj_margin_extra": 0.15,   # m, extra margin in trajectory collision checks
    "min_sg_dist":      8.0,     # m, minimum distance between one drone's start and goal
    "min_drone_sep":    2.5,     # m, minimum separation between different drones
    "max_place_tries":  2000,    # maximum random placement retries
    "start_goal_clearance": 1.0, # m, minimum clearance from obstacle voxels for starts/goals
    "periphery_ratio":  0.9,     # outer ring ratio, matching primitive_planner_data_gen

    # Primitive library path (None = auto-detect)
    "primitive_lib_path": None,

    # Planner parameters (must match the primitive library generation)
    "box_x":            6.0,     # m
    "box_y":            6.0,     # m
    "box_z":            6.0,     # m
    "lambda_l":         200.0,   # goal progress cost weight
    "lambda_b":         50.0,    # boundary cost weight

    # Replanning termination conditions
    "replan_dt":        0.2,     # s, replan every 0.2 s
    "goal_thresh":      0.5,     # m, goal reached threshold
    "max_plan_steps":   500,     # maximum replanning iterations
    "swarm_clearence":  0.24,    # m, minimum drone-drone clearance

    # Trajectory post-processing
    "smooth_traj":      False,   # whether to smooth the trajectory
    "smooth_window":    5,       # smoothing window size (odd)

    # Output
    "output_dir":       None,    # output directory, None = auto-generate
    "scenario_name":    None,    # explicit folder name for this scenario
    "scenario_index":   None,    # optional stable index used in folder names
    "random_seed":      None,    # integer or None
}

OBSTACLE_PRESETS = {
    "default": {},
    "dense_tall": {
        "n_obstacles": 80,
        "obs_radius_min": 0.45,
        "obs_radius_max": 0.90,
        "obs_height_min": 4.0,
        "obs_height_max": 10.0,
        "min_obs_spacing": 1.6,
        "obs_inner_ratio": 0.72,
    },
    "swarm_like": {
        "n_obstacles": 70,
        "obs_radius_min": 0.4,
        "obs_radius_max": 0.8,
        "obs_height_min": 2.0,
        "obs_height_max": 10.0,
        "min_obs_spacing": 2.0,
        "obs_inner_ratio": 0.60,
    },
}


def build_scenario_name(cfg: dict, prefix: str = "scenario") -> str:
    parts = [prefix]
    if cfg.get("scenario_index") is not None:
        parts.append(f"{int(cfg['scenario_index']):03d}")
    if cfg.get("random_seed") is not None:
        parts.append(f"seed{int(cfg['random_seed']):04d}")
    parts.append(f"d{int(cfg['n_drones'])}")
    parts.append(f"obs{int(cfg['n_obstacles'])}")
    return "_".join(parts)


def resolve_output_dir(cfg: dict, script_file: str, prefix: str = "scenario") -> tuple[str, str]:
    scenario_name = cfg.get("scenario_name") or build_scenario_name(cfg, prefix=prefix)
    if cfg["output_dir"]:
        out_dir = cfg["output_dir"]
    else:
        script_dir = os.path.dirname(os.path.abspath(script_file))
        out_dir = os.path.join(script_dir, "..", "..", "output", "scenarios", scenario_name)
    return out_dir, scenario_name


def apply_obstacle_preset(cfg: dict, preset: str) -> None:
    """Apply a named obstacle preset in-place."""
    if preset not in OBSTACLE_PRESETS:
        raise ValueError(
            f"Unknown obstacle preset '{preset}'. "
            f"Available presets: {', '.join(sorted(OBSTACLE_PRESETS))}"
        )
    cfg.update(OBSTACLE_PRESETS[preset])


# ================================================================
# Obstacle environment
# ================================================================
class ObstacleEnv:
    """Generate static cylinder obstacles and handle collision checks."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # list of dicts with center, radius, height, and base_z
        self.obstacles: list[dict] = []
        self._voxel_cloud_cache: dict[float, np.ndarray] = {}
        self._centers_xy = np.empty((0, 2), dtype=float)
        self._radii = np.empty((0,), dtype=float)
        self._heights = np.empty((0,), dtype=float)
        self._base_z = np.empty((0,), dtype=float)

    def generate(self) -> list[dict]:
        """Generate random static cylinders in the center region only."""
        cfg = self.cfg
        inner_ratio = cfg.get("obs_inner_ratio", 0.60)
        half_x = cfg["map_x"] / 2 * inner_ratio
        half_y = cfg["map_y"] / 2 * inner_ratio
        placed: list[dict] = []
        ground_z = float(cfg.get("ground_z", 0.0))
        height_max = min(float(cfg["obs_height_max"]), float(cfg["map_z"]) - ground_z)
        if height_max <= 0.0:
            raise ValueError("map_z must be greater than ground_z to place cylinder obstacles")
        height_min = min(float(cfg["obs_height_min"]), height_max)

        for _ in range(cfg["n_obstacles"]):
            for _try in range(10000):
                cx = np.random.uniform(-half_x, half_x)
                cy = np.random.uniform(-half_y, half_y)
                radius = np.random.uniform(cfg["obs_radius_min"], cfg["obs_radius_max"])
                height = np.random.uniform(height_min, height_max)
                candidate = {
                    "type": "cylinder",
                    "x": float(cx),
                    "y": float(cy),
                    "z": float(ground_z + height / 2.0),
                    "base_z": ground_z,
                    "radius": float(radius),
                    "height": float(height),
                }

                # Match the C++ data generator: spacing is enforced between obstacle centers.
                valid = True
                for existing in placed:
                    if self._cylinders_too_close(candidate, existing, cfg["min_obs_spacing"]):
                        valid = False
                        break
                if valid:
                    placed.append(candidate)
                    break
            else:
                print(f"[WARN] only placed {len(placed)} obstacles (target {cfg['n_obstacles']})")
                break

        self.obstacles = placed
        self._voxel_cloud_cache.clear()
        self._refresh_obstacle_arrays()
        return placed

    def _refresh_obstacle_arrays(self) -> None:
        if not self.obstacles:
            self._centers_xy = np.empty((0, 2), dtype=float)
            self._radii = np.empty((0,), dtype=float)
            self._heights = np.empty((0,), dtype=float)
            self._base_z = np.empty((0,), dtype=float)
            return

        self._centers_xy = np.array(
            [[obs["x"], obs["y"]] for obs in self.obstacles],
            dtype=float,
        )
        self._radii = np.array([obs["radius"] for obs in self.obstacles], dtype=float)
        self._heights = np.array([obs["height"] for obs in self.obstacles], dtype=float)
        self._base_z = np.array([obs.get("base_z", obs["z"] - obs["height"] / 2.0) for obs in self.obstacles], dtype=float)

    def _cylinders_too_close(self, a: dict, b: dict, spacing: float) -> bool:
        return math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < spacing

    def _voxelize_cylinder(self, obs: dict, voxel_size: float) -> np.ndarray:
        radius = float(obs["radius"])
        height = float(obs["height"])
        base_z = float(obs.get("base_z", obs["z"] - height / 2.0))

        cells_xy = max(1, int(math.ceil((2.0 * radius) / voxel_size)))
        cells_z = max(1, int(math.ceil(height / voxel_size)))
        xy_offsets = np.arange(-cells_xy // 2, cells_xy // 2 + 1, dtype=float) * voxel_size
        grid_x, grid_y = np.meshgrid(xy_offsets, xy_offsets, indexing="ij")
        mask = grid_x * grid_x + grid_y * grid_y <= radius * radius + 1e-9
        xy = np.column_stack([grid_x[mask], grid_y[mask]])
        if xy.size == 0:
            xy = np.zeros((1, 2), dtype=float)

        z_vals = base_z + (np.arange(cells_z, dtype=float) + 0.5) * voxel_size
        pts = np.empty((xy.shape[0] * cells_z, 3), dtype=float)
        pts[:, 0] = obs["x"] + np.repeat(xy[:, 0], cells_z)
        pts[:, 1] = obs["y"] + np.repeat(xy[:, 1], cells_z)
        pts[:, 2] = np.tile(z_vals, xy.shape[0])
        return pts

    def voxel_cloud(self, voxel_size: float | None = None) -> np.ndarray:
        voxel_size = max(1e-3, float(voxel_size or self.cfg.get("voxel_size", 0.1)))
        cache_key = round(voxel_size, 6)
        cached = self._voxel_cloud_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.obstacles:
            cloud = np.empty((0, 3), dtype=float)
        else:
            clouds = [self._voxelize_cylinder(obs, voxel_size) for obs in self.obstacles]
            cloud = np.vstack(clouds) if clouds else np.empty((0, 3), dtype=float)

        self._voxel_cloud_cache[cache_key] = cloud
        return cloud

    def export_global_cloud(self, voxel_size: float | None = None) -> list[list[float]]:
        return self.voxel_cloud(voxel_size).tolist()

    def _points_in_obstacles(self, pts_world: np.ndarray, margin: float) -> np.ndarray:
        if len(pts_world) == 0 or len(self.obstacles) == 0:
            return np.zeros((len(pts_world),), dtype=bool)

        dx = pts_world[:, None, 0] - self._centers_xy[None, :, 0]
        dy = pts_world[:, None, 1] - self._centers_xy[None, :, 1]
        inside_xy = dx * dx + dy * dy < np.square(self._radii[None, :] + margin)
        inside_z = (
            (pts_world[:, None, 2] > self._base_z[None, :] - margin)
            & (pts_world[:, None, 2] < self._base_z[None, :] + self._heights[None, :] + margin)
        )
        return np.any(inside_xy & inside_z, axis=1)

    def is_point_safe(self, x: float, y: float, z: float, clearance: float | None = None) -> bool:
        """Check whether a single point is safe and inside the map."""
        cfg = self.cfg
        margin = (
            float(clearance)
            if clearance is not None
            else cfg["drone_radius"] + cfg["safe_margin"]
        )
        half_x = cfg["map_x"] / 2
        half_y = cfg["map_y"] / 2
        if abs(x) > half_x - margin or abs(y) > half_y - margin:
            return False
        z_margin = max(0.1, cfg["drone_radius"])
        ground_z = cfg.get("ground_z", 0.0)
        if z < ground_z + z_margin or z > cfg["map_z"] - z_margin:
            return False
        return not bool(self._points_in_obstacles(np.array([[x, y, z]], dtype=float), margin)[0])

    def is_traj_safe(self, pts_world: np.ndarray) -> bool:
        """
        Batch-check whether all trajectory points are collision-free.
        pts_world: (N, 3) ndarray in world coordinates
        """
        cfg = self.cfg
        margin = cfg["drone_radius"] + cfg.get("traj_margin_extra", 0.05)
        xs, ys, zs = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]

        half_x = cfg["map_x"] / 2
        half_y = cfg["map_y"] / 2
        if np.any(np.abs(xs) > half_x - margin) or np.any(np.abs(ys) > half_y - margin):
            return False
        ground_z = cfg.get("ground_z", 0.0)
        if np.any(zs <= ground_z + margin) or np.any(zs >= cfg["map_z"] - margin):
            return False

        return not np.any(self._points_in_obstacles(pts_world, margin))


# ================================================================
# Primitive library loading
# ================================================================
class PrimitiveLibrary:
    """Load and manage the Primitive-Planner motion primitive library."""

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.path_count: int = 0
        self.path_ends: list[np.ndarray] = []    # [path_id] -> (3,)
        self.path_all:  dict[int, np.ndarray] = {}  # path_id -> (N,3), body frame
        self.path_length_max: float = 0.0
        # (vel_id, path_id) -> {'pos': (N,3), 'vel': (N,3), 'acc': (N,3), 'duration': float}
        self.trajectories: dict[tuple, dict] = {}
        self.infeasible: dict[int, set] = {}     # vel_id -> set of infeasible path IDs
        self.agent_correspondence: dict[int, dict[int, np.ndarray]] = {}
        self.obs_correspondence: dict[int, np.ndarray] = {}
        self.available_vel_ids: list[int] = []
        self.max_supported_vel: float = 0.0

    def load(self, max_vel: float = 1.0):
        """Load all required data from disk."""
        t0 = time.time()
        print("  Loading path_end.ply ...")
        self._load_path_ends()
        print(f"    {self.path_count} paths found")

        print("  Loading path_all.ply ...")
        self._load_path_all()

        self.available_vel_ids = self._discover_vel_ids()
        if not self.available_vel_ids:
            raise RuntimeError(f"No trajectory buckets found under {self.lib_path}/trajectory")

        self.max_supported_vel = self.available_vel_ids[-1] / 10.0
        if max_vel > self.max_supported_vel + 1e-6:
            raise ValueError(
                "Requested max_vel="
                f"{max_vel:.2f} m/s, but primitive library only supports up to "
                f"{self.max_supported_vel:.2f} m/s. Regenerate the library or lower --max-vel."
            )

        max_vel_id = int(round(max_vel * 10))
        print(f"  Loading infeasible path lists (vel_id 0~{max_vel_id})...")
        for vid in range(max_vel_id + 1):
            self._load_infeasible(vid)

        print(f"  Loading trajectory files (vel_id 0~{max_vel_id})...")
        for vid in range(max_vel_id + 1):
            self._load_trajectories(vid)

        elapsed = time.time() - t0
        print(f"  Loading complete in {elapsed:.1f}s, "
              f"cached {len(self.trajectories)} trajectories")

    def _discover_vel_ids(self) -> list[int]:
        traj_root = os.path.join(self.lib_path, "trajectory")
        if not os.path.isdir(traj_root):
            return []

        vel_ids: list[int] = []
        for entry in os.listdir(traj_root):
            full_path = os.path.join(traj_root, entry)
            if not os.path.isdir(full_path):
                continue
            try:
                vel_ids.append(int(entry))
            except ValueError:
                continue
        vel_ids.sort()
        return vel_ids

    # ------------------------------------------------------------------
    def _load_path_ends(self):
        fname = os.path.join(self.lib_path, "obs_correspondence", "path_end.ply")
        ends = []
        with open(fname, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    ends.append(np.array([float(parts[0]),
                                          float(parts[1]),
                                          float(parts[2])]))
        self.path_ends = ends
        self.path_count = len(ends)

    def _load_path_all(self):
        fname = os.path.join(self.lib_path, "obs_correspondence", "path_all.ply")
        path_all: dict[int, list] = {}
        length_acc: dict[int, float] = {}

        with open(fname, "r") as f:
            lines = f.readlines()

        total = int(lines[0].strip())
        prev_pid = -1
        prev_pos = None

        for i in range(total):
            parts = lines[1 + i].strip().split()
            pos = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            pid = int(parts[3])

            if pid not in path_all:
                path_all[pid] = []
                length_acc[pid] = 0.0

            # Store one point every 0.1 m (downsampled)
            if (len(path_all[pid]) == 0 or
                    np.linalg.norm(pos - path_all[pid][-1]) > 0.1):
                path_all[pid].append(pos)

            if pid == prev_pid and prev_pos is not None:
                length_acc[pid] += np.linalg.norm(pos - prev_pos)
                if length_acc[pid] > self.path_length_max:
                    self.path_length_max = length_acc[pid]

            prev_pos = pos
            prev_pid = pid

        self.path_all = {pid: np.array(pts) for pid, pts in path_all.items()}

    def _load_infeasible(self, vel_id: int):
        fname = os.path.join(self.lib_path, "trajectory", str(vel_id),
                             f"{vel_id}_infeasible_id.ply")
        inf_set: set = set()
        if os.path.exists(fname):
            with open(fname, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        inf_set.add(int(line))
        self.infeasible[vel_id] = inf_set

    def _load_trajectories(self, vel_id: int):
        traj_dir = os.path.join(self.lib_path, "trajectory", str(vel_id))
        if not os.path.isdir(traj_dir):
            return
        inf_ids = self.infeasible.get(vel_id, set())

        for path_id in range(self.path_count):
            if path_id in inf_ids:
                continue
            fname = os.path.join(traj_dir, f"{path_id}_trajectory.ply")
            if not os.path.exists(fname):
                continue
            try:
                with open(fname, "r") as f:
                    lines = f.readlines()
                n_pts     = int(lines[0].strip())
                duration  = float(lines[1].strip())
                pos_arr   = np.zeros((n_pts, 3))
                vel_arr   = np.zeros((n_pts, 3))
                acc_arr   = np.zeros((n_pts, 3))
                for i in range(n_pts):
                    v = lines[2 + i].strip().split()
                    pos_arr[i] = [float(v[0]), float(v[1]), float(v[2])]
                    vel_arr[i] = [float(v[3]), float(v[4]), float(v[5])]
                    acc_arr[i] = [float(v[6]), float(v[7]), float(v[8])]
                self.trajectories[(vel_id, path_id)] = {
                    "pos": pos_arr, "vel": vel_arr,
                    "acc": acc_arr, "duration": duration,
                }
            except Exception as exc:
                print(f"  [WARN] skipping corrupt trajectory file {fname}: {exc}")

    def load_agent_correspondence(self, vel_id: int, voxel_num_all: int) -> dict[int, np.ndarray]:
        """Load precomputed swarm path correspondences for one velocity bucket."""
        if vel_id in self.agent_correspondence:
            return self.agent_correspondence[vel_id]

        fname = os.path.join(
            self.lib_path, "agent_correspondence", f"{vel_id}_correspondence.txt"
        )
        corr: dict[int, np.ndarray] = {}
        if not os.path.exists(fname):
            self.agent_correspondence[vel_id] = corr
            return corr

        data = np.fromfile(fname, dtype=np.int32)
        idx = 0
        voxel_count = 0
        data_len = len(data)
        while idx < data_len and voxel_count < voxel_num_all:
            voxel_id = int(data[idx])
            idx += 1
            start = idx
            while idx < data_len and data[idx] != -1:
                idx += 1
            if idx > start:
                corr[voxel_id] = data[start:idx].copy()
            idx += 1
            voxel_count += 1

        self.agent_correspondence[vel_id] = corr
        return corr

    def load_obs_correspondence(self, voxel_num_all: int) -> dict[int, np.ndarray]:
        """Load precomputed obstacle-path correspondences."""
        if self.obs_correspondence:
            return self.obs_correspondence

        fname = os.path.join(self.lib_path, "obs_correspondence", "obs_correspondence.txt")
        corr: dict[int, np.ndarray] = {}
        if not os.path.exists(fname):
            self.obs_correspondence = corr
            return corr

        data = np.fromfile(fname, dtype=np.int32)
        idx = 0
        voxel_count = 0
        data_len = len(data)
        while idx < data_len and voxel_count < voxel_num_all:
            voxel_id = int(data[idx])
            idx += 1
            start = idx
            while idx < data_len and data[idx] != -1:
                idx += 1
            if idx > start:
                corr[voxel_id] = data[start:idx].copy()
            idx += 1
            voxel_count += 1

        self.obs_correspondence = corr
        return corr


# ================================================================
# Offline primitive planner
# ================================================================
class OfflinePlanner:
    """
    Offline greedy primitive selector.
    Reproduces the core Primitive-Planner logic without ROS dependencies.
    """

    def __init__(self, lib: PrimitiveLibrary, env: ObstacleEnv, cfg: dict):
        self.lib = lib
        self.env = env
        self.cfg = cfg

    # ------------------------------------------------------------------
    def _build_rwv(self, velocity: np.ndarray,
                   fallback_yaw: float = 0.0) -> np.ndarray:
        """
        Build the rotation matrix R_WV (velocity frame -> world frame).
        This matches the construction in pp_replan_fsm.cpp:
          xV = velocity.normalized()  (or heading direction if near zero)
          yV = cross(xV, [0,0,-1]).normalized()
          zV = cross(xV, yV)
        """
        if np.linalg.norm(velocity) > 0.05:
            xV = velocity / np.linalg.norm(velocity)
        else:
            xV = np.array([math.cos(fallback_yaw),
                           math.sin(fallback_yaw), 0.0])

        yV = np.cross(xV, np.array([0.0, 0.0, -1.0]))
        norm_yV = np.linalg.norm(yV)
        if norm_yV < 1e-6:
            yV = np.cross(xV, np.array([1.0, 0.0, 0.0]))
            norm_yV = np.linalg.norm(yV)
        yV /= norm_yV
        zV = np.cross(xV, yV)

        RWV = np.column_stack([xV, yV, zV])
        return RWV

    def _score_path(self, path_id: int, start_pos: np.ndarray,
                    RWV: np.ndarray, goal: np.ndarray) -> float:
        """Score one path. Lower is better, matching planner_manager.cpp::scorePaths."""
        cfg = self.cfg
        end_body  = self.lib.path_ends[path_id]
        end_world = start_pos + RWV @ end_body

        # Goal progress cost (matching the C++ logic)
        heading = RWV[:, 0]
        if (np.linalg.norm(start_pos - goal) > self.lib.path_length_max or
                heading.dot(goal - start_pos) <= 0):
            goal_dist = (np.linalg.norm(end_world - goal)
                         - np.linalg.norm(start_pos - goal))
        else:
            # Near the goal, use the minimum distance from any trajectory point to the goal
            pts_body  = self.lib.path_all[path_id]
            pts_world = start_pos + (RWV @ pts_body.T).T
            goal_dist = float(np.min(np.linalg.norm(pts_world - goal, axis=1)))

        # Boundary cost
        hx = cfg["map_x"] / 2
        hy = cfg["map_y"] / 2
        hz = cfg["map_z"]
        bound_cost = 10.0 if (
            end_world[0] < -hx or end_world[0] > hx or
            end_world[1] < -hy or end_world[1] > hy or
            end_world[2] < 0   or end_world[2] > hz
        ) else 0.0

        return cfg["lambda_l"] * goal_dist + cfg["lambda_b"] * bound_cost

    def _interp_pos_at_t(self, ts: np.ndarray, pos: np.ndarray, t: float) -> np.ndarray:
        """Interpolate position at time t."""
        if t <= ts[0]:
            return pos[0].copy()
        if t >= ts[-1]:
            return pos[-1].copy()
        idx = np.searchsorted(ts, t, side="right") - 1
        if idx < 0:
            return pos[0].copy()
        if idx >= len(ts) - 1:
            return pos[-1].copy()
        s = (t - ts[idx]) / (ts[idx + 1] - ts[idx])
        return pos[idx] * (1 - s) + pos[idx + 1] * s

    def _traj_respects_vertical_bounds(self, pts_world: np.ndarray) -> bool:
        """Reject trajectories that dip below the ground or above the ceiling."""
        if len(pts_world) == 0:
            return True
        cfg = self.cfg
        margin = cfg["drone_radius"] + cfg.get("traj_margin_extra", 0.05)
        zs = pts_world[:, 2]
        return bool(np.all(zs > margin) and np.all(zs < cfg["map_z"] - margin))

    def _traj_vs_swarm_safe(
        self,
        our_ts: np.ndarray,
        our_pos: np.ndarray,
        other_trajs: list[dict],
        clearence: float,
    ) -> bool:
        """Check whether this segment conflicts with already planned drone trajectories."""
        for ot in other_trajs:
            ots = np.array(ot["timestamps"])
            opos = np.array(ot["positions"])
            for i in range(len(our_ts)):
                op = self._interp_pos_at_t(ots, opos, our_ts[i])
                if np.linalg.norm(our_pos[i] - op) < clearence:
                    return False
        return True

    def _traj_local_ts(self, n_pts: int, duration: float) -> np.ndarray:
        """Return local timestamps for a sampled primitive trajectory."""
        if n_pts <= 1:
            return np.array([0.0])
        return np.linspace(0.0, duration, n_pts, endpoint=True)

    def _blocked_paths_from_swarm(
        self,
        start_pos: np.ndarray,
        start_vel: np.ndarray,
        start_time: float,
        RWV: np.ndarray,
        other_trajs: list[dict],
    ) -> set[int]:
        """
        Mirror planner_manager.cpp::labelAgentCollisionPaths using the
        precomputed agent correspondence tables from swarm_path.py.
        """
        if not other_trajs:
            return set()

        cfg = self.cfg
        vel_id = min(
            int(round(np.linalg.norm(start_vel) * 10)),
            int(round(cfg["max_vel"] * 10)),
        )
        voxel_size = cfg.get("voxel_size", 0.1)
        voxel_x = cfg["box_x"]
        voxel_y = cfg["box_y"] / 2.0
        voxel_z = cfg["box_z"] / 2.0
        voxel_num_x = int(cfg["box_x"] / voxel_size)
        voxel_num_y = int(cfg["box_y"] / voxel_size)
        voxel_num_z = int(cfg["box_z"] / voxel_size)
        voxel_num_all = voxel_num_x * voxel_num_y * voxel_num_z
        stride = max(1, int(math.floor(voxel_size / cfg["max_vel"] * 100 / 5)))
        rot_vw = RWV.T

        corr = self.lib.load_agent_correspondence(vel_id, voxel_num_all)
        if not corr:
            return set()

        blocked: set[int] = set()
        for ot in other_trajs:
            ots = np.array(ot["timestamps"], dtype=float)
            opos = np.array(ot["positions"], dtype=float)
            for j in range(0, len(opos), stride):
                pos_v = rot_vw @ (opos[j] - start_pos)
                if not (
                    (1e-4 <= pos_v[0] <= voxel_x - 1e-4)
                    and (-voxel_y + 1e-4 <= pos_v[1] <= voxel_y - 1e-4)
                    and (-voxel_z + 1e-4 <= pos_v[2] <= voxel_z - 1e-4)
                ):
                    continue

                ind_x = math.floor((voxel_x - pos_v[0]) / voxel_size)
                ind_y = math.floor((voxel_y - pos_v[1]) / voxel_size)
                ind_z = math.floor((voxel_z - pos_v[2]) / voxel_size)
                voxel_id = voxel_num_y * voxel_num_z * ind_x + voxel_num_z * ind_y + ind_z

                entries = corr.get(voxel_id)
                if entries is None:
                    continue

                other_cur_time = float(ots[j])
                occ_num = len(entries) // 3
                for k in range(occ_num):
                    path_id = int(entries[3 * k])
                    t_from = start_time + float(entries[3 * k + 1]) / 1000.0
                    t_to = start_time + float(entries[3 * k + 2]) / 1000.0
                    if t_from < other_cur_time < t_to:
                        blocked.add(path_id)

        return blocked

    def _blocked_paths_from_obstacles(
        self,
        start_pos: np.ndarray,
        RWV: np.ndarray,
    ) -> set[int]:
        """Mirror planner_manager.cpp::labelObsCollisionPaths using obstacle correspondences."""
        cfg = self.cfg
        voxel_size = cfg.get("voxel_size", 0.1)
        voxel_x = cfg["box_x"]
        voxel_y = cfg["box_y"] / 2.0
        voxel_z = cfg["box_z"] / 2.0
        voxel_num_x = int(cfg["box_x"] / voxel_size)
        voxel_num_y = int(cfg["box_y"] / voxel_size)
        voxel_num_z = int(cfg["box_z"] / voxel_size)
        voxel_num_all = voxel_num_x * voxel_num_y * voxel_num_z
        rot_vw = RWV.T

        corr = self.lib.load_obs_correspondence(voxel_num_all)
        if not corr:
            return set()

        cloud_world = self.env.voxel_cloud(voxel_size)
        if len(cloud_world) == 0:
            return set()

        pos_v = (cloud_world - start_pos) @ rot_vw.T
        inside = (
            (pos_v[:, 0] >= 1e-4)
            & (pos_v[:, 0] <= voxel_x - 1e-4)
            & (pos_v[:, 1] >= -voxel_y + 1e-4)
            & (pos_v[:, 1] <= voxel_y - 1e-4)
            & (pos_v[:, 2] >= -voxel_z + 1e-4)
            & (pos_v[:, 2] <= voxel_z - 1e-4)
        )
        if not np.any(inside):
            return set()

        pos_v = pos_v[inside]
        ind_x = np.floor((voxel_x - pos_v[:, 0]) / voxel_size).astype(int)
        ind_y = np.floor((voxel_y - pos_v[:, 1]) / voxel_size).astype(int)
        ind_z = np.floor((voxel_z - pos_v[:, 2]) / voxel_size).astype(int)
        voxel_ids = np.unique(voxel_num_y * voxel_num_z * ind_x + voxel_num_z * ind_y + ind_z)

        blocked: set[int] = set()
        for voxel_id in voxel_ids:
            entries = corr.get(int(voxel_id))
            if entries is None:
                continue
            blocked.update(int(path_id) for path_id in entries)

        return blocked

    def plan(self, start: list, goal: list,
             other_trajs: list[dict] | None = None) -> dict | None:
        """
        Plan from start to goal.
        Return a trajectory dictionary, or None if planning fails.
        """
        cfg = self.cfg
        max_vel = cfg["max_vel"]
        swarm_clearence = cfg.get("swarm_clearence", DEFAULT_CONFIG["swarm_clearence"])
        other_trajs = other_trajs or []

        cur_pos = np.array(start, dtype=float)
        cur_vel = np.zeros(3)
        goal_arr = np.array(goal, dtype=float)

        # Initial heading points toward the goal
        d0 = goal_arr - cur_pos
        cur_yaw = math.atan2(d0[1], d0[0])

        all_pos, all_vel, all_acc, all_ts = [], [], [], []
        t_elapsed = 0.0

        for step in range(cfg["max_plan_steps"]):
            dist_to_goal = np.linalg.norm(cur_pos - goal_arr)
            if dist_to_goal < cfg["goal_thresh"]:
                # Close to the goal: append a straight approach segment
                approach_dur = max(0.05, dist_to_goal / max_vel)
                n_approach = max(2, int(approach_dur * 100))
                s_vals = np.linspace(0, 1, n_approach, endpoint=True)
                pos_app = np.outer(1 - s_vals, cur_pos) + np.outer(s_vals, goal_arr)
                vel_app = np.tile((goal_arr - cur_pos) / approach_dur, (n_approach, 1))
                acc_app = np.zeros_like(pos_app)
                ts_app = t_elapsed + np.linspace(0, approach_dur, n_approach, endpoint=True)
                all_pos.append(pos_app)
                all_vel.append(vel_app)
                all_acc.append(acc_app)
                all_ts.append(ts_app)
                t_elapsed += approach_dur
                cur_pos = goal_arr.copy()
                cur_vel = np.zeros(3)
                print(f"      [ok] reached goal, step={step}, t={t_elapsed:.2f}s")
                break

            RWV = self._build_rwv(cur_vel, cur_yaw)
            blocked_paths = self._blocked_paths_from_obstacles(cur_pos, RWV)
            blocked_paths.update(self._blocked_paths_from_swarm(
                cur_pos, cur_vel, t_elapsed, RWV, other_trajs
            ))

            # vel_id is based on the current speed magnitude
            vel_id = min(int(round(np.linalg.norm(cur_vel) * 10)),
                         int(round(max_vel * 10)))

            # Collision filtering + scoring
            scored: list[tuple] = []
            for pid in range(self.lib.path_count):
                if pid in self.lib.infeasible.get(vel_id, set()):
                    continue
                if pid in blocked_paths:
                    continue
                traj = self.lib.trajectories.get((vel_id, pid))
                if traj is None:
                    continue
                candidate_pos_w = cur_pos + (RWV @ traj["pos"].T).T
                if not self._traj_respects_vertical_bounds(candidate_pos_w):
                    continue

                score = self._score_path(pid, cur_pos, RWV, goal_arr)
                scored.append((score, pid))

            if not scored:
                print(f"      [WARN] step={step}: no feasible path")
                break

            scored.sort(key=lambda x: x[0])

            best_pid = None
            traj = None
            pos_w = vel_w = acc_w = None
            for _, candidate_pid in scored:
                candidate_traj = self.lib.trajectories[(vel_id, candidate_pid)]
                best_pid = candidate_pid
                traj = candidate_traj
                pos_w = cur_pos + (RWV @ candidate_traj["pos"].T).T
                if not self._traj_respects_vertical_bounds(pos_w):
                    best_pid = None
                    traj = None
                    continue
                vel_w = (RWV @ candidate_traj["vel"].T).T
                acc_w = (RWV @ candidate_traj["acc"].T).T
                break

            if best_pid is None or traj is None:
                print(f"      [WARN] step={step}: all candidate primitives failed exact swarm validation")
                break

            # Receding horizon: execute only the first replan_dt seconds
            replan_dt = cfg.get("replan_dt", 0.2)
            local_ts = self._traj_local_ts(len(pos_w), traj["duration"])
            exec_mask = local_ts < replan_dt
            if not np.any(exec_mask):
                exec_mask[0] = True

            pos_exec = pos_w[exec_mask]
            vel_exec = vel_w[exec_mask]
            acc_exec = acc_w[exec_mask]
            local_ts_exec = local_ts[exec_mask]
            ts_exec = t_elapsed + local_ts_exec
            t_exec = (float(local_ts_exec[-1])
                      if len(local_ts_exec) > 1
                      else min(replan_dt, traj["duration"]))

            all_pos.append(pos_exec)
            all_vel.append(vel_exec)
            all_acc.append(acc_exec)
            all_ts.append(ts_exec)

            # Update the state to the end of the executed segment for the next replan
            cur_pos = pos_exec[-1].copy()
            cur_vel = vel_exec[-1].copy()
            t_elapsed += t_exec
            if np.linalg.norm(cur_vel[:2]) > 0.05:
                cur_yaw = math.atan2(cur_vel[1], cur_vel[0])

        else:
            print(f"      [WARN] reached maximum planning steps {cfg['max_plan_steps']}")

        if not all_pos:
            return None

        full_pos = np.vstack(all_pos)
        full_vel = np.vstack(all_vel)
        full_acc = np.vstack(all_acc)
        full_ts  = np.concatenate(all_ts)

        # Optional smoothing (reduces jaggedness from primitive concatenation)
        if cfg.get("smooth_traj", False):
            if uniform_filter1d is None:
                print("      [WARN] scipy is not installed, skipping trajectory smoothing")
            else:
                w = max(3, int(cfg.get("smooth_window", 5)) | 1)  # odd
                full_pos = np.column_stack([
                    uniform_filter1d(full_pos[:, 0], size=w, mode="nearest"),
                    uniform_filter1d(full_pos[:, 1], size=w, mode="nearest"),
                    uniform_filter1d(full_pos[:, 2], size=w, mode="nearest"),
                ])
                dt_val = float(np.median(np.diff(full_ts)))
                full_vel = np.gradient(full_pos, dt_val, axis=0)
                full_acc = np.gradient(full_vel, dt_val, axis=0)
        return {
            "duration":      float(full_ts[-1]),
            "dt":            0.01,
            "timestamps":    full_ts.tolist(),
            "positions":     full_pos.tolist(),
            "velocities":    full_vel.tolist(),
            "accelerations": full_acc.tolist(),
        }


# ================================================================
# Utility functions
# ================================================================
def find_primitive_lib() -> str | None:
    """Auto-detect the primitive library path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "..", "planner",
                     "plan_manage", "primitive_library"),
        os.path.join(script_dir, "primitive_library"),
        os.path.join(script_dir, "..", "..", "primitive_library"),
    ]
    for c in candidates:
        c = os.path.realpath(c)
        if os.path.isdir(os.path.join(c, "obs_correspondence")):
            return c
    return None


def generate_starts_goals(env: ObstacleEnv,
                          cfg: dict) -> tuple[list, list]:
    """
    Place starts and goals on the outer ring, outside the obstacle region,
    with collision-free spacing between drones.
    """
    ground_z = float(cfg.get("ground_z", 0.0))
    z = max(float(cfg["flight_height"]), ground_z + 0.2)
    map_radius = min(cfg["map_x"], cfg["map_y"]) / 2
    inner_ratio = cfg.get("obs_inner_ratio", 0.60)
    periphery_ratio = cfg.get("periphery_ratio", 0.9)
    periphery_radius = min(
        max(map_radius * periphery_ratio, map_radius * (inner_ratio + 0.35)),
        map_radius * 0.92,
    )

    n = cfg["n_drones"]
    starts, goals = [], []
    max_tries = cfg.get("max_place_tries", 2000)
    min_sep = cfg["min_drone_sep"]
    min_sg_dist = cfg["min_sg_dist"]
    obstacle_clearance = max(
        cfg["drone_radius"] + cfg["safe_margin"],
        float(cfg.get("start_goal_clearance", 1.0)),
    )

    chord = 2.0 * periphery_radius * math.sin(math.pi / max(n, 1))
    if n > 1 and chord < min_sep:
        raise RuntimeError(
            f"current map with min_drone_sep={min_sep} cannot fit ring starts/goals for {n} drones"
        )

    chosen_angles: list[float] = []
    if n == 1:
        chosen_angles = [float(np.random.uniform(0.0, 2.0 * np.pi))]

    if n > 1:
        for _ in range(max_tries):
            angles = np.sort(np.random.uniform(0.0, 2.0 * np.pi, n))
            valid = True
            for i in range(n):
                delta = (angles[(i + 1) % n] - angles[i]) % (2.0 * np.pi)
                sep = 2.0 * periphery_radius * math.sin(delta / 2.0)
                if sep < min_sep:
                    valid = False
                    break
            if valid:
                chosen_angles = angles.tolist()
                break
        else:
            raise RuntimeError("could not find a start/goal angle distribution satisfying drone spacing constraints")

    for i, angle in enumerate(chosen_angles):
        found = False
        for delta in np.linspace(-0.35, 0.35, 15):
            sa = angle + float(delta)
            sx = math.cos(sa) * periphery_radius
            sy = math.sin(sa) * periphery_radius
            gx = -sx
            gy = -sy

            start = [float(sx), float(sy), z]
            goal = [float(gx), float(gy), z]

            if np.linalg.norm(np.array(goal) - np.array(start)) < min_sg_dist:
                continue
            if not env.is_point_safe(start[0], start[1], start[2], clearance=obstacle_clearance):
                continue
            if not env.is_point_safe(goal[0], goal[1], goal[2], clearance=obstacle_clearance):
                continue
            if any(np.linalg.norm(np.array(start) - np.array(s)) < min_sep for s in starts):
                continue
            if any(np.linalg.norm(np.array(goal) - np.array(g)) < min_sep for g in goals):
                continue

            starts.append(start)
            goals.append(goal)
            found = True
            break

        if not found:
            raise RuntimeError(f"could not find valid start/goal points for drone {i}")

    return starts, goals


def validate_scenario(output: dict, cfg: dict) -> list[str]:
    """Validate exported trajectories against static obstacles and swarm clearance."""
    issues: list[str] = []
    drones = [d for d in output["drones"] if d.get("trajectory") is not None]
    obstacles = output.get("obstacles", [])
    drone_radius = cfg["drone_radius"]
    swarm_clearence = cfg["swarm_clearence"]

    def interp(ts: np.ndarray, pos: np.ndarray, t: float) -> np.ndarray:
        if t <= ts[0]:
            return pos[0]
        if t >= ts[-1]:
            return pos[-1]
        idx = np.searchsorted(ts, t, side="right") - 1
        alpha = (t - ts[idx]) / (ts[idx + 1] - ts[idx])
        return pos[idx] * (1 - alpha) + pos[idx + 1] * alpha

    for drone in drones:
        pts = np.array(drone["trajectory"]["positions"], dtype=float)
        for obs_idx, obs in enumerate(obstacles):
            if obs.get("type") == "cylinder":
                base_z = float(obs.get("base_z", obs["z"] - obs["height"] / 2.0))
                d_xy = np.hypot(pts[:, 0] - obs["x"], pts[:, 1] - obs["y"])
                inside_xy = d_xy < obs["radius"] + drone_radius
                inside_z = (
                    (pts[:, 2] > base_z - drone_radius)
                    & (pts[:, 2] < base_z + obs["height"] + drone_radius)
                )
                if np.any(inside_xy & inside_z):
                    issues.append(
                        f"drone {drone['id']} intersects cylinder obstacle {obs_idx}"
                    )
                    break
            else:
                c = math.cos(obs["yaw"])
                s = math.sin(obs["yaw"])
                dx = pts[:, 0] - obs["x"]
                dy = pts[:, 1] - obs["y"]
                local_x = c * dx + s * dy
                local_y = -s * dx + c * dy
                inside_x = np.abs(local_x) < obs["size_x"] / 2.0 + drone_radius
                inside_y = np.abs(local_y) < obs["size_y"] / 2.0 + drone_radius
                inside_z = np.abs(pts[:, 2] - obs["z"]) < obs["size_z"] / 2.0 + drone_radius
                if np.any(inside_x & inside_y & inside_z):
                    issues.append(
                        f"drone {drone['id']} intersects box obstacle {obs_idx}"
                    )
                    break

    for i in range(len(drones)):
        ti = np.array(drones[i]["trajectory"]["timestamps"], dtype=float)
        pi = np.array(drones[i]["trajectory"]["positions"], dtype=float)
        for j in range(i + 1, len(drones)):
            tj = np.array(drones[j]["trajectory"]["timestamps"], dtype=float)
            pj = np.array(drones[j]["trajectory"]["positions"], dtype=float)
            t0 = max(ti[0], tj[0])
            t1 = min(ti[-1], tj[-1])
            if t1 <= t0:
                continue
            for t in np.arange(t0, t1, 0.01):
                dist = np.linalg.norm(interp(ti, pi, t) - interp(tj, pj, t))
                if dist < swarm_clearence:
                    issues.append(
                        f"drones {drones[i]['id']} and {drones[j]['id']} conflict at t={t:.2f}s: dist={dist:.3f}m"
                    )
                    break

    return issues


# ================================================================
# Main entry point
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Primitive-Planner offline scenario generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Drones
    p.add_argument("--n-drones",    type=int,   default=DEFAULT_CONFIG["n_drones"],
                   help="number of drones")
    p.add_argument("--drone-radius",type=float, default=DEFAULT_CONFIG["drone_radius"],
                   help="drone body radius (m)")
    p.add_argument("--max-vel",     type=float, default=DEFAULT_CONFIG["max_vel"],
                   help="maximum velocity (m/s), must match the primitive library")
    p.add_argument("--flight-height",type=float,default=DEFAULT_CONFIG["flight_height"],
                   help="fixed flight height (m)")

    # Map
    p.add_argument("--map-x",       type=float, default=DEFAULT_CONFIG["map_x"],
                   help="total map X size (m)")
    p.add_argument("--map-y",       type=float, default=DEFAULT_CONFIG["map_y"],
                   help="total map Y size (m)")
    p.add_argument("--map-z",       type=float, default=DEFAULT_CONFIG["map_z"],
                   help="map Z height (m)")

    # Obstacles
    p.add_argument("--obstacle-preset", type=str, default="default",
                   choices=sorted(OBSTACLE_PRESETS.keys()),
                   help="named obstacle preset for denser/taller scenes")
    p.add_argument("--n-obstacles", type=int,   default=DEFAULT_CONFIG["n_obstacles"],
                   help="number of obstacles")
    p.add_argument("--obs-r-min",   type=float, default=DEFAULT_CONFIG["obs_radius_min"],
                   help="minimum obstacle cylinder radius (m)")
    p.add_argument("--obs-r-max",   type=float, default=DEFAULT_CONFIG["obs_radius_max"],
                   help="maximum obstacle cylinder radius (m)")
    p.add_argument("--obs-h-min",   type=float, default=DEFAULT_CONFIG["obs_height_min"],
                   help="minimum obstacle height (m)")
    p.add_argument("--obs-h-max",   type=float, default=DEFAULT_CONFIG["obs_height_max"],
                   help="maximum obstacle height (m)")
    p.add_argument("--min-obs-spacing", type=float, default=DEFAULT_CONFIG["min_obs_spacing"],
                   help="minimum center-to-center spacing between obstacle cylinders (m)")
    p.add_argument("--obs-inner-ratio", type=float, default=DEFAULT_CONFIG["obs_inner_ratio"],
                   help="fraction of the map used for obstacle placement near the center")
    p.add_argument("--voxel-size", type=float, default=DEFAULT_CONFIG["voxel_size"],
                   help="occupied voxel size used to approximate cylinders and export the global cloud (m)")

    # Paths
    p.add_argument("--safe-margin", type=float, default=DEFAULT_CONFIG["safe_margin"],
                   help="minimum clearance from starts/goals to obstacles (m)")
    p.add_argument("--start-goal-clearance", type=float, default=DEFAULT_CONFIG["start_goal_clearance"],
                   help="minimum obstacle clearance for sampled starts/goals, matching the C++ data generator (m)")
    p.add_argument("--min-drone-sep", type=float, default=DEFAULT_CONFIG["min_drone_sep"],
                   help="minimum separation between different drones' starts/goals (m)")
    p.add_argument("--min-sg-dist", type=float, default=DEFAULT_CONFIG["min_sg_dist"],
                   help="minimum straight-line distance between one drone's start and goal (m)")
    p.add_argument("--periphery-ratio", type=float, default=DEFAULT_CONFIG["periphery_ratio"],
                   help="outer-ring ratio used when sampling starts/goals")
    p.add_argument("--traj-margin-extra", type=float, default=DEFAULT_CONFIG["traj_margin_extra"],
                   help="extra margin for trajectory collision checking (m)")
    p.add_argument("--swarm-clearence", type=float, default=DEFAULT_CONFIG["swarm_clearence"],
                   help="minimum safety distance between drone trajectories (m)")
    p.add_argument("--smooth", action="store_true",
                   help="enable trajectory smoothing (off by default, with post-checks)")
    p.add_argument("--no-smooth", action="store_true",
                   help="legacy flag; same effect as leaving smoothing disabled")
    p.add_argument("--smooth-window", type=int, default=5,
                   help="smoothing window size (odd)")

    # Primitive library
    p.add_argument("--primitive-lib", type=str,  default=None,
                   help="primitive library directory (auto-detected by default)")

    # Output
    p.add_argument("--output-dir",  type=str,   default=None,
                   help="output directory (auto-generated under output/scenarios/<scenario_name>/ by default)")
    p.add_argument("--scenario-name", type=str, default=None,
                   help="stable scenario folder name; default is derived from index/seed/drone/obstacle counts")
    p.add_argument("--scenario-index", type=int, default=None,
                   help="optional stable scenario index used in the default folder name")
    p.add_argument("--seed",        type=int,   default=None,
                   help="random seed for reproducibility")

    return p.parse_args()


def main():
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update({
        "n_drones":           args.n_drones,
        "drone_radius":       args.drone_radius,
        "max_vel":            args.max_vel,
        "flight_height":      args.flight_height,
        "map_x":              args.map_x,
        "map_y":              args.map_y,
        "map_z":              args.map_z,
        "ground_z":           DEFAULT_CONFIG["ground_z"],
        "n_obstacles":        args.n_obstacles,
        "obs_radius_min":     args.obs_r_min,
        "obs_radius_max":     args.obs_r_max,
        "obs_height_min":     args.obs_h_min,
        "obs_height_max":     args.obs_h_max,
        "min_obs_spacing":    args.min_obs_spacing,
        "obs_inner_ratio":    args.obs_inner_ratio,
        "voxel_size":         args.voxel_size,
        "safe_margin":        args.safe_margin,
        "start_goal_clearance": args.start_goal_clearance,
        "min_drone_sep":      args.min_drone_sep,
        "min_sg_dist":        args.min_sg_dist,
        "periphery_ratio":    args.periphery_ratio,
        "traj_margin_extra":  args.traj_margin_extra,
        "swarm_clearence":    args.swarm_clearence,
        "smooth_traj":        args.smooth and not args.no_smooth,
        "smooth_window":      args.smooth_window,
        "output_dir":         args.output_dir,
        "scenario_name":      args.scenario_name,
        "scenario_index":     args.scenario_index,
        "random_seed":        args.seed,
    })

    if args.obstacle_preset != "default":
        apply_obstacle_preset(cfg, args.obstacle_preset)

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # ------------------------------------------------------------------
    # Step 1: Generate obstacles
    # ------------------------------------------------------------------
    print("\n=== Step 1: Generate obstacles ===")
    print(
        "Obstacle settings: "
        f"count={cfg['n_obstacles']}, "
        f"radius=[{cfg['obs_radius_min']:.2f}, {cfg['obs_radius_max']:.2f}], "
        f"height=[{cfg['obs_height_min']:.2f}, {cfg['obs_height_max']:.2f}], "
        f"center_spacing={cfg['min_obs_spacing']:.2f}, "
        f"inner_ratio={cfg['obs_inner_ratio']:.2f}, "
        f"voxel={cfg['voxel_size']:.2f}"
    )
    env = ObstacleEnv(cfg)
    obstacles = env.generate()
    global_cloud_world = env.export_global_cloud(cfg["voxel_size"])
    print(
        f"Placed {len(obstacles)} static cylinder obstacles "
        f"({len(global_cloud_world)} occupied voxels @ {cfg['voxel_size']:.2f} m)"
    )

    # ------------------------------------------------------------------
    # Step 2: Generate starts and goals
    # ------------------------------------------------------------------
    print("\n=== Step 2: Generate starts and goals ===")
    starts, goals = generate_starts_goals(env, cfg)
    for i, (s, g) in enumerate(zip(starts, goals)):
        d = math.hypot(g[0] - s[0], g[1] - s[1])
        print(f"  Drone {i:2d}: "
              f"start=({s[0]:+6.2f}, {s[1]:+6.2f}, {s[2]:.2f})  "
              f"goal=({g[0]:+6.2f}, {g[1]:+6.2f}, {g[2]:.2f})  "
              f"distance={d:.2f}m")

    # ------------------------------------------------------------------
    # Step 3: Load primitive library
    # ------------------------------------------------------------------
    print("\n=== Step 3: Load primitive library ===")
    lib_path = args.primitive_lib or find_primitive_lib()
    if lib_path is None:
        raise RuntimeError(
            "Cannot find the primitive library. Use --primitive-lib to specify it, "
            "or run python3 swarm_path.py first to generate the library.")
    print(f"Library path: {lib_path}")
    prim_lib = PrimitiveLibrary(lib_path)
    prim_lib.load(max_vel=cfg["max_vel"])

    # ------------------------------------------------------------------
    # Step 4: Plan trajectories for each drone sequentially
    # ------------------------------------------------------------------
    print("\n=== Step 4: Plan trajectories (with swarm avoidance) ===")
    planner = OfflinePlanner(prim_lib, env, cfg)
    drone_trajs: list[dict | None] = []

    for i in range(cfg["n_drones"]):
        print(f"  Planning drone {i} ...")
        other_trajs = [drone_trajs[j] for j in range(i) if drone_trajs[j] is not None]
        traj = planner.plan(starts[i], goals[i], other_trajs=other_trajs)
        if traj is None:
            print(f"  [FAIL] planning failed for drone {i}")
        else:
            n_pts = len(traj["timestamps"])
            print(f"  [OK]   drone {i}: {n_pts} trajectory points, "
                  f"duration {traj['duration']:.2f}s")
        drone_trajs.append(traj)

    # ------------------------------------------------------------------
    # Step 5: Save JSON
    # ------------------------------------------------------------------
    # Determine output directory
    out_dir, scenario_name = resolve_output_dir(cfg, __file__, prefix="scenario")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "scenario_data.json")

    print(f"\n=== Step 5: Save to {out_json} ===")

    output = {
        "_comment": (
            "Schema: metadata stores global scenario settings; obstacles is a list of "
            "analytic cylinders {type, x, y, z, radius, height, base_z}; "
            "scenario.global_cloud_world stores voxelized obstacle centers that mimic each cylinder using small cubes; "
            "drones is a list of {id, start, goal, trajectory}. "
            "trajectory stores duration, dt, timestamps, positions, velocities, and accelerations "
            "in world coordinates."
        ),
        "metadata": {
            "n_drones":     cfg["n_drones"],
            "n_obstacles":  len(obstacles),
            "global_cloud_points": len(global_cloud_world),
            "obstacle_voxel_size": cfg["voxel_size"],
            "scenario_name": scenario_name,
            "scenario_index": cfg["scenario_index"],
            "dt":           0.01,
            "max_vel":      cfg["max_vel"],
            "drone_radius": cfg["drone_radius"],
            "flight_height":cfg["flight_height"],
            "map_size": {
                "x": cfg["map_x"],
                "y": cfg["map_y"],
                "z": cfg["map_z"],
            },
            "random_seed":  cfg["random_seed"],
            "primitive_lib":lib_path,
            "description": (
                "Offline Primitive-Planner scenario for Isaac-Sim. "
                "Trajectory sampled at 100Hz. Positions/velocities/accelerations "
                "are in world frame (NED-compatible, Z-up)."
            ),
        },
        "scenario": {
            "global_cloud_world": global_cloud_world,
        },
        "obstacles": [dict(obs) for obs in obstacles],
        "drones": [],
    }

    n_success = 0
    for i in range(cfg["n_drones"]):
        traj = drone_trajs[i]
        drone_data: dict = {
            "id":         i,
            "start":      starts[i],
            "goal":       goals[i],
            "trajectory": traj,    # None if planning failed
        }
        output["drones"].append(drone_data)
        if traj is not None:
            n_success += 1

    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_json}")
    print(f"Output directory: {os.path.realpath(out_dir)}")
    print(f"\n========================================")
    print(f"Planning success: {n_success} / {cfg['n_drones']} drones")
    print(f"Obstacle count: {len(obstacles)}")
    if n_success > 0:
        avg_len = np.mean([
            len(drone_trajs[i]["timestamps"])
            for i in range(cfg["n_drones"])
            if drone_trajs[i] is not None
        ])
        avg_dur = np.mean([
            drone_trajs[i]["duration"]
            for i in range(cfg["n_drones"])
            if drone_trajs[i] is not None
        ])
        print(f"Average trajectory length: {avg_len:.0f} points (~{avg_len / 100:.1f}s @ 100 Hz)")
        print(f"Average flight duration: {avg_dur:.2f}s")
    print(f"========================================")


if __name__ == "__main__":
    main()
