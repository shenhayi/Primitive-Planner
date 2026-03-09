#!/usr/bin/env python3
"""
Primitive-Planner offline scenario generator (fast variant)
==========================================================
Features:
  1. Randomly generate a static cylindrical obstacle environment
  2. Randomly sample collision-free start and goal points for multiple drones
  3. Use the Primitive-Planner core logic and precomputed library for fast planning
  4. Export position, velocity, and acceleration at each time step
  5. Save everything to JSON for Isaac Sim replay

Usage examples:
  # Default parameters (5 drones, 40 obstacles)
  conda run -n silent python3 gen_scenario_fast.py

  # Custom parameters
  conda run -n silent python3 gen_scenario_fast.py --n-drones 10 --n-obstacles 60 --map-x 30 --map-y 30 --seed 42

  # Specify the primitive library path and output file
  conda run -n silent python3 gen_scenario_fast.py --primitive-lib /path/to/primitive_library --output my_scene.json

Output JSON schema:
  {
    "metadata": { "n_drones", "dt", "max_vel", ... },
    "obstacles": [ {"x", "y", "radius", "height"}, ... ],
    "drones": [
      {
        "id": 0, "start": [x,y,z], "goal": [x,y,z],
        "trajectory": {
          "duration": float,
          "timestamps": [...],
          "positions": [[x,y,z], ...],      # 100Hz
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
import datetime
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
    "map_z":            3.0,     # m, Z height (0 ~ map_z)

    # Cylindrical obstacle parameters (center region only)
    "n_obstacles":      40,
    "obs_radius_min":   0.2,     # m
    "obs_radius_max":   0.5,     # m
    "obs_height_min":   0.0,     # m
    "obs_height_max":   3.0,     # m
    "min_obs_spacing":  0.5,     # m, minimum obstacle edge spacing
    "obs_inner_ratio":  0.55,    # obstacles are placed in the center region only

    # Start/goal parameters
    "safe_margin":      0.5,     # m, minimum clearance from obstacle surfaces
    "traj_margin_extra": 0.05,   # m, extra margin in trajectory collision checks
    "min_sg_dist":      8.0,     # m, minimum distance between one drone's start and goal
    "min_drone_sep":    2.5,     # m, minimum separation between different drones
    "max_place_tries":  2000,    # maximum random placement retries

    # Primitive library path (None = auto-detect)
    "primitive_lib_path": None,

    # Planner parameters (must match the primitive library generation)
    "box_x":            6.0,     # m
    "box_y":            6.0,     # m
    "box_z":            6.0,     # m
    "lambda_l":         12.0,    # goal progress cost weight
    "lambda_b":         12.0,    # boundary cost weight

    # Replanning termination conditions
    "replan_dt":        0.2,     # s, replan every 0.2 s
    "goal_thresh":      0.5,     # m, goal reached threshold
    "max_plan_steps":   500,     # maximum replanning iterations
    "swarm_clearence":  0.35,    # m, minimum drone-drone clearance

    # Trajectory post-processing
    "smooth_traj":      False,   # whether to smooth the trajectory
    "smooth_window":    5,       # smoothing window size (odd)

    # Output
    "output_dir":       None,    # output directory, None = auto-generate
    "random_seed":      None,    # integer or None
}


# ================================================================
# Obstacle environment
# ================================================================
class ObstacleEnv:
    """Generate random cylindrical obstacles and handle collision checks."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        # list of (cx, cy, radius, height)
        self.obstacles: list[tuple] = []

    def generate(self) -> list[tuple]:
        """Generate random cylindrical obstacles in the center region only."""
        cfg = self.cfg
        inner_ratio = cfg.get("obs_inner_ratio", 0.55)
        half_x = cfg["map_x"] / 2 * inner_ratio
        half_y = cfg["map_y"] / 2 * inner_ratio
        placed: list[tuple] = []

        for _ in range(cfg["n_obstacles"]):
            for _try in range(10000):
                cx = np.random.uniform(-half_x, half_x)
                cy = np.random.uniform(-half_y, half_y)
                r  = np.random.uniform(cfg["obs_radius_min"], cfg["obs_radius_max"])
                h  = np.random.uniform(cfg["obs_height_min"], cfg["obs_height_max"])

                # Keep spacing from previously placed obstacles
                valid = True
                for (ex, ey, er, _) in placed:
                    if math.hypot(cx - ex, cy - ey) < r + er + cfg["min_obs_spacing"]:
                        valid = False
                        break
                if valid:
                    placed.append((cx, cy, r, h))
                    break
            else:
                print(f"[WARN] only placed {len(placed)} obstacles (target {cfg['n_obstacles']})")
                break

        self.obstacles = placed
        return placed

    def _in_cylinder(self, x: float, y: float, z: float, margin: float) -> bool:
        """Check whether a point lies inside any cylinder with margin."""
        for (cx, cy, r, h) in self.obstacles:
            if 0 < z < h and math.hypot(x - cx, y - cy) < r + margin:
                return True
        return False

    def is_point_safe(self, x: float, y: float, z: float) -> bool:
        """Check whether a single point is safe and inside the map."""
        cfg = self.cfg
        margin = cfg["drone_radius"] + cfg["safe_margin"]
        half_x = cfg["map_x"] / 2
        half_y = cfg["map_y"] / 2
        if abs(x) > half_x - margin or abs(y) > half_y - margin:
            return False
        z_margin = max(0.1, cfg["drone_radius"])
        if z < z_margin or z > cfg["map_z"] - z_margin:
            return False
        return not self._in_cylinder(x, y, z, margin)

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
        if np.any(np.abs(xs) > half_x) or np.any(np.abs(ys) > half_y):
            return False
        if np.any(zs <= 0) or np.any(zs >= cfg["map_z"]):
            return False

        for (cx, cy, r, h) in self.obstacles:
            in_z = (zs > 0) & (zs < h)
            if not np.any(in_z):
                continue
            d_xy = np.hypot(xs - cx, ys - cy)
            if np.any((d_xy < r + margin) & in_z):
                return False

        return True


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
        self.infeasible: dict[int, set] = {}     # vel_id -> set of infeasible path_ids
        self.agent_correspondence: dict[int, dict[int, np.ndarray]] = {}

    def load(self, max_vel: float = 1.0):
        """Load all required data from disk."""
        t0 = time.time()
        print("  Loading path_end.ply ...")
        self._load_path_ends()
        print(f"    {self.path_count} paths found")

        print("  Loading path_all.ply ...")
        self._load_path_all()

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
            blocked_paths = self._blocked_paths_from_swarm(
                cur_pos, cur_vel, t_elapsed, RWV, other_trajs
            )

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

                # Use dense trajectory points for collision (not path_all 0.1m sampling)
                pts_body  = traj["pos"]
                pts_world = cur_pos + (RWV @ pts_body.T).T
                if not self.env.is_traj_safe(pts_world):
                    continue

                score = self._score_path(pid, cur_pos, RWV, goal_arr)
                scored.append((score, pid))

            if not scored:
                print(f"      [WARN] step={step}: no feasible path")
                break

            scored.sort(key=lambda x: x[0])

            best_pid = scored[0][1]
            traj = self.lib.trajectories[(vel_id, best_pid)]
            pos_w = cur_pos + (RWV @ traj["pos"].T).T
            vel_w = (RWV @ traj["vel"].T).T
            acc_w = (RWV @ traj["acc"].T).T

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
                if not self.env.is_traj_safe(full_pos):
                    print("      [WARN] smoothed trajectory collides with obstacles, discarding smoothed result")
                    full_pos = np.vstack(all_pos)
                    full_vel = np.vstack(all_vel)
                    full_acc = np.vstack(all_acc)

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
    z = cfg["flight_height"]
    map_radius = min(cfg["map_x"], cfg["map_y"]) / 2
    # Outer radius: stay outside the obstacle region with safety margin
    inner_ratio = cfg.get("obs_inner_ratio", 0.55)
    periphery_radius = map_radius * (inner_ratio + 0.35)  # 0.55+0.35=0.9, outside obstacle region
    periphery_radius = min(periphery_radius, map_radius * 0.92)

    n = cfg["n_drones"]
    starts, goals = [], []
    max_tries = cfg.get("max_place_tries", 2000)
    min_sep = cfg["min_drone_sep"]
    min_sg_dist = cfg["min_sg_dist"]

    chord = 2.0 * periphery_radius * math.sin(math.pi / max(n, 1))
    if n > 1 and chord < min_sep:
        raise RuntimeError(
            f"current map with min_drone_sep={min_sep} cannot fit ring starts/goals for {n} drones"
        )

    chosen_angles: list[float] = []
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
            if not env.is_point_safe(start[0], start[1], start[2]):
                continue
            if not env.is_point_safe(goal[0], goal[1], goal[2]):
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


# ================================================================
# Main entry point
# ================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Primitive-Planner offline scenario generator (fast variant)",
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
    p.add_argument("--n-obstacles", type=int,   default=DEFAULT_CONFIG["n_obstacles"],
                   help="number of obstacles")
    p.add_argument("--obs-r-min",   type=float, default=DEFAULT_CONFIG["obs_radius_min"],
                   help="minimum obstacle radius (m)")
    p.add_argument("--obs-r-max",   type=float, default=DEFAULT_CONFIG["obs_radius_max"],
                   help="maximum obstacle radius (m)")
    p.add_argument("--obs-h-min",   type=float, default=DEFAULT_CONFIG["obs_height_min"],
                   help="minimum obstacle height (m)")
    p.add_argument("--obs-h-max",   type=float, default=DEFAULT_CONFIG["obs_height_max"],
                   help="maximum obstacle height (m)")

    # Paths
    p.add_argument("--safe-margin", type=float, default=DEFAULT_CONFIG["safe_margin"],
                   help="minimum clearance from starts/goals to obstacles (m)")
    p.add_argument("--min-drone-sep", type=float, default=DEFAULT_CONFIG["min_drone_sep"],
                   help="minimum separation between different drones' starts/goals (m)")
    p.add_argument("--min-sg-dist", type=float, default=DEFAULT_CONFIG["min_sg_dist"],
                   help="minimum straight-line distance between one drone's start and goal (m)")
    p.add_argument("--traj-margin-extra", type=float, default=0.05,
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
                   help="output directory (auto-generated under output/scenarios/<timestamp>/ by default)")
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
        "n_obstacles":        args.n_obstacles,
        "obs_radius_min":     args.obs_r_min,
        "obs_radius_max":     args.obs_r_max,
        "obs_height_min":     args.obs_h_min,
        "obs_height_max":     args.obs_h_max,
        "safe_margin":        args.safe_margin,
        "min_drone_sep":      args.min_drone_sep,
        "min_sg_dist":        args.min_sg_dist,
        "traj_margin_extra":  args.traj_margin_extra,
        "swarm_clearence":    args.swarm_clearence,
        "smooth_traj":        args.smooth and not args.no_smooth,
        "smooth_window":      args.smooth_window,
        "output_dir":         args.output_dir,
        "random_seed":        args.seed,
    })

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # ------------------------------------------------------------------
    # Step 1: Generate obstacles
    # ------------------------------------------------------------------
    print("\n=== Step 1: Generate obstacles ===")
    env = ObstacleEnv(cfg)
    obstacles = env.generate()
    print(f"Placed {len(obstacles)} cylindrical obstacles")

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
    print("\n=== Step 4: Fast trajectory planning (primitive-library-based) ===")
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
    if cfg["output_dir"]:
        out_dir = cfg["output_dir"]
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"_seed{cfg['random_seed']}" if cfg["random_seed"] is not None else ""
        folder_name = f"scenario_{ts}{seed_str}_d{cfg['n_drones']}_obs{cfg['n_obstacles']}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "..", "..", "output", "scenarios", folder_name)

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "scenario_data.json")

    print(f"\n=== Step 5: Save to {out_json} ===")

    output = {
        "metadata": {
            "n_drones":     cfg["n_drones"],
            "n_obstacles":  len(obstacles),
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
        "obstacles": [
            {
                "x":      float(cx),
                "y":      float(cy),
                "radius": float(r),
                "height": float(h),
            }
            for (cx, cy, r, h) in obstacles
        ],
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
