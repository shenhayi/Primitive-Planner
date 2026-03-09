#!/usr/bin/env python3
"""
Primitive-Planner offline scenario generator with decentralized synchronous replanning.

This variant keeps random starts and goals, replans all drones on a shared
clock, executes the first horizon segment together, records the executed
motion, and exports the resulting traces to JSON for Isaac Sim.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from gen_scenario import (
    DEFAULT_CONFIG,
    ObstacleEnv,
    OfflinePlanner,
    PrimitiveLibrary,
    find_primitive_lib,
    generate_starts_goals,
)


@dataclass
class DroneRuntime:
    drone_id: int
    start: list[float]
    goal: list[float]
    pos: np.ndarray
    vel: np.ndarray
    yaw: float
    done: bool = False
    commitment: dict | None = None
    pos_log: list[np.ndarray] = field(default_factory=list)
    vel_log: list[np.ndarray] = field(default_factory=list)
    acc_log: list[np.ndarray] = field(default_factory=list)
    ts_log: list[np.ndarray] = field(default_factory=list)


class DecentralizedOfflinePlanner(OfflinePlanner):
    def _stationary_commitment(
        self, pos: np.ndarray, t_start: float, hold_dur: float
    ) -> dict:
        ts = np.array([t_start, t_start + hold_dur], dtype=float)
        pts = np.vstack([pos, pos])
        zeros = np.zeros_like(pts)
        return {
            "timestamps": ts.tolist(),
            "positions": pts.tolist(),
            "velocities": zeros.tolist(),
            "accelerations": zeros.tolist(),
            "duration": float(hold_dur),
        }

    def _proposal_from_pid(
        self,
        start_pos: np.ndarray,
        RWV: np.ndarray,
        vel_id: int,
        pid: int,
        t_start: float,
    ) -> dict:
        traj = self.lib.trajectories[(vel_id, pid)]
        local_ts = self._traj_local_ts(len(traj["pos"]), traj["duration"])
        pos_w = start_pos + (RWV @ traj["pos"].T).T
        vel_w = (RWV @ traj["vel"].T).T
        acc_w = (RWV @ traj["acc"].T).T
        return {
            "path_id": pid,
            "duration": float(traj["duration"]),
            "timestamps": (t_start + local_ts).tolist(),
            "positions": pos_w.tolist(),
            "velocities": vel_w.tolist(),
            "accelerations": acc_w.tolist(),
        }

    def _approach_proposal(
        self, cur_pos: np.ndarray, goal: np.ndarray, t_start: float
    ) -> dict:
        max_vel = self.cfg["max_vel"]
        dist = float(np.linalg.norm(cur_pos - goal))
        dur = max(0.05, dist / max_vel)
        n_pts = max(2, int(dur * 100))
        s_vals = np.linspace(0.0, 1.0, n_pts, endpoint=True)
        pos = np.outer(1 - s_vals, cur_pos) + np.outer(s_vals, goal)
        vel = np.tile((goal - cur_pos) / dur, (n_pts, 1))
        acc = np.zeros_like(pos)
        ts = t_start + np.linspace(0.0, dur, n_pts, endpoint=True)
        return {
            "path_id": -1,
            "duration": float(dur),
            "timestamps": ts.tolist(),
            "positions": pos.tolist(),
            "velocities": vel.tolist(),
            "accelerations": acc.tolist(),
        }

    def build_candidates(
        self,
        state: DroneRuntime,
        t_start: float,
        visible_trajs: list[dict],
    ) -> list[tuple[float, dict]]:
        goal = np.array(state.goal, dtype=float)
        dist_to_goal = float(np.linalg.norm(state.pos - goal))
        if dist_to_goal < self.cfg["goal_thresh"]:
            return [(0.0, self._approach_proposal(state.pos, goal, t_start))]

        RWV = self._build_rwv(state.vel, state.yaw)
        blocked = self._blocked_paths_from_swarm(
            state.pos, state.vel, t_start, RWV, visible_trajs
        )
        vel_id = min(
            int(round(np.linalg.norm(state.vel) * 10)),
            int(round(self.cfg["max_vel"] * 10)),
        )

        scored: list[tuple[float, dict]] = []
        for pid in range(self.lib.path_count):
            if pid in self.lib.infeasible.get(vel_id, set()):
                continue
            if pid in blocked:
                continue
            traj = self.lib.trajectories.get((vel_id, pid))
            if traj is None:
                continue

            pts_world = state.pos + (RWV @ traj["pos"].T).T
            if not self.env.is_traj_safe(pts_world):
                continue

            score = self._score_path(pid, state.pos, RWV, goal)
            scored.append((score, self._proposal_from_pid(state.pos, RWV, vel_id, pid, t_start)))

        scored.sort(key=lambda item: item[0])
        return scored

    def trim_execution(self, proposal: dict) -> dict:
        replan_dt = self.cfg["replan_dt"]
        ts = np.array(proposal["timestamps"], dtype=float)
        pos = np.array(proposal["positions"], dtype=float)
        vel = np.array(proposal["velocities"], dtype=float)
        acc = np.array(proposal["accelerations"], dtype=float)
        t0 = float(ts[0])

        mask = (ts - t0) < replan_dt
        if not np.any(mask):
            mask[0] = True

        return {
            "timestamps": ts[mask],
            "positions": pos[mask],
            "velocities": vel[mask],
            "accelerations": acc[mask],
        }

    def simulate(self, starts: list, goals: list) -> list[DroneRuntime]:
        cfg = self.cfg
        runtimes: list[DroneRuntime] = []
        for i, (start, goal) in enumerate(zip(starts, goals)):
            d0 = np.array(goal, dtype=float) - np.array(start, dtype=float)
            runtimes.append(
                DroneRuntime(
                    drone_id=i,
                    start=start,
                    goal=goal,
                    pos=np.array(start, dtype=float),
                    vel=np.zeros(3),
                    yaw=math.atan2(d0[1], d0[0]),
                )
            )

        global_t = 0.0
        hold_dur = max(cfg["replan_dt"], 0.5)
        for cycle in range(cfg["max_plan_steps"]):
            if all(rt.done for rt in runtimes):
                break

            visible_prev: dict[int, dict] = {}
            for rt in runtimes:
                if rt.done:
                    visible_prev[rt.drone_id] = self._stationary_commitment(rt.pos, global_t, hold_dur)
                elif rt.commitment is not None:
                    visible_prev[rt.drone_id] = rt.commitment

            proposals: dict[int, list[tuple[float, dict]]] = {}
            for rt in runtimes:
                if rt.done:
                    continue
                visible_trajs = [
                    traj for did, traj in visible_prev.items() if did != rt.drone_id
                ]
                proposals[rt.drone_id] = self.build_candidates(rt, global_t, visible_trajs)

            accepted: dict[int, dict] = {}
            accepted_trajs: list[dict] = []
            for rt in runtimes:
                if rt.done:
                    continue

                chosen = None
                for _, proposal in proposals.get(rt.drone_id, []):
                    ts = np.array(proposal["timestamps"], dtype=float)
                    pos = np.array(proposal["positions"], dtype=float)
                    if accepted_trajs and not self._traj_vs_swarm_safe(
                        ts, pos, accepted_trajs, cfg["swarm_clearence"]
                    ):
                        continue
                    chosen = proposal
                    break

                if chosen is None:
                    chosen = self._stationary_commitment(rt.pos, global_t, cfg["replan_dt"])

                accepted[rt.drone_id] = chosen
                accepted_trajs.append(
                    {
                        "timestamps": chosen["timestamps"],
                        "positions": chosen["positions"],
                    }
                )

            for rt in runtimes:
                if rt.done:
                    rt.commitment = self._stationary_commitment(rt.pos, global_t, hold_dur)
                    continue

                proposal = accepted[rt.drone_id]
                rt.commitment = proposal
                executed = self.trim_execution(proposal)
                rt.pos_log.append(executed["positions"])
                rt.vel_log.append(executed["velocities"])
                rt.acc_log.append(executed["accelerations"])
                rt.ts_log.append(executed["timestamps"])

                rt.pos = executed["positions"][-1].copy()
                rt.vel = executed["velocities"][-1].copy()
                if np.linalg.norm(rt.vel[:2]) > 0.05:
                    rt.yaw = math.atan2(rt.vel[1], rt.vel[0])

                if np.linalg.norm(rt.pos - np.array(rt.goal, dtype=float)) < cfg["goal_thresh"]:
                    rt.done = True
                    rt.pos = np.array(rt.goal, dtype=float)
                    rt.vel = np.zeros(3)
                    rt.commitment = self._stationary_commitment(rt.pos, global_t + cfg["replan_dt"], hold_dur)

            global_t += cfg["replan_dt"]
        else:
            print(f"[WARN] reached maximum decentralized cycles: {cfg['max_plan_steps']}")

        return runtimes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Primitive-Planner offline scenario generator with decentralized replanning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-drones", type=int, default=DEFAULT_CONFIG["n_drones"])
    parser.add_argument("--drone-radius", type=float, default=DEFAULT_CONFIG["drone_radius"])
    parser.add_argument("--max-vel", type=float, default=DEFAULT_CONFIG["max_vel"])
    parser.add_argument("--flight-height", type=float, default=DEFAULT_CONFIG["flight_height"])
    parser.add_argument("--map-x", type=float, default=DEFAULT_CONFIG["map_x"])
    parser.add_argument("--map-y", type=float, default=DEFAULT_CONFIG["map_y"])
    parser.add_argument("--map-z", type=float, default=DEFAULT_CONFIG["map_z"])
    parser.add_argument("--n-obstacles", type=int, default=DEFAULT_CONFIG["n_obstacles"])
    parser.add_argument("--obs-r-min", type=float, default=DEFAULT_CONFIG["obs_radius_min"])
    parser.add_argument("--obs-r-max", type=float, default=DEFAULT_CONFIG["obs_radius_max"])
    parser.add_argument("--obs-h-min", type=float, default=DEFAULT_CONFIG["obs_height_min"])
    parser.add_argument("--obs-h-max", type=float, default=DEFAULT_CONFIG["obs_height_max"])
    parser.add_argument("--safe-margin", type=float, default=DEFAULT_CONFIG["safe_margin"])
    parser.add_argument("--min-drone-sep", type=float, default=DEFAULT_CONFIG["min_drone_sep"])
    parser.add_argument("--min-sg-dist", type=float, default=DEFAULT_CONFIG["min_sg_dist"])
    parser.add_argument("--traj-margin-extra", type=float, default=DEFAULT_CONFIG["traj_margin_extra"])
    parser.add_argument("--swarm-clearence", type=float, default=DEFAULT_CONFIG["swarm_clearence"])
    parser.add_argument("--primitive-lib", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = DEFAULT_CONFIG.copy()
    cfg.update(
        {
            "n_drones": args.n_drones,
            "drone_radius": args.drone_radius,
            "max_vel": args.max_vel,
            "flight_height": args.flight_height,
            "map_x": args.map_x,
            "map_y": args.map_y,
            "map_z": args.map_z,
            "n_obstacles": args.n_obstacles,
            "obs_radius_min": args.obs_r_min,
            "obs_radius_max": args.obs_r_max,
            "obs_height_min": args.obs_h_min,
            "obs_height_max": args.obs_h_max,
            "safe_margin": args.safe_margin,
            "min_drone_sep": args.min_drone_sep,
            "min_sg_dist": args.min_sg_dist,
            "traj_margin_extra": args.traj_margin_extra,
            "swarm_clearence": args.swarm_clearence,
            "output_dir": args.output_dir,
            "random_seed": args.seed,
            "smooth_traj": False,
        }
    )

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    print("\n=== Step 1: Generate obstacles ===")
    env = ObstacleEnv(cfg)
    obstacles = env.generate()
    print(f"Placed {len(obstacles)} cylindrical obstacles")

    print("\n=== Step 2: Generate random starts and goals ===")
    starts, goals = generate_starts_goals(env, cfg)
    for i, (s, g) in enumerate(zip(starts, goals)):
        print(
            f"  Drone {i:2d}: start=({s[0]:+6.2f}, {s[1]:+6.2f}, {s[2]:.2f})  "
            f"goal=({g[0]:+6.2f}, {g[1]:+6.2f}, {g[2]:.2f})"
        )

    print("\n=== Step 3: Load primitive library ===")
    lib_path = args.primitive_lib or find_primitive_lib()
    if lib_path is None:
        raise RuntimeError("Cannot find the primitive library. Use --primitive-lib to specify it.")
    print(f"Library path: {lib_path}")
    prim_lib = PrimitiveLibrary(lib_path)
    prim_lib.load(max_vel=cfg["max_vel"])

    print("\n=== Step 4: Synchronous decentralized replanning ===")
    planner = DecentralizedOfflinePlanner(prim_lib, env, cfg)
    runtimes = planner.simulate(starts, goals)

    if cfg["output_dir"]:
        out_dir = cfg["output_dir"]
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        seed_str = f"_seed{cfg['random_seed']}" if cfg["random_seed"] is not None else ""
        folder_name = f"scenario_sync_{ts}{seed_str}_d{cfg['n_drones']}_obs{cfg['n_obstacles']}"
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "..", "..", "output", "scenarios", folder_name)

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "scenario_data.json")
    print(f"\n=== Step 5: Save to {out_json} ===")

    output = {
        "_comment": (
            "Schema: metadata stores global scenario settings; obstacles is a list of "
            "cylinders {x, y, radius, height}; drones is a list of {id, start, goal, trajectory}. "
            "trajectory stores duration, dt, timestamps, positions, velocities, and accelerations "
            "in world coordinates."
        ),
        "metadata": {
            "mode": "decentralized_synchronous_replan",
            "n_drones": cfg["n_drones"],
            "n_obstacles": len(obstacles),
            "dt": 0.01,
            "replan_dt": cfg["replan_dt"],
            "max_vel": cfg["max_vel"],
            "drone_radius": cfg["drone_radius"],
            "flight_height": cfg["flight_height"],
            "map_size": {"x": cfg["map_x"], "y": cfg["map_y"], "z": cfg["map_z"]},
            "random_seed": cfg["random_seed"],
            "primitive_lib": lib_path,
        },
        "obstacles": [
            {"x": float(cx), "y": float(cy), "radius": float(r), "height": float(h)}
            for (cx, cy, r, h) in obstacles
        ],
        "drones": [],
    }

    success = 0
    for rt in runtimes:
        if rt.ts_log:
            traj = {
                "duration": float(rt.ts_log[-1][-1]),
                "dt": 0.01,
                "timestamps": np.concatenate(rt.ts_log).tolist(),
                "positions": np.vstack(rt.pos_log).tolist(),
                "velocities": np.vstack(rt.vel_log).tolist(),
                "accelerations": np.vstack(rt.acc_log).tolist(),
            }
            success += 1
        else:
            traj = None
        output["drones"].append(
            {"id": rt.drone_id, "start": rt.start, "goal": rt.goal, "trajectory": traj}
        )

    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved to {out_json}")
    print(f"Planning success: {success} / {cfg['n_drones']} drones")


if __name__ == "__main__":
    main()
