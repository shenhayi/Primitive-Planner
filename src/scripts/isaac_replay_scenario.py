#!/usr/bin/env python3
"""
Replay Primitive-Planner scenario_data.json inside Isaac Sim 2023.1.1.

This script:
1. Loads the scenario JSON produced by gen_scenario.py or gen_scenario_decentralized.py
2. Builds a simple USD scene with a ground plane, static box obstacles, and one prim per drone
3. Replays each drone trajectory by interpolating the logged timestamps

Run from the Isaac Sim root:
  ./python.sh /abs/path/to/src/scripts/isaac_replay_scenario.py \
      --json /abs/path/to/scenario_data.json

Optional:
  --headless                 Run without a viewport
  --speed 2.0                Replay at 2x speed
  --save-stage /tmp/a.usda   Save the generated scene before playback
  --drone-shape sphere       Use sphere, cube, or capsule placeholders
  --drone-scale 0.30         Visual size in meters
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Primitive-Planner scenario_data.json in Isaac Sim 2023.1.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--json", required=True, help="Path to scenario_data.json")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument(
        "--drone-shape",
        choices=("sphere", "cube", "capsule"),
        default="sphere",
        help="Placeholder visual used for drones",
    )
    parser.add_argument(
        "--drone-scale",
        type=float,
        default=0.30,
        help="Placeholder drone size in meters",
    )
    parser.add_argument(
        "--ground-size",
        type=float,
        default=60.0,
        help="Ground plane side length in meters",
    )
    parser.add_argument(
        "--save-stage",
        type=str,
        default=None,
        help="Optional .usd/.usda output path for the generated stage",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop playback after the final timestamp",
    )
    return parser.parse_args()


def load_scenario(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def lerp_positions(ts: np.ndarray, positions: np.ndarray, t: float) -> np.ndarray:
    if len(ts) == 0:
        return np.zeros(3, dtype=float)
    if t <= ts[0]:
        return positions[0]
    if t >= ts[-1]:
        return positions[-1]
    idx = np.searchsorted(ts, t, side="right") - 1
    idx = max(0, min(idx, len(ts) - 2))
    alpha = (t - ts[idx]) / max(ts[idx + 1] - ts[idx], 1e-9)
    return positions[idx] * (1.0 - alpha) + positions[idx + 1] * alpha


def yaw_from_velocity(vel: np.ndarray) -> float:
    if np.linalg.norm(vel[:2]) < 1e-5:
        return 0.0
    return math.atan2(float(vel[1]), float(vel[0]))


def drone_color(drone_id: int) -> tuple[float, float, float]:
    palette = [
        (0.85, 0.22, 0.18),
        (0.16, 0.55, 0.90),
        (0.17, 0.70, 0.36),
        (0.94, 0.71, 0.12),
        (0.66, 0.31, 0.80),
        (0.12, 0.70, 0.73),
        (0.95, 0.44, 0.21),
        (0.45, 0.45, 0.50),
    ]
    return palette[drone_id % len(palette)]


def obstacle_color() -> tuple[float, float, float]:
    return (0.35, 0.35, 0.38)


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.json).expanduser().resolve()
    scenario = load_scenario(scenario_path)

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    try:
        import omni
        from omni.isaac.core import World
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        world = World(stage_units_in_meters=1.0)
        stage = world.stage

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world.scene.add_default_ground_plane()

        root = stage.DefinePrim("/World/Scenario", "Xform")
        obstacles_root = stage.DefinePrim("/World/Scenario/Obstacles", "Xform")
        drones_root = stage.DefinePrim("/World/Scenario/Drones", "Xform")

        looks_scope = UsdGeom.Scope.Define(stage, "/World/Looks")

        def ensure_material(name: str, rgb: tuple[float, float, float]) -> str:
            material_path = f"/World/Looks/{name}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*rgb)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material_path

        obstacle_material = ensure_material("Obstacle", obstacle_color())

        drone_materials: dict[int, str] = {}
        for drone in scenario.get("drones", []):
            drone_id = int(drone["id"])
            drone_materials[drone_id] = ensure_material(
                f"Drone_{drone_id:03d}", drone_color(drone_id)
            )

        def bind_material(prim, material_path: str) -> None:
            material = UsdShade.Material.Get(stage, material_path)
            UsdShade.MaterialBindingAPI(prim).Bind(material)

        for obs_idx, obs in enumerate(scenario.get("obstacles", [])):
            if obs.get("type") != "box":
                continue
            prim_path = f"/World/Scenario/Obstacles/Obstacle_{obs_idx:03d}"
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.XformCommonAPI(cube)
            xform.SetTranslate((float(obs["x"]), float(obs["y"]), float(obs["z"])))
            xform.SetRotate(
                (0.0, 0.0, math.degrees(float(obs.get("yaw", 0.0)))),
                UsdGeom.XformCommonAPI.RotationOrderXYZ,
            )
            xform.SetScale(
                (float(obs["size_x"]), float(obs["size_y"]), float(obs["size_z"]))
            )
            bind_material(cube.GetPrim(), obstacle_material)

        drone_prims: dict[int, str] = {}
        drone_tracks: dict[int, dict] = {}
        drone_radius = float(
            scenario.get("metadata", {}).get("drone_radius", max(0.05, args.drone_scale / 2.0))
        )

        for drone in scenario.get("drones", []):
            drone_id = int(drone["id"])
            prim_path = f"/World/Scenario/Drones/Drone_{drone_id:03d}"
            shape = args.drone_shape

            if shape == "cube":
                geom = UsdGeom.Cube.Define(stage, prim_path)
                geom.CreateSizeAttr(1.0)
                xform = UsdGeom.XformCommonAPI(geom)
                xform.SetScale((args.drone_scale, args.drone_scale, args.drone_scale * 0.35))
                prim = geom.GetPrim()
            elif shape == "capsule":
                geom = UsdGeom.Capsule.Define(stage, prim_path)
                geom.CreateRadiusAttr(args.drone_scale * 0.35)
                geom.CreateHeightAttr(args.drone_scale)
                prim = geom.GetPrim()
            else:
                geom = UsdGeom.Sphere.Define(stage, prim_path)
                geom.CreateRadiusAttr(max(drone_radius, args.drone_scale * 0.5))
                prim = geom.GetPrim()

            bind_material(prim, drone_materials[drone_id])
            drone_prims[drone_id] = prim_path

            traj = drone.get("trajectory")
            if traj is None:
                ts = np.array([0.0], dtype=float)
                pos = np.array([drone["start"]], dtype=float)
                vel = np.zeros((1, 3), dtype=float)
            else:
                ts = np.array(traj["timestamps"], dtype=float)
                pos = np.array(traj["positions"], dtype=float)
                vel = np.array(traj["velocities"], dtype=float)
            drone_tracks[drone_id] = {
                "timestamps": ts,
                "positions": pos,
                "velocities": vel,
            }

            start_pos = pos[0]
            xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim_path))
            xform.SetTranslate((float(start_pos[0]), float(start_pos[1]), float(start_pos[2])))

        world.reset()

        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport is not None:
            viewport.set_camera_position("/OmniverseKit_Persp", 18.0, -18.0, 16.0, True)
            viewport.set_camera_target("/OmniverseKit_Persp", 0.0, 0.0, 0.0, True)

        if args.save_stage:
            save_path = str(Path(args.save_stage).expanduser().resolve())
            omni.usd.get_context().save_as_stage(save_path)
            print(f"Saved stage to {save_path}")

        duration = 0.0
        for track in drone_tracks.values():
            if len(track["timestamps"]) > 0:
                duration = max(duration, float(track["timestamps"][-1]))

        if duration <= 0.0:
            duration = 0.1

        print(f"Loaded scenario: {scenario_path}")
        print(f"Obstacles: {len(scenario.get('obstacles', []))}")
        print(f"Drones: {len(drone_tracks)}")
        print(f"Playback duration: {duration:.2f}s @ speed {args.speed:.2f}x")

        start_wall = time.perf_counter()
        while simulation_app.is_running():
            world.step(render=True)

            elapsed = (time.perf_counter() - start_wall) * args.speed
            if args.loop and duration > 1e-6:
                sim_t = elapsed % duration
            else:
                sim_t = min(elapsed, duration)

            for drone_id, track in drone_tracks.items():
                pos = lerp_positions(track["timestamps"], track["positions"], sim_t)
                vel = lerp_positions(track["timestamps"], track["velocities"], sim_t)
                yaw_deg = math.degrees(yaw_from_velocity(vel))
                xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(drone_prims[drone_id]))
                xform.SetTranslate((float(pos[0]), float(pos[1]), float(pos[2])))
                xform.SetRotate((0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

            if not args.loop and elapsed >= duration:
                for _ in range(60):
                    if not simulation_app.is_running():
                        break
                    world.step(render=True)
                break

    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
