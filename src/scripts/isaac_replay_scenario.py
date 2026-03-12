#!/usr/bin/env python3
"""
Replay Primitive-Planner scenario_data.json inside Isaac Sim 2023.1.1.

This script:
1. Loads the scenario JSON produced by gen_scenario.py or gen_scenario_decentralized.py
2. Builds a simple USD scene with a ground plane, static obstacles, and one prim per drone
3. Replays each drone trajectory by interpolating the logged timestamps
4. Optionally records the viewport to a local mp4 and uploads it to Weights & Biases

Run from the Isaac Sim root:
  ./python.sh /abs/path/to/src/scripts/isaac_replay_scenario.py \
      --json /abs/path/to/scenario_data.json
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import datetime
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
        "--camera-eye",
        nargs=3,
        type=float,
        default=(18.0, -18.0, 16.0),
        help="Viewer camera eye position",
    )
    parser.add_argument(
        "--camera-lookat",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        help="Viewer camera look-at target",
    )
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
    parser.add_argument(
        "--record",
        action="store_true",
        help="Capture viewport frames and save a replay video",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=2,
        help="Capture every N rendered frames",
    )
    parser.add_argument(
        "--video-tag",
        type=str,
        default="recording",
        help="Weights & Biases metric key for the replay video",
    )
    parser.add_argument(
        "--video-output",
        type=str,
        default=None,
        help="Optional local mp4 output path",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Upload replay metadata and video to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="primitive-planner-replay",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline", "disabled"),
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional Weights & Biases run id to resume",
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


def yaw_from_velocity(vel: np.ndarray, fallback_yaw: float = 0.0) -> float:
    if np.linalg.norm(vel[:2]) < 1e-5:
        return fallback_yaw
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


def scenario_duration(drones: list[dict]) -> float:
    duration = 0.0
    for drone in drones:
        traj = drone.get("trajectory")
        if traj and traj.get("timestamps"):
            duration = max(duration, float(traj["timestamps"][-1]))
    return max(duration, 0.1)


def run_async(awaitable):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(awaitable)


def capture_rgb_frame(viewport_api, viewport_utility) -> np.ndarray | None:
    frame: dict[str, np.ndarray] = {}
    errors: list[Exception] = []

    def capture_callback(buffer, buffer_size, width, height, _format) -> None:
        try:
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.POINTER(ctypes.c_byte * buffer_size)
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            content = ctypes.pythonapi.PyCapsule_GetPointer(buffer, None)
            rgba = np.frombuffer(bytes(content.contents), dtype=np.uint8).reshape(height, width, 4)
            frame["rgb"] = rgba[..., :3].copy()
        except Exception as exc:  # pragma: no cover - Isaac runtime specific
            errors.append(exc)

    capture = viewport_utility.capture_viewport_to_buffer(viewport_api, capture_callback)
    if capture is None:
        return None
    result = run_async(capture.wait_for_result())
    if errors:
        raise RuntimeError(f"Viewport frame capture failed: {errors[0]}") from errors[0]
    if not result:
        return None
    return frame.get("rgb")


def estimate_video_fps(capture_times: list[float], fallback_fps: int = 30) -> int:
    if len(capture_times) < 2:
        return fallback_fps
    diffs = np.diff(np.asarray(capture_times, dtype=float))
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return fallback_fps
    return max(1, int(round(1.0 / float(np.median(diffs)))))


def default_video_output_path(scenario_path: Path) -> Path:
    return scenario_path.with_name(f"{scenario_path.stem}_replay.mp4")


def write_video(video_path: Path, frames: np.ndarray, fps: int) -> None:
    try:
        import torch
        from torchvision.io import write_video as tv_write_video

        tv_write_video(str(video_path), torch.from_numpy(frames), fps=fps)
        return
    except Exception:
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(video_path), frames, fps=fps)
            return
        except Exception as exc:
            raise RuntimeError(
                "Failed to save replay video. Install torchvision or imageio in the Isaac Sim Python environment."
            ) from exc


def main() -> None:
    args = parse_args()
    scenario_path = Path(args.json).expanduser().resolve()
    scenario = load_scenario(scenario_path)

    from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "anti_aliasing": 1})

    wandb_run = None
    try:
        import omni
        from omni.isaac.core import World
        from pxr import Gf, Sdf, UsdGeom, UsdShade

        if args.wandb:
            import wandb

            run_name = args.wandb_name or (
                f"{scenario_path.stem}-{datetime.datetime.now().strftime('%m-%d_%H-%M-%S')}"
            )
            init_kwargs = {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": run_name,
                "mode": args.wandb_mode,
                "config": {
                    "json": str(scenario_path),
                    "speed": args.speed,
                    "loop": args.loop,
                    "record": args.record,
                    "record_every": args.record_every,
                    "drone_shape": args.drone_shape,
                    "drone_scale": args.drone_scale,
                },
            }
            if args.wandb_run_id is not None:
                init_kwargs["id"] = args.wandb_run_id
                init_kwargs["resume"] = "allow"
            wandb_run = wandb.init(**init_kwargs)

        world = World(stage_units_in_meters=1.0)
        stage = world.stage

        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)

        world.scene.add_default_ground_plane()

        stage.DefinePrim("/World/Scenario", "Xform")
        stage.DefinePrim("/World/Scenario/Obstacles", "Xform")
        stage.DefinePrim("/World/Scenario/Drones", "Xform")
        UsdGeom.Scope.Define(stage, "/World/Looks")

        def ensure_material(name: str, rgb: tuple[float, float, float]) -> str:
            material_path = f"/World/Looks/{name}"
            material = UsdShade.Material.Define(stage, material_path)
            shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
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

        global_cloud_world = scenario.get("scenario", {}).get("global_cloud_world", [])
        obstacle_voxel_size = float(scenario.get("metadata", {}).get("obstacle_voxel_size", 0.1))
        if global_cloud_world:
            for obs_idx, point in enumerate(global_cloud_world):
                prim_path = f"/World/Scenario/Obstacles/Voxel_{obs_idx:05d}"
                cube = UsdGeom.Cube.Define(stage, prim_path)
                cube.CreateSizeAttr(1.0)
                xform = UsdGeom.XformCommonAPI(cube)
                xform.SetTranslate((float(point[0]), float(point[1]), float(point[2])))
                xform.SetScale((obstacle_voxel_size, obstacle_voxel_size, obstacle_voxel_size))
                bind_material(cube.GetPrim(), obstacle_material)
        else:
            for obs_idx, obs in enumerate(scenario.get("obstacles", [])):
                prim_path = f"/World/Scenario/Obstacles/Obstacle_{obs_idx:03d}"
                if obs.get("type") == "cylinder":
                    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
                    cyl.CreateRadiusAttr(float(obs["radius"]))
                    cyl.CreateHeightAttr(float(obs["height"]))
                    xform = UsdGeom.XformCommonAPI(cyl)
                    xform.SetTranslate((float(obs["x"]), float(obs["y"]), float(obs["z"])))
                    bind_material(cyl.GetPrim(), obstacle_material)
                    continue
                if obs.get("type") != "box":
                    continue
                cube = UsdGeom.Cube.Define(stage, prim_path)
                cube.CreateSizeAttr(1.0)
                xform = UsdGeom.XformCommonAPI(cube)
                xform.SetTranslate((float(obs["x"]), float(obs["y"]), float(obs["z"])))
                xform.SetRotate(
                    (0.0, 0.0, math.degrees(float(obs.get("yaw", 0.0)))),
                    UsdGeom.XformCommonAPI.RotationOrderXYZ,
                )
                xform.SetScale((float(obs["size_x"]), float(obs["size_y"]), float(obs["size_z"])))
                bind_material(cube.GetPrim(), obstacle_material)

        drone_prims: dict[int, str] = {}
        drone_tracks: dict[int, dict] = {}
        drone_radius = float(
            scenario.get("metadata", {}).get("drone_radius", max(0.05, args.drone_scale / 2.0))
        )

        for drone in scenario.get("drones", []):
            drone_id = int(drone["id"])
            prim_path = f"/World/Scenario/Drones/Drone_{drone_id:03d}"

            if args.drone_shape == "cube":
                geom = UsdGeom.Cube.Define(stage, prim_path)
                geom.CreateSizeAttr(1.0)
                xform = UsdGeom.XformCommonAPI(geom)
                xform.SetScale((args.drone_scale, args.drone_scale, args.drone_scale * 0.35))
                prim = geom.GetPrim()
            elif args.drone_shape == "capsule":
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
                "yaw": yaw_from_velocity(vel[0], 0.0),
            }

            start_pos = pos[0]
            xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim_path))
            xform.SetTranslate((float(start_pos[0]), float(start_pos[1]), float(start_pos[2])))

        world.reset()

        viewport = omni.kit.viewport.utility.get_active_viewport()
        if viewport is not None:
            viewport.set_camera_position("/OmniverseKit_Persp", *args.camera_eye, True)
            viewport.set_camera_target("/OmniverseKit_Persp", *args.camera_lookat, True)

        if args.record and viewport is None:
            print("Recording disabled: no active viewport is available.")
            args.record = False

        captured_frames: list[np.ndarray] = []
        capture_times: list[float] = []
        video_path = None
        if args.record:
            video_path = (
                Path(args.video_output).expanduser().resolve()
                if args.video_output
                else default_video_output_path(scenario_path)
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            run_async(viewport.wait_for_rendered_frames())

        if args.save_stage:
            save_path = str(Path(args.save_stage).expanduser().resolve())
            omni.usd.get_context().save_as_stage(save_path)
            print(f"Saved stage to {save_path}")

        duration = scenario_duration(scenario.get("drones", []))

        print(f"Loaded scenario: {scenario_path}")
        print(
            f"Obstacles: {len(scenario.get('obstacles', []))} "
            f"({len(global_cloud_world)} voxels)"
        )
        print(f"Drones: {len(drone_tracks)}")
        print(f"Playback duration: {duration:.2f}s @ speed {args.speed:.2f}x")
        if args.record and video_path is not None:
            print(f"Recording replay video to {video_path}")

        world.step(render=True)
        if args.record and viewport is not None:
            frame = capture_rgb_frame(viewport, omni.kit.viewport.utility)
            if frame is not None:
                captured_frames.append(frame)
                capture_times.append(0.0)

        start_wall = time.perf_counter()
        render_step = 0
        while simulation_app.is_running():
            elapsed = (time.perf_counter() - start_wall) * args.speed
            sim_t = elapsed % duration if args.loop and duration > 1e-6 else min(elapsed, duration)

            for drone_id, track in drone_tracks.items():
                pos = lerp_positions(track["timestamps"], track["positions"], sim_t)
                vel = lerp_positions(track["timestamps"], track["velocities"], sim_t)
                track["yaw"] = yaw_from_velocity(vel, track["yaw"])
                yaw_deg = math.degrees(track["yaw"])
                xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(drone_prims[drone_id]))
                xform.SetTranslate((float(pos[0]), float(pos[1]), float(pos[2])))
                xform.SetRotate((0.0, 0.0, yaw_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)

            world.step(render=True)
            render_step += 1

            should_capture = (
                args.record
                and viewport is not None
                and render_step % max(1, args.record_every) == 0
                and (not args.loop or elapsed <= duration)
            )
            if should_capture:
                frame = capture_rgb_frame(viewport, omni.kit.viewport.utility)
                if frame is not None:
                    captured_frames.append(frame)
                    capture_times.append(float(sim_t))

            if not args.loop and elapsed >= duration:
                for _ in range(60):
                    if not simulation_app.is_running():
                        break
                    world.step(render=True)
                break

        if captured_frames and video_path is not None:
            frames = np.stack(captured_frames, axis=0).astype(np.uint8, copy=False)
            fps = estimate_video_fps(capture_times)
            write_video(video_path, frames, fps=fps)
            print(f"Saved replay video to {video_path}")

            if wandb_run is not None:
                import wandb

                wandb_run.log(
                    {
                        "scenario_duration": duration,
                        "num_drones": len(drone_tracks),
                        "num_obstacles": len(global_cloud_world)
                        if global_cloud_world
                        else len(scenario.get("obstacles", [])),
                        args.video_tag: wandb.Video(str(video_path), fps=fps, format="mp4"),
                    }
                )
        elif wandb_run is not None:
            wandb_run.log(
                {
                    "scenario_duration": duration,
                    "num_drones": len(drone_tracks),
                    "num_obstacles": len(global_cloud_world)
                    if global_cloud_world
                    else len(scenario.get("obstacles", [])),
                }
            )

    finally:
        try:
            if wandb_run is not None:
                wandb_run.finish()
        finally:
            simulation_app.close()


if __name__ == "__main__":
    main()
