#!/usr/bin/python3
"""
scenario_data.json -> RViz visualization node
=====================================================
Published topics:
  /scenario/obstacles      - MarkerArray  static box obstacles
  /scenario/start_goals    - MarkerArray  start points (green), goals (red), links
  /scenario/traj_full      - MarkerArray  full planned paths (one color per drone)
  /scenario/drone_pos      - MarkerArray  current simulated positions (animated spheres)
  /scenario/drone_trail    - MarkerArray  flight history trails

Usage:
  # Make sure ROS has been sourced first:
  rosrun primitive_planner viz_scenario.py _json:=/path/to/scenario_data.json

  # Or launch through roslaunch:
  roslaunch primitive_planner viz_scenario.launch json:=/path/to/scenario_data.json speed:=1.0
"""

import rospy
import json
import math
import bisect
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header

# Predefined per-drone colors (RGBA, reused cyclically)
DRONE_COLORS = [
    (0.12, 0.47, 0.71, 1.0),   # blue
    (1.00, 0.50, 0.05, 1.0),   # orange
    (0.17, 0.63, 0.17, 1.0),   # green
    (0.84, 0.15, 0.16, 1.0),   # red
    (0.58, 0.40, 0.74, 1.0),   # purple
    (0.55, 0.34, 0.29, 1.0),   # brown
    (0.89, 0.47, 0.76, 1.0),   # pink
    (0.50, 0.50, 0.50, 1.0),   # gray
    (0.74, 0.74, 0.13, 1.0),   # yellow-green
    (0.09, 0.75, 0.81, 1.0),   # cyan
]


def make_color(r, g, b, a=1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = r, g, b, a
    return c


def make_point(x, y, z) -> Point:
    p = Point()
    p.x, p.y, p.z = x, y, z
    return p


def make_header(frame_id="world") -> Header:
    h = Header()
    h.stamp = rospy.Time.now()
    h.frame_id = frame_id
    return h


def set_yaw(marker: Marker, yaw: float) -> None:
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = math.sin(yaw / 2.0)
    marker.pose.orientation.w = math.cos(yaw / 2.0)


# ------------------------------------------------------------------
# Static marker builders
# ------------------------------------------------------------------

def build_obstacle_markers(obstacles: list) -> MarkerArray:
    """Obstacles as semi-transparent gray boxes."""
    ma = MarkerArray()
    for i, obs in enumerate(obstacles):
        m = Marker()
        m.header = make_header()
        m.ns = "obstacles"
        m.id = i
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position = make_point(obs["x"], obs["y"], obs["z"])
        set_yaw(m, obs.get("yaw", 0.0))
        m.scale.x = obs["size_x"]
        m.scale.y = obs["size_y"]
        m.scale.z = obs["size_z"]
        m.color = make_color(0.55, 0.55, 0.55, 0.70)
        m.lifetime = rospy.Duration(0)
        ma.markers.append(m)
    return ma


def build_start_goal_markers(drones: list) -> MarkerArray:
    """Start markers, goal markers, and connecting lines."""
    ma = MarkerArray()
    mid = 0
    for drone in drones:
        if drone.get("trajectory") is None:
            continue
        did = drone["id"]
        color = DRONE_COLORS[did % len(DRONE_COLORS)]

        sx, sy, sz = drone["start"]
        gx, gy, gz = drone["goal"]

        # Start sphere
        ms = Marker()
        ms.header = make_header()
        ms.ns = "starts"
        ms.id = mid; mid += 1
        ms.type = Marker.SPHERE
        ms.action = Marker.ADD
        ms.pose.position = make_point(sx, sy, sz)
        ms.pose.orientation.w = 1.0
        ms.scale.x = ms.scale.y = ms.scale.z = 0.4
        ms.color = make_color(0.1, 0.9, 0.1, 1.0)
        ms.lifetime = rospy.Duration(0)
        ma.markers.append(ms)

        # Goal sphere
        mg = Marker()
        mg.header = make_header()
        mg.ns = "goals"
        mg.id = mid; mid += 1
        mg.type = Marker.SPHERE
        mg.action = Marker.ADD
        mg.pose.position = make_point(gx, gy, gz)
        mg.pose.orientation.w = 1.0
        mg.scale.x = mg.scale.y = mg.scale.z = 0.4
        mg.color = make_color(0.9, 0.1, 0.1, 1.0)
        mg.lifetime = rospy.Duration(0)
        ma.markers.append(mg)

        # Start-goal link (drawn as a line strip)
        ml = Marker()
        ml.header = make_header()
        ml.ns = "sg_lines"
        ml.id = mid; mid += 1
        ml.type = Marker.LINE_STRIP
        ml.action = Marker.ADD
        ml.pose.orientation.w = 1.0
        ml.scale.x = 0.05
        ml.color = make_color(*color[:3], 0.4)
        ml.points = [make_point(sx, sy, sz), make_point(gx, gy, gz)]
        ml.lifetime = rospy.Duration(0)
        ma.markers.append(ml)

        # Drone label above the start point
        mt = Marker()
        mt.header = make_header()
        mt.ns = "drone_labels"
        mt.id = mid; mid += 1
        mt.type = Marker.TEXT_VIEW_FACING
        mt.action = Marker.ADD
        mt.pose.position = make_point(sx, sy, sz + 0.6)
        mt.pose.orientation.w = 1.0
        mt.scale.z = 0.5
        mt.color = make_color(1.0, 1.0, 1.0, 1.0)
        mt.text = f"UAV{did}"
        mt.lifetime = rospy.Duration(0)
        ma.markers.append(mt)

    return ma


def build_traj_full_markers(drones: list) -> MarkerArray:
    """Full planned trajectory as one colored polyline per drone."""
    ma = MarkerArray()
    for drone in drones:
        traj = drone.get("trajectory")
        if traj is None:
            continue
        did = drone["id"]
        color = DRONE_COLORS[did % len(DRONE_COLORS)]

        positions = traj["positions"]

        m = Marker()
        m.header = make_header()
        m.ns = "traj_full"
        m.id = did
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.06
        m.color = make_color(*color[:3], 0.85)
        m.lifetime = rospy.Duration(0)

        # Downsample to keep marker size manageable.
        step = max(1, len(positions) // 500)
        m.points = [make_point(p[0], p[1], p[2])
                    for p in positions[::step]]
        ma.markers.append(m)

    return ma


# ------------------------------------------------------------------
# Dynamic markers for playback
# ------------------------------------------------------------------

def build_drone_pos_markers(drones: list, indices: list) -> MarkerArray:
    """
    Current simulated drone positions as spheres.
    indices[i] is the current trajectory index for drone i.
    """
    ma = MarkerArray()
    for drone in drones:
        traj = drone.get("trajectory")
        if traj is None:
            continue
        did = drone["id"]
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        idx = min(indices[did], len(traj["positions"]) - 1)
        pos = traj["positions"][idx]

        m = Marker()
        m.header = make_header()
        m.ns = "drone_pos"
        m.id = did
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = make_point(*pos)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.30
        m.color = make_color(*color)
        m.lifetime = rospy.Duration(0.15)  # Auto-expire to avoid ghosting.
        ma.markers.append(m)

    return ma


def build_drone_trail_markers(drones: list, indices: list,
                               trail_len: int = 200) -> MarkerArray:
    """Flight history trail as a fading line strip."""
    ma = MarkerArray()
    for drone in drones:
        traj = drone.get("trajectory")
        if traj is None:
            continue
        did = drone["id"]
        color = DRONE_COLORS[did % len(DRONE_COLORS)]
        idx = min(indices[did], len(traj["positions"]) - 1)

        start_idx = max(0, idx - trail_len)
        positions = traj["positions"][start_idx:idx + 1]
        if len(positions) < 2:
            continue

        m = Marker()
        m.header = make_header()
        m.ns = "drone_trail"
        m.id = did
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.08
        m.lifetime = rospy.Duration(0.15)

        step = max(1, len(positions) // 100)
        pts = positions[::step]
        n = len(pts)
        for j, p in enumerate(pts):
            m.points.append(make_point(p[0], p[1], p[2]))
            alpha = 0.1 + 0.85 * (j / max(n - 1, 1))
            m.colors.append(make_color(*color[:3], alpha))

        ma.markers.append(m)

    return ma


# ------------------------------------------------------------------
# Main logic
# ------------------------------------------------------------------

def main():
    rospy.init_node("viz_scenario", anonymous=False)

    json_path = rospy.get_param("~json", "")
    speed     = float(rospy.get_param("~speed", 1.0))   # Playback speed multiplier.
    loop      = rospy.get_param("~loop", True)          # Replay after reaching the end.
    trail_len = int(rospy.get_param("~trail_len", 300)) # Trail length in trajectory points.

    if not json_path:
        rospy.logerr("[viz_scenario] Please provide the JSON file path via the _json argument")
        rospy.logerr("  Example: rosrun primitive_planner viz_scenario.py _json:=/path/to/scenario_data.json")
        return

    rospy.loginfo(f"[viz_scenario] Loading {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    obstacles = data["obstacles"]
    drones    = data["drones"]
    meta      = data["metadata"]
    dt        = meta["dt"]  # Trajectory sample step, usually 0.01 s.
    n_drones  = len(drones)

    rospy.loginfo(f"[viz_scenario] {n_drones} drones, {len(obstacles)} obstacles, "
                  f"playback speed {speed}x, {'looping' if loop else 'single pass'}")

    # Publishers
    pub_obs   = rospy.Publisher("/scenario/obstacles",   MarkerArray, queue_size=1, latch=True)
    pub_sg    = rospy.Publisher("/scenario/start_goals", MarkerArray, queue_size=1, latch=True)
    pub_traj  = rospy.Publisher("/scenario/traj_full",   MarkerArray, queue_size=1, latch=True)
    pub_pos   = rospy.Publisher("/scenario/drone_pos",   MarkerArray, queue_size=1)
    pub_trail = rospy.Publisher("/scenario/drone_trail", MarkerArray, queue_size=1)

    rospy.sleep(0.5)  # Wait for RViz subscriptions.

    # Publish static scene content once using latched topics.
    pub_obs.publish(build_obstacle_markers(obstacles))
    pub_sg.publish(build_start_goal_markers(drones))
    pub_traj.publish(build_traj_full_markers(drones))
    rospy.loginfo("[viz_scenario] Published static scene markers (obstacles, starts/goals, full trajectories)")

    # Compute the max trajectory index for each drone.
    max_indices = []
    traj_timestamps = []
    for drone in drones:
        traj = drone.get("trajectory")
        if traj:
            max_indices.append(len(traj["positions"]) - 1)
            traj_timestamps.append(traj.get("timestamps", []))
        else:
            max_indices.append(0)
            traj_timestamps.append([])

    # Playback loop
    publish_rate = 50.0  # Hz animation refresh rate.
    rate = rospy.Rate(publish_rate)
    if speed <= 0.0:
        rospy.logwarn("[viz_scenario] speed <= 0, falling back to 1.0x")
        speed = 1.0

    rospy.loginfo("[viz_scenario] Starting playback driven by trajectory timestamps")

    while not rospy.is_shutdown():
        sim_time = 0.0
        while not rospy.is_shutdown():
            cur_indices = []
            for i in range(n_drones):
                ts = traj_timestamps[i]
                if not ts:
                    cur_indices.append(0)
                    continue
                idx = bisect.bisect_right(ts, sim_time) - 1
                idx = max(0, min(idx, max_indices[i]))
                cur_indices.append(idx)

            pub_pos.publish(build_drone_pos_markers(drones, cur_indices))
            pub_trail.publish(build_drone_trail_markers(drones, cur_indices, trail_len))

            # All drones have reached the end.
            if all(cur_indices[i] >= max_indices[i] for i in range(n_drones)):
                rospy.loginfo("[viz_scenario] All drones reached the goal")
                # Keep the final pose visible for 2 seconds.
                for _ in range(int(2.0 * publish_rate)):
                    if rospy.is_shutdown():
                        break
                    pub_pos.publish(build_drone_pos_markers(drones, cur_indices))
                    pub_trail.publish(build_drone_trail_markers(
                        drones, cur_indices, trail_len))
                    rate.sleep()
                break

            sim_time += speed / publish_rate
            rate.sleep()

        if not loop or rospy.is_shutdown():
            break
        rospy.loginfo("[viz_scenario] Restarting playback ...")
        rospy.sleep(1.0)

    rospy.loginfo("[viz_scenario] Exiting")


if __name__ == "__main__":
    main()
