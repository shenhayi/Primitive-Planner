#!/usr/bin/python3
"""
scenario_data.json -> RViz 可视化节点
=====================================================
发布的 topic：
  /scenario/obstacles      - MarkerArray  圆柱障碍物
  /scenario/start_goals    - MarkerArray  起点(绿)、终点(红)、连线
  /scenario/traj_full      - MarkerArray  完整规划路径（彩色线段，每架无人机一种颜色）
  /scenario/drone_pos      - MarkerArray  当前仿真位置（球体，实时动画）
  /scenario/drone_trail    - MarkerArray  飞行历史轨迹（线段）

用法：
  # 确保 ROS 已 source，然后：
  rosrun primitive_planner viz_scenario.py _json:=/path/to/scenario_data.json

  # 或通过 launch 文件：
  roslaunch primitive_planner viz_scenario.launch json:=/path/to/scenario_data.json speed:=1.0
"""

import rospy
import json
import math
import bisect
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header

# 每架无人机预定义颜色（RGBA，循环使用）
DRONE_COLORS = [
    (0.12, 0.47, 0.71, 1.0),   # 蓝
    (1.00, 0.50, 0.05, 1.0),   # 橙
    (0.17, 0.63, 0.17, 1.0),   # 绿
    (0.84, 0.15, 0.16, 1.0),   # 红
    (0.58, 0.40, 0.74, 1.0),   # 紫
    (0.55, 0.34, 0.29, 1.0),   # 棕
    (0.89, 0.47, 0.76, 1.0),   # 粉
    (0.50, 0.50, 0.50, 1.0),   # 灰
    (0.74, 0.74, 0.13, 1.0),   # 黄绿
    (0.09, 0.75, 0.81, 1.0),   # 青
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


# ──────────────────────────────────────────────────────────
# 静态 Marker 构建
# ──────────────────────────────────────────────────────────

def build_obstacle_markers(obstacles: list) -> MarkerArray:
    """障碍物 —— 半透明灰色圆柱。"""
    ma = MarkerArray()
    for i, obs in enumerate(obstacles):
        m = Marker()
        m.header = make_header()
        m.ns = "obstacles"
        m.id = i
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position = make_point(obs["x"], obs["y"], obs["height"] / 2.0)
        m.pose.orientation.w = 1.0
        m.scale.x = obs["radius"] * 2
        m.scale.y = obs["radius"] * 2
        m.scale.z = obs["height"]
        m.color = make_color(0.55, 0.55, 0.55, 0.70)
        m.lifetime = rospy.Duration(0)
        ma.markers.append(m)
    return ma


def build_start_goal_markers(drones: list) -> MarkerArray:
    """起点（绿球）、终点（红球）、连接线（虚线）。"""
    ma = MarkerArray()
    mid = 0
    for drone in drones:
        if drone.get("trajectory") is None:
            continue
        did = drone["id"]
        color = DRONE_COLORS[did % len(DRONE_COLORS)]

        sx, sy, sz = drone["start"]
        gx, gy, gz = drone["goal"]

        # 起点球
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

        # 终点球
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

        # 起终点连线（虚线用 LINE_STRIP 模拟）
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

        # 无人机编号文字（起点上方）
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
    """完整规划轨迹 —— 每架无人机一条彩色折线。"""
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

        # 降采样：每10个点取1个，减少 marker 大小
        step = max(1, len(positions) // 500)
        m.points = [make_point(p[0], p[1], p[2])
                    for p in positions[::step]]
        ma.markers.append(m)

    return ma


# ──────────────────────────────────────────────────────────
# 动态 Marker（动画播放）
# ──────────────────────────────────────────────────────────

def build_drone_pos_markers(drones: list, indices: list) -> MarkerArray:
    """
    当前仿真时刻各无人机位置 —— 球体。
    indices[i] 是第 i 架无人机当前的轨迹索引。
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
        m.lifetime = rospy.Duration(0.15)  # 自动消失，防止残影
        ma.markers.append(m)

    return ma


def build_drone_trail_markers(drones: list, indices: list,
                               trail_len: int = 200) -> MarkerArray:
    """飞行历史尾迹 —— 渐变透明线段。"""
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


# ──────────────────────────────────────────────────────────
# 主逻辑
# ──────────────────────────────────────────────────────────

def main():
    rospy.init_node("viz_scenario", anonymous=False)

    json_path = rospy.get_param("~json", "")
    speed     = float(rospy.get_param("~speed", 1.0))   # 播放速度倍率
    loop      = rospy.get_param("~loop", True)           # 播完后循环
    trail_len = int(rospy.get_param("~trail_len", 300))  # 尾迹长度（轨迹点数）

    if not json_path:
        rospy.logerr("[viz_scenario] 请通过 _json 参数指定 JSON 文件路径")
        rospy.logerr("  示例: rosrun primitive_planner viz_scenario.py _json:=/path/to/scenario_data.json")
        return

    rospy.loginfo(f"[viz_scenario] 加载 {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)

    obstacles = data["obstacles"]
    drones    = data["drones"]
    meta      = data["metadata"]
    dt        = meta["dt"]  # 轨迹时间步长（通常 0.01s）
    n_drones  = len(drones)

    rospy.loginfo(f"[viz_scenario] {n_drones} 架无人机，{len(obstacles)} 个障碍物，"
                  f"播放速度 {speed}x，{'循环' if loop else '单次'}")

    # Publishers
    pub_obs   = rospy.Publisher("/scenario/obstacles",   MarkerArray, queue_size=1, latch=True)
    pub_sg    = rospy.Publisher("/scenario/start_goals", MarkerArray, queue_size=1, latch=True)
    pub_traj  = rospy.Publisher("/scenario/traj_full",   MarkerArray, queue_size=1, latch=True)
    pub_pos   = rospy.Publisher("/scenario/drone_pos",   MarkerArray, queue_size=1)
    pub_trail = rospy.Publisher("/scenario/drone_trail", MarkerArray, queue_size=1)

    rospy.sleep(0.5)  # 等待 RViz 订阅

    # 发布静态内容（latch，只需一次）
    pub_obs.publish(build_obstacle_markers(obstacles))
    pub_sg.publish(build_start_goal_markers(drones))
    pub_traj.publish(build_traj_full_markers(drones))
    rospy.loginfo("[viz_scenario] 静态场景已发布（障碍物、起终点、完整轨迹）")

    # 计算每架无人机的最大轨迹索引
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

    # 动画播放
    publish_rate = 50.0  # Hz，动画刷新率
    rate = rospy.Rate(publish_rate)
    if speed <= 0.0:
        rospy.logwarn("[viz_scenario] speed <= 0，回退到 1.0x")
        speed = 1.0

    rospy.loginfo("[viz_scenario] 开始动画播放（按轨迹时间戳推进）")

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

            # 所有无人机均到达终点
            if all(cur_indices[i] >= max_indices[i] for i in range(n_drones)):
                rospy.loginfo("[viz_scenario] 所有无人机到达终点")
                # 保持最终位置显示 2s
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
        rospy.loginfo("[viz_scenario] 重新开始播放 ...")
        rospy.sleep(1.0)

    rospy.loginfo("[viz_scenario] 退出")


if __name__ == "__main__":
    main()
