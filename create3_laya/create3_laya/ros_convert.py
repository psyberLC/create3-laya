"""Message <-> pilot conversions for the ROS 2 node, kept free of rclpy so they can be tested
without ROS. Inputs only need the attributes the irobot_create_msgs / nav_msgs types have."""

import math

from . import create3

BACKUP_LIMIT, BUMP, CLIFF, WHEEL_DROP = 0, 1, 2, 4   # irobot_create_msgs/HazardDetection


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def odom_to_map(odom_pose, spawn):
    """Odometry starts at the spawn pose; return (x, y, yaw) in the world (map) frame."""
    sx, sy, syaw = spawn
    p = odom_pose.position
    yaw = yaw_from_quaternion(odom_pose.orientation)
    c, s = math.cos(syaw), math.sin(syaw)
    return (sx + c * p.x - s * p.y, sy + s * p.x + c * p.y,
            math.atan2(math.sin(syaw + yaw), math.cos(syaw + yaw)))


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class GyroOdometry:
    """Gyro-aided dead reckoning, as a real Create 3 does it: heading from the IMU, distance
    from the wheels. create3_sim's odometry is wheels only and over-reads turning (90 deg
    turns came out 84 deg in the depot), so its position is integrated along a heading that
    drifts; here the distance it rolled is laid along the gyro heading instead."""

    def __init__(self):
        self.pose = self.last = self.offset = None

    def update(self, odom, imu_yaw):
        """odom: wheel odometry (x, y, yaw); imu_yaw: the IMU's yaw or None.
        Returns the fused (x, y, yaw), or `odom` itself until a gyro reading arrives."""
        if imu_yaw is None and self.pose is None:
            return odom
        if self.pose is None:
            self.offset = _wrap(odom[2] - imu_yaw)        # align the gyro with odometry
            self.pose = self.last = odom
            return odom
        (x0, y0, yaw0), (x1, y1, _) = self.last, odom
        rolled = (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)
        x, y, previous = self.pose
        yaw = previous if imu_yaw is None else _wrap(imu_yaw + self.offset)
        mid = previous + _wrap(yaw - previous) / 2
        self.pose = (x + rolled * math.cos(mid), y + rolled * math.sin(mid), yaw)
        self.last = odom
        return self.pose


def transform_to_pose(transform):
    """geometry_msgs/Transform -> (x, y, yaw)."""
    return (transform.translation.x, transform.translation.y,
            yaw_from_quaternion(transform.rotation))


def ir_vector_readings(msg):
    """IrIntensityVector -> {sensor name: value}, skipping readings without a usable frame_id.

    create3_sim's Gazebo IR plugin does not set frame_id on its readings (as of the jazzy
    branch), so in simulation this usually returns nothing and the per-sensor topics are used.
    """
    readings = {}
    for reading in msg.readings:
        name = create3.ir_sensor_for_frame(reading.header.frame_id)
        if name:
            readings[name] = int(reading.value)
    return readings


def scan_points(scan, mount, near=0.6):
    """sensor_msgs/LaserScan returns closer than `near` m to the lidar, as (x, y) in the robot
    frame. `mount` is the lidar's (x, y, yaw) on the robot."""
    mx, my, myaw = mount
    points = []
    for i, r in enumerate(scan.ranges):
        if math.isfinite(r) and scan.range_min < r < near:
            a = myaw + scan.angle_min + i * scan.angle_increment
            points.append((mx + r * math.cos(a), my + r * math.sin(a)))
    return points


def hazards(msg):
    """HazardDetectionVector -> frozenset of bump zone names, plus 'cliff' and 'backup_limit'
    (the Create 3 refuses to reverse more than about 15 cm without its safety override)."""
    found = set()
    for detection in msg.detections:
        if detection.type == BUMP and detection.header.frame_id in create3.BUMP_LABELS:
            found.add(detection.header.frame_id)
        elif detection.type in (CLIFF, WHEEL_DROP):
            found.add('cliff')
        elif detection.type == BACKUP_LIMIT:
            found.add('backup_limit')
    return frozenset(found)
