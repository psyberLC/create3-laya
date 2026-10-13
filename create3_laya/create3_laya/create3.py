"""iRobot Create 3 constants and sensor models, taken from create3_sim (jazzy branch).

Sources in https://github.com/iRobotEducation/create3_sim:
  irobot_create_description/urdf/create3.urdf.xacro                 body, wheels, IR sensor poses
  irobot_create_description/urdf/sensors/ir_intensity.urdf.xacro    IR range and field of view
  irobot_create_gz_toolbox/src/sensors/ir_intensity.cpp             range -> intensity model
  irobot_create_toolbox/include/irobot_create_toolbox/sensors/bumpers.hpp   bumper zones
  irobot_create_nodes/include/irobot_create_nodes/motion_control_node.hpp  speed limit
"""

import math
from dataclasses import dataclass

# Body. The URDF body cylinder is 0.164 m; the bumper shell sits a little outside it.
ROBOT_RADIUS = 0.17
WHEEL_SEPARATION = 0.233
MAX_WHEEL_SPEED = 0.306  # SAFETY_ON_MAX_SPEED, m/s

# IR intensity sensors: 62 Hz, 10 deg FOV, 25-200 mm range, 5 rays across the FOV.
IR_MIN_RANGE = 0.025
IR_MAX_RANGE = 0.200
IR_FOV = math.radians(10.0)
IR_RAYS = 5
IR_PEAK = 3500.0


@dataclass(frozen=True)
class IrSensor:
    name: str     # create3_sim sensor name (topic suffix)
    label: str    # plain-language name used in the state Laya sees
    x: float
    y: float
    yaw: float


# Ordered left to right. create3_sim's names are not symmetric ("side_left" pairs with
# "right"), so the state Laya sees uses labels derived from the mounting angle instead.
IR_SENSORS = (
    IrSensor('side_left', 'far left', 0.0399, 0.1488, 1.309),
    IrSensor('left', 'left', 0.0990, 0.1180, 0.873),
    IrSensor('front_left', 'front left', 0.1396, 0.0651, 0.436),
    IrSensor('front_center_left', 'front', 0.1540, 0.0, 0.0),
    IrSensor('front_center_right', 'front right', 0.1396, -0.0651, -0.436),
    IrSensor('front_right', 'right', 0.0990, -0.1180, -0.873),
    IrSensor('right', 'far right', 0.0399, -0.1488, -1.309),
)
IR_BY_NAME = {s.name: s for s in IR_SENSORS}
FRONT_SENSORS = ('front_left', 'front_center_left', 'front_center_right')
LEFT_SENSORS = ('side_left', 'left', 'front_left')
RIGHT_SENSORS = ('right', 'front_right', 'front_center_right')

# Bumper zones (robot frame, radians, counter-clockwise positive) -> HazardDetection frame_id.
BUMPER_ZONES = (
    (-math.pi / 2, -3 * math.pi / 10, 'bump_right'),
    (-3 * math.pi / 10, -math.pi / 10, 'bump_front_right'),
    (-math.pi / 10, math.pi / 10, 'bump_front_center'),
    (math.pi / 10, 3 * math.pi / 10, 'bump_front_left'),
    (3 * math.pi / 10, math.pi / 2, 'bump_left'),
)
BUMP_LABELS = {
    'bump_right': 'right',
    'bump_front_right': 'front right',
    'bump_front_center': 'front',
    'bump_front_left': 'front left',
    'bump_left': 'left',
}


def range_to_intensity(distance):
    """create3_sim's IR model: 3500 * exp(-d * 2e / range_max), truncated to int16."""
    d = min(max(distance, IR_MIN_RANGE), IR_MAX_RANGE)
    return int(IR_PEAK * math.exp(-d * 2 * math.e / IR_MAX_RANGE))


def intensity_to_range(intensity):
    """Invert range_to_intensity. Returns IR_MAX_RANGE when nothing is detected."""
    if intensity <= 0:
        return IR_MAX_RANGE
    d = -math.log(min(intensity, IR_PEAK) / IR_PEAK) * IR_MAX_RANGE / (2 * math.e)
    return min(max(d, 0.0), IR_MAX_RANGE)


def ir_from_points(points):
    """IR intensities from obstacle points in the robot frame (x ahead, y left in m), for a
    robot whose IR sensors are left out of the simulation but which has a scanning lidar (the
    TurtleBot 4). Each sensor reads the nearest point inside its field of view and range, as
    create3_sim's GPU-lidar IR sensor would; nothing in view reads 0."""
    readings = {}
    for s in IR_SENSORS:
        c, si = math.cos(s.yaw), math.sin(s.yaw)
        nearest = None
        for px, py in points:
            dx, dy = px - s.x, py - s.y
            along = dx * c + dy * si
            # The cone, widened by a centimetre for the gaps between the lidar's points.
            if 0.0 < along <= IR_MAX_RANGE and \
                    abs(dy * c - dx * si) <= along * math.tan(IR_FOV / 2) + 0.01:
                nearest = along if nearest is None else min(nearest, along)
        readings[s.name] = 0 if nearest is None else range_to_intensity(nearest)
    return readings


def bumper_zone(angle):
    """Map a contact angle in the robot frame to a create3_sim bumper zone, or None."""
    for lo, hi, name in BUMPER_ZONES:
        if lo <= angle <= hi:
            return name
    return None


def ir_sensor_for_frame(frame_id):
    """Match a frame_id or topic name such as 'ir_intensity_front_left' to a sensor name."""
    tail = frame_id.rsplit('/', 1)[-1]
    if tail.startswith('ir_intensity_'):
        tail = tail[len('ir_intensity_'):]
    return tail if tail in IR_BY_NAME else None
