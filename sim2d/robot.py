"""Kinematic Create 3 for the 2D world: diff drive, the 7 IR intensity sensors, and the bumper.

Sensor poses, ranges and the intensity model come from create3_sim (see create3_laya/create3.py),
so the pilot sees the same kind of readings it gets from the Gazebo robot.
"""

import math

from create3_laya import create3
from create3_laya.pilot import Snapshot, wrap

IR_REACH = math.hypot(0.154, 0) + create3.IR_MAX_RANGE + 0.02


class SimCreate3:
    def __init__(self, world, x=0.0, y=0.0, yaw=0.0):
        self.world = world
        self.x, self.y, self.yaw = x, y, yaw
        self.bumps = frozenset()
        self.collisions = 0
        self.odometer = 0.0

    def sense(self, t):
        return Snapshot(t, self.x, self.y, self.yaw, self.read_ir(), self.bumps)

    def read_ir(self):
        nearby = self.world.near(self.x, self.y, IR_REACH)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        readings = {}
        for sensor in create3.IR_SENSORS:
            sx = self.x + c * sensor.x - s * sensor.y
            sy = self.y + s * sensor.x + c * sensor.y
            best = create3.IR_MAX_RANGE
            if nearby:
                for k in range(create3.IR_RAYS):
                    angle = (self.yaw + sensor.yaw - create3.IR_FOV / 2
                             + k * create3.IR_FOV / (create3.IR_RAYS - 1))
                    hit = self.world.raycast(sx, sy, angle, create3.IR_MAX_RANGE, nearby)
                    if hit is not None:
                        best = min(best, hit)
            readings[sensor.name] = create3.range_to_intensity(best)
        return readings

    def step(self, v, w, dt):
        half = create3.WHEEL_SEPARATION / 2
        left, right = v - w * half, v + w * half
        scale = max(1.0, abs(left) / create3.MAX_WHEEL_SPEED, abs(right) / create3.MAX_WHEEL_SPEED)
        left, right = left / scale, right / scale
        v, w = (left + right) / 2, (right - left) / create3.WHEEL_SEPARATION

        heading = self.yaw + w * dt / 2
        nx, ny = self.x + v * math.cos(heading) * dt, self.y + v * math.sin(heading) * dt
        self.yaw = wrap(self.yaw + w * dt)
        nearby = self.world.near(nx, ny, create3.ROBOT_RADIUS + 0.01)
        gap, contact = self.world.clearance(nx, ny, nearby)
        if gap < create3.ROBOT_RADIUS:
            # Blocked: stay put. Pushing into something with the front half presses the bumper.
            self.collisions += 1
            angle = wrap(math.atan2(contact[1] - self.y, contact[0] - self.x) - self.yaw)
            zone = create3.bumper_zone(angle) if v > 0 else None
            self.bumps = frozenset([zone]) if zone else frozenset()
            return
        self.odometer += math.hypot(nx - self.x, ny - self.y)
        self.x, self.y = nx, ny
        self.bumps = frozenset()
