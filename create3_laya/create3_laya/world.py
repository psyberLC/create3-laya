"""2D world loaded from a create3_sim SDF file: each <box> collision becomes an oriented box."""

import math
import xml.etree.ElementTree as ET


def _pose(element):
    node = element.find('pose')
    if node is None or not (node.text or '').strip():
        return 0.0, 0.0, 0.0
    values = [float(v) for v in node.text.split()]
    return values[0], values[1], values[5]


def _compose(parent, child):
    px, py, pyaw = parent
    cx, cy, cyaw = child
    c, s = math.cos(pyaw), math.sin(pyaw)
    return px + c * cx - s * cy, py + s * cx + c * cy, pyaw + cyaw


class Box:
    def __init__(self, name, x, y, yaw, size_x, size_y):
        self.name = name
        self.x, self.y, self.yaw = x, y, yaw
        self.hx, self.hy = size_x / 2, size_y / 2
        self.c, self.s = math.cos(yaw), math.sin(yaw)
        ex = abs(self.c) * self.hx + abs(self.s) * self.hy
        ey = abs(self.s) * self.hx + abs(self.c) * self.hy
        self.aabb = (x - ex, y - ey, x + ex, y + ey)

    def corners(self):
        pts = []
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, ly = sx * self.hx, sy * self.hy
            pts.append((self.x + self.c * lx - self.s * ly, self.y + self.s * lx + self.c * ly))
        return pts

    def _local(self, px, py):
        dx, dy = px - self.x, py - self.y
        return self.c * dx + self.s * dy, -self.s * dx + self.c * dy

    def closest(self, px, py):
        """Distance from a point to the box and the closest point on it (world frame)."""
        lx, ly = self._local(px, py)
        qx = min(max(lx, -self.hx), self.hx)
        qy = min(max(ly, -self.hy), self.hy)
        wx = self.x + self.c * qx - self.s * qy
        wy = self.y + self.s * qx + self.c * qy
        return math.hypot(px - wx, py - wy), (wx, wy)

    def ray(self, ox, oy, angle):
        """Distance along a ray to the box, or None. Slab test in the box frame."""
        lx, ly = self._local(ox, oy)
        a = angle - self.yaw
        dx, dy = math.cos(a), math.sin(a)
        t_near, t_far = 0.0, math.inf
        for o, d, h in ((lx, dx, self.hx), (ly, dy, self.hy)):
            if abs(d) < 1e-12:
                if abs(o) > h:
                    return None
                continue
            t1, t2 = (-h - o) / d, (h - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_near, t_far = max(t_near, t1), min(t_far, t2)
            if t_near > t_far:
                return None
        return t_near


class World:
    def __init__(self, boxes, name='world'):
        self.boxes = boxes
        self.name = name

    @classmethod
    def from_sdf(cls, path):
        root = ET.parse(path).getroot()
        world = root.find('world')
        boxes = []
        for model in world.iter('model'):
            model_pose = _pose(model)
            for link in model.findall('link'):
                link_pose = _compose(model_pose, _pose(link))
                for collision in link.findall('collision'):
                    size = collision.find('geometry/box/size')
                    if size is None:
                        continue
                    sx, sy, _ = (float(v) for v in size.text.split())
                    x, y, yaw = _compose(link_pose, _pose(collision))
                    boxes.append(Box(f"{model.get('name')}/{link.get('name')}", x, y, yaw, sx, sy))
        return cls(boxes, world.get('name', 'world'))

    def near(self, x, y, radius):
        return [b for b in self.boxes
                if b.aabb[0] - radius <= x <= b.aabb[2] + radius
                and b.aabb[1] - radius <= y <= b.aabb[3] + radius]

    def clearance(self, x, y, boxes=None):
        """Distance to the nearest box and the contact point on it."""
        best, point = math.inf, None
        for box in self.boxes if boxes is None else boxes:
            d, p = box.closest(x, y)
            if d < best:
                best, point = d, p
        return best, point

    def raycast(self, ox, oy, angle, max_range, boxes=None):
        best = None
        for box in self.boxes if boxes is None else boxes:
            t = box.ray(ox, oy, angle)
            if t is not None and t <= max_range and (best is None or t < best):
                best = t
        return best
