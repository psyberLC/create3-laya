"""Global route planning over a known world, done in code.

The model never plans routes; it makes the local driving decision toward the next waypoint
(and handles whatever the map does not show, such as the dock or another robot). A* runs on
a grid whose cells carry their clearance to the nearest wall: cells the robot cannot fit in
are blocked, and cells near walls cost more, so routes keep to the middle of corridors.
"""

import heapq
import math

SQRT2 = math.sqrt(2.0)
NEIGHBORS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


class GridPlanner:
    def __init__(self, world, resolution=0.1, min_clearance=0.24, comfort=0.6, margin=0.3):
        self.res = resolution
        self.min_clearance = min_clearance
        self.comfort = comfort
        boxes = world.boxes
        self.x0 = min(b.aabb[0] for b in boxes) - margin
        self.y0 = min(b.aabb[1] for b in boxes) - margin
        self.nx = int(math.ceil((max(b.aabb[2] for b in boxes) + margin - self.x0) / resolution))
        self.ny = int(math.ceil((max(b.aabb[3] for b in boxes) + margin - self.y0) / resolution))
        self.clearance = self._clearance(boxes)

    # -- grid helpers ----------------------------------------------------------------

    def cell(self, x, y):
        ix = min(max(int((x - self.x0) / self.res), 0), self.nx - 1)
        iy = min(max(int((y - self.y0) / self.res), 0), self.ny - 1)
        return ix, iy

    def center(self, ix, iy):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def clearance_at(self, x, y):
        ix, iy = self.cell(x, y)
        return self.clearance[iy * self.nx + ix]

    def _clearance(self, boxes):
        """Distance from each cell to the nearest occupied cell (two-pass chamfer transform)."""
        nx, ny, res = self.nx, self.ny, self.res
        dist = [math.inf] * (nx * ny)
        for b in boxes:
            ix0, iy0 = self.cell(b.aabb[0], b.aabb[1])
            ix1, iy1 = self.cell(b.aabb[2], b.aabb[3])
            for iy in range(iy0, iy1 + 1):
                for ix in range(ix0, ix1 + 1):
                    if b.closest(*self.center(ix, iy))[0] <= res * 0.5:
                        dist[iy * nx + ix] = 0.0
        forward = ((-1, 0, 1.0), (0, -1, 1.0), (-1, -1, SQRT2), (1, -1, SQRT2))
        backward = ((1, 0, 1.0), (0, 1, 1.0), (1, 1, SQRT2), (-1, 1, SQRT2))
        for passes, ys, xs in ((forward, range(ny), range(nx)),
                               (backward, range(ny - 1, -1, -1), range(nx - 1, -1, -1))):
            for iy in ys:
                for ix in xs:
                    i = iy * nx + ix
                    best = dist[i]
                    for dx, dy, w in passes:
                        jx, jy = ix + dx, iy + dy
                        if 0 <= jx < nx and 0 <= jy < ny:
                            d = dist[jy * nx + jx] + w
                            if d < best:
                                best = d
                    dist[i] = best
        # Cell-center distance in metres, minus half a cell for the occupied cell's extent.
        return [max(0.0, d * res - res * 0.5) for d in dist]

    def free(self, ix, iy):
        return self.clearance[iy * self.nx + ix] >= self.min_clearance

    def _nearest_free(self, ix, iy, radius=8):
        best, best_d = None, math.inf
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < self.nx and 0 <= jy < self.ny and self.free(jx, jy):
                    d = dx * dx + dy * dy
                    if d < best_d:
                        best, best_d = (jx, jy), d
        return best

    # -- planning --------------------------------------------------------------------

    def plan(self, start, goal):
        """A* from start to goal (world x, y). Returns a list of points, or None."""
        s, g = self.cell(*start), self.cell(*goal)
        if not self.free(*s):
            s = self._nearest_free(*s)
        if not self.free(*g):
            g = self._nearest_free(*g)
        if s is None or g is None:
            return None
        nx = self.nx
        start_i, goal_i = s[1] * nx + s[0], g[1] * nx + g[0]
        cost = {start_i: 0.0}
        parent = {start_i: None}
        frontier = [(0.0, start_i)]
        while frontier:
            _, i = heapq.heappop(frontier)
            if i == goal_i:
                break
            ix, iy = i % nx, i // nx
            for dx, dy, w in NEIGHBORS:
                jx, jy = ix + dx, iy + dy
                if not (0 <= jx < nx and 0 <= jy < self.ny) or not self.free(jx, jy):
                    continue
                j = jy * nx + jx
                near_wall = max(0.0, self.comfort - self.clearance[j]) / self.comfort
                c = cost[i] + w * (1.0 + 4.0 * near_wall)
                if c < cost.get(j, math.inf):
                    cost[j] = c
                    parent[j] = i
                    h = math.hypot(jx - g[0], jy - g[1])
                    heapq.heappush(frontier, (c + h, j))
        if goal_i not in parent:
            return None
        cells, i = [], goal_i
        while i is not None:
            cells.append(self.center(i % nx, i // nx))
            i = parent[i]
        cells.reverse()
        return [tuple(start)] + cells[1:-1] + [tuple(goal)]

    def line_of_sight(self, a, b, slack=0.8):
        steps = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res * 0.5)))
        for k in range(steps + 1):
            t = k / steps
            if self.clearance_at(a[0] + (b[0] - a[0]) * t,
                                 a[1] + (b[1] - a[1]) * t) < self.min_clearance * slack:
                return False
        return True


class Route:
    """A planned path plus the bookkeeping to follow it: lookahead target and remaining length."""

    def __init__(self, planner, goal, lookahead=0.8, max_offset=0.6):
        self.planner = planner
        self.goal = goal
        self.lookahead = lookahead
        self.max_offset = max_offset
        self.path = None
        self.cumulative = []
        self.index = 0

    def replan(self, x, y):
        self.path = self.planner.plan((x, y), self.goal)
        self.index = 0
        self.cumulative = [0.0]
        for a, b in zip(self.path or [], (self.path or [])[1:]):
            self.cumulative.append(self.cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        return self.path is not None

    def _nearest(self, x, y):
        window = range(self.index, min(len(self.path), self.index + 40))
        i = min(window, key=lambda k: (self.path[k][0] - x) ** 2 + (self.path[k][1] - y) ** 2)
        return i, math.hypot(self.path[i][0] - x, self.path[i][1] - y)

    def target(self, x, y):
        """Lookahead point to steer for, replanning when the robot has strayed from the path."""
        if self.path is None and not self.replan(x, y):
            return self.goal
        i, offset = self._nearest(x, y)
        if offset > self.max_offset and self.replan(x, y):
            i, offset = self._nearest(x, y)
        self.index = i
        best = self.path[min(i + 1, len(self.path) - 1)]
        for point in self.path[i:]:
            if math.hypot(point[0] - x, point[1] - y) > self.lookahead:
                break
            if self.planner.line_of_sight((x, y), point):
                best = point
        return best

    def remaining(self, x, y):
        if self.path is None:
            return math.hypot(self.goal[0] - x, self.goal[1] - y)
        p = self.path[self.index]
        return math.hypot(p[0] - x, p[1] - y) + self.cumulative[-1] - self.cumulative[self.index]
