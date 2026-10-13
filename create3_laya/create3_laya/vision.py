"""Seeing for robots that have a camera (TurtleBot 4's OAK-D): detection client, geometry,
memory of what was seen where, and the 'find' behavior. No ROS imports, so it is testable.

Laya reads text, not images. The local OWLv2 server (vision/serve_vision.py) turns a frame
into boxes for text queries such as "a pallet"; code turns boxes into bearings and ranges
using the camera intrinsics and the aligned depth image, then into map positions.
"""

import base64
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .pilot import Command, wrap

DEFAULT_VISION_URL = 'http://127.0.0.1:8001'

# What the robot can be asked to find: Laya picks one (or 'other'), the detector gets the
# phrases. Kept to 10 options including 'other' (Laya's calibration covers up to 10).
OBJECTS = {
    'pallet': ('Pallets, or boxes stacked on pallets.', ['a pallet', 'boxes stacked on a pallet']),
    'shelf': ('Shelves or storage racks.', ['a shelf', 'a storage rack']),
    'box': ('Boxes, crates or cartons.', ['a cardboard box', 'a crate']),
    'forklift': ('A forklift.', ['a forklift']),
    'stairs': ('Stairs or a staircase.', ['stairs']),
    'door': ('A door or doorway.', ['a door']),
    'pillar': ('A pillar or column.', ['a pillar']),
    'barrel': ('Barrels or drums.', ['a barrel']),
    'person': ('A person.', ['a person']),
    'other': ('Some other object, not in this list.', []),
}


class VisionError(RuntimeError):
    pass


class VisionClient:
    def __init__(self, url=None, timeout=3.0):
        self.url = (url or DEFAULT_VISION_URL).rstrip('/') + '/v1/detect'
        self.timeout = timeout

    def detect(self, image_bytes, queries, threshold=0.2):
        body = json.dumps({'image': base64.b64encode(image_bytes).decode(),
                           'queries': list(queries), 'threshold': threshold}).encode()
        request = urllib.request.Request(self.url, data=body, method='POST',
                                         headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            raise VisionError(f'vision server at {self.url}: {e}')


@dataclass
class Camera:
    """Pinhole intrinsics from sensor_msgs/CameraInfo (fx, cx) plus the image width."""
    fx: float
    cx: float
    width: int

    def bearing(self, u):
        """Horizontal pixel -> bearing in rad, left positive (camera looks along +x)."""
        return math.atan2(self.cx - u, self.fx)


def box_range(depth, box, max_range=8.0):
    """Median depth (m) in the central half of a box. `depth` is a row-major list of rows, or
    anything indexable as depth[v][u]; NaN, inf and zero mean no return."""
    x0, y0, x1, y1 = box
    u0, u1 = int(x0 + (x1 - x0) * 0.25), int(x1 - (x1 - x0) * 0.25)
    v0, v1 = int(y0 + (y1 - y0) * 0.25), int(y1 - (y1 - y0) * 0.25)
    values = []
    for v in range(max(0, v0), max(v0 + 1, v1)):
        row = depth[v]
        for u in range(max(0, u0), max(u0 + 1, u1)):
            d = float(row[u])
            if math.isfinite(d) and 0.05 < d < max_range:
                values.append(d)
    if not values:
        return None
    values.sort()
    return values[len(values) // 2]


@dataclass
class Sighting:
    label: str
    score: float
    x: float
    y: float
    t: float
    bearing: float = 0.0
    distance: float = 0.0


def locate(pose, camera, box, depth, label, score, t):
    """A detection box -> Sighting at a map position, or None without a usable depth."""
    x, y, yaw = pose
    u = (box[0] + box[2]) / 2
    bearing = camera.bearing(u)
    distance = box_range(depth, box)
    if distance is None:
        return None
    heading = yaw + bearing
    return Sighting(label, score, x + distance * math.cos(heading),
                    y + distance * math.sin(heading), t, bearing, distance)


@dataclass
class SemanticMemory:
    """Where things were seen: each label's best sighting (a place to go back to), and its
    recent sightings (what a search can choose from without seeing them again)."""
    sightings: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)
    keep: int = 200

    def add(self, sighting):
        best = self.sightings.get(sighting.label)
        if best is None or sighting.score >= best.score:
            self.sightings[sighting.label] = sighting
        recent = self.history.setdefault(sighting.label, [])
        if not recent or recent[-1] is not sighting:
            recent.append(sighting)
            del recent[:-self.keep]

    def places(self):
        """Remembered sightings as task-interpreter places: 'seen_pallet' etc."""
        return {f'seen_{label}': {'xy': (s.x, s.y),
                                  'description': f'where the {label} was seen'}
                for label, s in self.sightings.items()}


SIDE_WORDS = {0.0: 'ahead', math.pi / 2: 'on the left', -math.pi / 2: 'on the right',
              math.pi: 'behind'}


def _groups(sightings, gap):
    """Sightings split into groups whose members chain together at most `gap` metres apart."""
    left, groups = list(sightings), []
    while left:
        group = [left.pop()]
        for p in group:                   # grows while it is walked
            near = {id(q) for q in left if math.hypot(p.x - q.x, p.y - q.y) <= gap}
            group += [q for q in left if id(q) in near]
            left = [q for q in left if id(q) not in near]
        groups.append(group)
    return groups


@dataclass
class FindBehavior:
    """Look for `label` with the camera, then drive up to the one that was asked for.

    scan: face `prefer` (the asked-for direction, relative to the heading the search started
          with: 0 ahead, +pi/2 left, -pi/2 right, pi behind; None when the text gave none),
          hold still for `settle` s so several frames come in, then keep turning if need be.
          Choose once a candidate has been in view for `settle` s, or after a full turn:
          - with `near` (a reference: "the pallet near the shelves with red barrels"), the
            candidate closest to the middle of the reference (`reference_spot`), always
            after a full turn: the middle of long shelves is known only once all of them
            have been seen (seeing their near end first put it 2 m off);
          - otherwise the candidate nearest the robot and the direction it faces: distance
            plus `turn_cost` metres per radian off it.
          An asked-for direction is strict: only candidates within `sector` of it count, so
          a close pallet behind never wins "the pallet before you". Only when a second look
          finds none there either does the best one elsewhere count, and the result says so.
          Scores only gate sightings: OWLv2's 0.27 and 0.34 for the same kind of object mean
          nothing different, and the first frame's weak far detection must not beat the near
          one in the next. A sighting at `sure_score` counts at once; a weaker one, down to
          `min_score`, once a second lands in about the same place (`same_place`): in the
          depot most pallet detections score 0.15-0.25, the crate pallet's all of them, and
          a lone false detection rarely repeats in place. Sightings from before the search
          take part too, ranked the same way; with no direction asked they end the look
          early only when in view.
    look again: a full turn without what was asked for, though it may be there: OWLv2 misses
         a pallet 4-5 m off that it finds at 3 m. Once per search, move closer and scan
         again: to `look_from` metres short of the reference ("X near Y", Y in sight but no
         X by it), or `hop_distance` further the asked-for way.
    approach: navigate to a point `standoff` metres short of it.
    hop: nothing seen, now or before: `hop_distance` towards the most open direction, then
         scan again.
    Navigation is the host's: `navigate(target, first) -> 'running' | 'done' | 'failed'`.
    """
    label: str
    navigate: object
    memory: SemanticMemory
    prefer: float = None              # the asked-for direction, None for none
    near: str = None                  # label of the reference sightings, if any
    min_score: float = 0.18           # weakest sighting that counts, once seen twice
    sure_score: float = 0.25          # a sighting this strong counts at once
    same_place: float = 1.0           # metres: two sightings this close are one object
    standoff: float = 0.8
    turn_rate: float = 0.35
    turn_cost: float = 1.5
    near_radius: float = 3.5          # a candidate this close to the reference's middle
    same_object: float = 2.0          # reference sightings chained this close are one object
    inside: float = 0.75              # a candidate this close to a reference sighting is on it
    settle: float = 2.0
    half_view: float = 0.6            # half the OAK-D's ~69 deg horizontal field of view
    sector: float = 0.9               # half-width of "ahead", "on the left": about 50 deg
    hop_distance: float = 2.5
    look_from: float = 1.5
    max_hops: int = 3
    phase: str = 'scan'
    hops: int = 0
    looked_again: bool = False
    turned: float = 0.0
    last_yaw: float = None
    target: tuple = None
    first_nav: bool = True
    start: tuple = None               # (x, y, yaw) where the search began
    hold_until: float = None          # holding still to let frames come in
    found_at: float = None            # when a good enough candidate first appeared
    candidates: list = field(default_factory=list)  # target sightings that count
    weak: list = field(default_factory=list)        # weak ones waiting for a second sighting
    references: list = field(default_factory=list)
    remembered: set = field(default_factory=set)   # ids of sightings from before this search
    chosen: Sighting = None
    spot: tuple = None                # cached reference_spot() for spot_count references
    spot_count: int = 0

    def offer(self, sighting):
        """A sighting of the target, or of the reference, from the latest frame."""
        if sighting.label == self.near and self.near is not None:
            if sighting.score >= self.sure_score:
                self.references.append(sighting)
            return
        if sighting.label != self.label:
            return
        self.memory.add(sighting)
        self._consider(sighting)

    def _consider(self, s):
        if s.score >= self.sure_score:
            self.candidates.append(s)
            return
        if s.score < self.min_score:
            return
        twins = {id(w) for w in self.weak
                 if math.hypot(w.x - s.x, w.y - s.y) <= self.same_place}
        if twins or any(math.hypot(c.x - s.x, c.y - s.y) <= self.same_place
                        for c in self.candidates):
            self.candidates += [w for w in self.weak if id(w) in twins] + [s]
            self.weak = [w for w in self.weak if id(w) not in twins]
        else:
            self.weak.append(s)

    def off(self, sighting):
        """Radians between the sighting and the asked-for direction (ahead with none), as
        seen from where the search began."""
        x, y, yaw = self.start
        return abs(wrap(math.atan2(sighting.y - y, sighting.x - x) - (yaw + (self.prefer or 0.0))))

    def cost(self, sighting):
        x, y, _ = self.start
        return math.hypot(sighting.x - x, sighting.y - y) + self.turn_cost * self.off(sighting)

    def reference_spot(self):
        """The middle of the best-supported group of reference sightings, or None. Shelves
        are long: measured to the nearest red barrel seen, the crate pallet in front of the
        depot's shelves and a pallet past their far end were both ~2.6 m off; measured to the
        middle, 2.9 m and 4.6 m."""
        if self.references and len(self.references) != self.spot_count:
            best = max(_groups(self.references, self.same_object),
                       key=lambda group: sum(r.score for r in group))
            self.spot = (sum(r.x for r in best) / len(best), sum(r.y for r in best) / len(best))
            self.spot_count = len(self.references)
        return self.spot

    def from_reference(self, sighting):
        spot = self.reference_spot()
        return math.inf if spot is None else math.hypot(sighting.x - spot[0],
                                                        sighting.y - spot[1])

    def in_view(self, sighting):
        return self.off(sighting) <= self.half_view

    def that_way(self, sighting):
        return self.prefer is None or self.off(sighting) <= self.sector

    def on_reference(self, sighting):
        """Part of the reference rather than near it: crates on the depot's shelves read as
        weak "pallets" right among the red barrels."""
        return any(math.hypot(sighting.x - r.x, sighting.y - r.y) <= self.inside
                   for r in self.references)

    def choose(self):
        pool = [s for s in self.candidates if self.that_way(s)] or self.candidates
        if self.near is not None and self.references:
            x, y, _ = self.start
            pool = [s for s in pool if not self.on_reference(s)] or pool
            return min(pool, key=lambda s: self.from_reference(s)
                       + 0.1 * math.hypot(s.x - x, s.y - y))
        return min(pool, key=self.cost)

    def good_enough(self):
        """Something to choose from: a candidate in the asked-for direction (with none asked,
        one seen now, or one remembered from before in view), near a reference when one was
        asked for."""
        ready = [s for s in self.candidates if self.that_way(s) and (
            self.prefer is not None or id(s) not in self.remembered or self.in_view(s))]
        if self.near is None:
            return bool(ready)
        return any(self.from_reference(s) <= self.near_radius and not self.on_reference(s)
                   for s in ready)

    def _recall(self):
        """What was seen before this search: a pallet the robot stands at may be too close
        for the camera to make out, but it is still the one in front."""
        for s in self.memory.history.get(self.label, []):
            self.remembered.add(id(s))
            self._consider(s)
        for s in self.memory.history.get(self.near, []) if self.near else []:
            if s.score >= self.sure_score:
                self.remembered.add(id(s))
                self.references.append(s)

    def step(self, snap, open_bearing=0.0):
        """One control tick. Returns ('running', Command), 'running', ('done', detail) or
        ('failed', detail). `open_bearing` is the most open direction (rad, robot frame)."""
        if self.start is None:
            self.start = (snap.x, snap.y, snap.yaw)
            self._recall()
        if self.phase == 'scan':
            if self.last_yaw is not None:
                self.turned += abs(wrap(snap.yaw - self.last_yaw))
            self.last_yaw = snap.yaw
            facing = (self.prefer is None or
                      abs(wrap(snap.yaw - self.start[2] - self.prefer)) <= self.half_view)
            if facing and self.hold_until is None:
                self.hold_until = snap.t + self.settle
            if self.near is None and self.good_enough() and self.found_at is None:
                self.found_at = snap.t
            full = self.turned >= 2 * math.pi
            again = self._look_again_from(snap) if full and not self.good_enough() else None
            if self.candidates and again is None and (
                    full or (self.found_at is not None
                             and snap.t - self.found_at >= self.settle)):
                self.chosen = self.choose()
                return self._approach(snap)
            if facing and snap.t < self.hold_until:
                return 'running', Command(0.0, 0.0, 'step')
            if not full:
                return 'running', Command(0.0, math.copysign(self.turn_rate, self.prefer or 1.0),
                                          'step')
            if again is not None:
                self.looked_again, self.target = True, again
            elif self.hops >= self.max_hops:
                return 'failed', f'no {self.label} seen after {self.hops} moves and scans'
            else:
                self.hops += 1
                heading = snap.yaw + open_bearing
                self.target = (snap.x + self.hop_distance * math.cos(heading),
                               snap.y + self.hop_distance * math.sin(heading))
            self.phase, self.first_nav = 'hop', True
        status = self.navigate(self.target, self.first_nav)
        self.first_nav = False
        if status == 'running':
            return 'running'
        if self.phase == 'approach':
            if status != 'done':
                return 'failed', f'could not reach the {self.label}'
            s, (x, y, _) = self.chosen, self.start
            earlier = ', seen earlier' if id(s) in self.remembered else ''
            side = (None if self.prefer is None
                    else SIDE_WORDS.get(self.prefer, f'{math.degrees(self.prefer):+.0f} deg'))
            name = f'the {self.label}' + (f' {side}' if side and self.that_way(s) else '')
            where = ''
            if self.near is not None:
                where = (f', {self.from_reference(s):.1f} m from the {self.near}'
                         if self.references else f"; I didn't see the {self.near}")
            if side and not self.that_way(s):
                where += f'; there was none {side}'
            return 'done', (f'reached {name} {math.hypot(s.x - x, s.y - y):.1f} m '
                            f'away (score {s.score:.2f}{earlier}){where}')
        # hop finished (or failed: scan from wherever we are)
        self.phase, self.turned, self.last_yaw = 'scan', 0.0, None
        self.hold_until = self.found_at = None
        return 'running'

    def _look_again_from(self, snap):
        """Where to look again from, once per search: next to the reference in sight, or
        further the asked-for way; None if neither applies."""
        if self.looked_again:
            return None
        if self.near is not None and self.references:
            return self._short_of(snap, self.reference_spot(), self.look_from)
        if self.prefer is not None:
            heading = self.start[2] + self.prefer
            return (snap.x + self.hop_distance * math.cos(heading),
                    snap.y + self.hop_distance * math.sin(heading))
        return None

    @staticmethod
    def _short_of(snap, point, gap):
        dx, dy = point[0] - snap.x, point[1] - snap.y
        distance = math.hypot(dx, dy)
        keep = max(0.0, distance - gap) / max(distance, 1e-6)
        return snap.x + dx * keep, snap.y + dy * keep

    def _approach(self, snap):
        self.target = self._short_of(snap, (self.chosen.x, self.chosen.y), self.standoff)
        self.phase, self.first_nav = 'approach', True
        return self.step(snap)
