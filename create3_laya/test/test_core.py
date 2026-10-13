"""Unit tests for the ROS-free core: sensors, world, planner, pilot. Run with
`python -m unittest discover -s create3_laya/test -t create3_laya` or `colcon test`."""

import math
import os
import unittest

from create3_laya import create3
from create3_laya.laya_client import LayaError, top_confidence
from create3_laya.pilot import (ACTIONS, LayaPilot, Snapshot, forward_limit, make_state,
                                rule_action, rule_detour)
from create3_laya.planner import GridPlanner, Route
from create3_laya.tasks import load_places
from create3_laya.world import World

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
MAZE = os.path.join(REPO, 'sim2d', 'worlds', 'maze.sdf')
PLACES = os.path.join(os.path.dirname(HERE), 'config', 'maze_places.json')
CLEAR = {label: 'clear' for label in [s.label for s in create3.IR_SENSORS]}


def sensors(**levels):
    out = dict(CLEAR)
    out.update({k.replace('_', ' '): v for k, v in levels.items()})
    return out


class SensorModel(unittest.TestCase):
    def test_intensity_round_trip(self):
        for d in (0.03, 0.08, 0.15):
            self.assertAlmostEqual(create3.intensity_to_range(create3.range_to_intensity(d)), d,
                                   delta=0.002)

    def test_no_detection_reads_max_range(self):
        self.assertEqual(create3.range_to_intensity(1.0), create3.range_to_intensity(0.2))
        self.assertGreaterEqual(create3.intensity_to_range(15), 0.19)

    def test_bumper_zones(self):
        self.assertEqual(create3.bumper_zone(0.0), 'bump_front_center')
        self.assertEqual(create3.bumper_zone(math.radians(80)), 'bump_left')
        self.assertIsNone(create3.bumper_zone(math.pi))

    def test_ir_from_lidar_points(self):
        # A wall across the robot 0.25 m ahead of its centre: 0.10 m from the front sensor.
        wall = [(0.25, y / 100) for y in range(-30, 31)]
        readings = create3.ir_from_points(wall)
        self.assertAlmostEqual(create3.intensity_to_range(readings['front_center_left']), 0.096,
                               delta=0.005)
        self.assertGreater(readings['front_left'], 0)       # sees it at an angle, farther off
        self.assertEqual(readings['side_left'], 0)          # looks sideways, past it
        self.assertEqual(set(create3.ir_from_points([]).values()), {0})
        # Nothing in view must read as clear, not as a missing sensor (which counts as a wall).
        self.assertEqual(forward_limit({n: create3.intensity_to_range(v) for n, v in
                                        create3.ir_from_points([]).items()}),
                         create3.MAX_WHEEL_SPEED)

    def test_ir_frame_names(self):
        self.assertEqual(create3.ir_sensor_for_frame('ir_intensity_front_left'), 'front_left')
        self.assertEqual(create3.ir_sensor_for_frame('robot/ir_intensity_side_left'), 'side_left')
        self.assertIsNone(create3.ir_sensor_for_frame(''))


class WorldAndPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World.from_sdf(MAZE)
        cls.planner = GridPlanner(cls.world)

    def test_maze_loads(self):
        self.assertEqual(len(self.world.boxes), 41)
        self.assertGreater(self.world.clearance(0.0, 0.0)[0], 0.5)

    def test_every_place_is_reachable_with_clearance(self):
        for name, place in load_places(PLACES).items():
            path = self.planner.plan((0.0, 0.0), place['xy'])
            self.assertIsNotNone(path, name)
            worst = min(self.world.clearance(x, y)[0] for x, y in path[1:-1])
            self.assertGreaterEqual(worst, self.planner.min_clearance - 0.1, name)

    def test_route_lookahead_has_line_of_sight(self):
        route = Route(self.planner, (7.0, -2.0))
        target = route.target(0.0, 0.0)
        self.assertTrue(self.planner.line_of_sight((0.0, 0.0), target))
        self.assertLess(math.hypot(*target), 0.9)
        self.assertGreater(route.remaining(0.0, 0.0), 8.0)


class RulePolicy(unittest.TestCase):
    def test_bumper_backs_up(self):
        state = make_state('straight ahead', 'far (over 4 m)', CLEAR, bumps=['front'])
        self.assertEqual(rule_action(state), 'back_up')

    def test_open_road(self):
        self.assertEqual(rule_action(make_state('straight ahead', 'far (over 4 m)', CLEAR)),
                         'forward')
        self.assertEqual(rule_action(make_state('ahead, slightly left', 'far (over 4 m)', CLEAR)),
                         'veer_left')
        self.assertEqual(rule_action(make_state('behind, on the right', 'far (over 4 m)', CLEAR)),
                         'turn_right')

    def test_blocked_front_turns_to_open_side(self):
        state = make_state('straight ahead', 'far (over 4 m)',
                           sensors(front='near', front_left='near', left='near'))
        self.assertEqual(rule_action(state), 'turn_right')
        self.assertEqual(rule_detour(state), 'right')

    def test_detour_plan_keeps_obstacle_beside(self):
        state = make_state('ahead, slightly right', 'far (over 4 m)',
                           sensors(right='far', far_right='far'), detour='left')
        self.assertEqual(rule_action(state), 'forward')

    def test_rule_labels_are_valid_actions(self):
        for direction in ('straight ahead', 'directly behind', 'to the left'):
            self.assertIn(rule_action(make_state(direction, 'far (over 4 m)', CLEAR)), ACTIONS)


class FakeClient:
    def __init__(self, answers=None, error=None):
        self.answers, self.error = answers, error

    def decide(self, state, questions):
        if self.error:
            raise self.error
        return {'answers': self.answers}


def snapshot(t=0.0, **ranges):
    ir = {s.name: create3.range_to_intensity(ranges.get(s.name, 0.2)) for s in create3.IR_SENSORS}
    return Snapshot(t, 0.0, 0.0, 0.0, ir)


class PilotRouting(unittest.TestCase):
    def test_confident_answer_is_used(self):
        pilot = LayaPilot((3.0, 0.0), driver='laya')
        probs = {a: 0.01 for a in ACTIONS}
        probs['creep'] = 1 - 0.01 * (len(ACTIONS) - 1)
        d = pilot.decide(snapshot(), FakeClient({'action': {'choice': 'creep',
                                                            'probabilities': probs}}))
        self.assertEqual((d.action, d.source), ('creep', 'laya'))

    def test_low_confidence_falls_back_to_rules(self):
        pilot = LayaPilot((3.0, 0.0), driver='laya', min_confidence=0.3)
        probs = {a: 1 / len(ACTIONS) for a in ACTIONS}
        d = pilot.decide(snapshot(), FakeClient({'action': {'choice': 'back_up',
                                                            'probabilities': probs}}))
        self.assertEqual((d.action, d.source), ('forward', 'fallback:low_confidence'))

    def test_transient_error_falls_back_and_fatal_error_raises(self):
        pilot = LayaPilot((3.0, 0.0), driver='laya')
        d = pilot.decide(snapshot(), FakeClient(error=LayaError('busy', 529, retryable=True)))
        self.assertEqual(d.source, 'fallback:error')
        with self.assertRaises(LayaError):
            pilot.decide(snapshot(), FakeClient(error=LayaError('bad request', 422)))

    def test_answer_for_an_old_goal_is_discarded(self):
        pilot = LayaPilot((3.0, 0.0), driver='laya')
        query = pilot.prepare(snapshot())
        pilot.set_goal((0.0, 3.0))
        self.assertIsNone(pilot.apply(query, 0.1, {'answers': {}}))

    def test_top_confidence(self):
        self.assertAlmostEqual(top_confidence({'a': 0.5, 'b': 0.5}), 0.0)
        self.assertAlmostEqual(top_confidence({'a': 1.0, 'b': 0.0, 'c': 0.0}), 1.0)


class SafetyLayer(unittest.TestCase):
    def test_forward_speed_shrinks_near_obstacles(self):
        far = {s.name: 0.2 for s in create3.IR_SENSORS}
        self.assertEqual(forward_limit(far), create3.MAX_WHEEL_SPEED)
        self.assertEqual(forward_limit({**far, 'front_center_left': 0.03}), 0.0)
        self.assertEqual(forward_limit({**far, 'side_left': 0.03}, w=0.0), create3.MAX_WHEEL_SPEED)
        self.assertEqual(forward_limit({**far, 'side_left': 0.03}, w=0.5), 0.0)

    def test_safety_overrides_forward_into_wall(self):
        pilot = LayaPilot((3.0, 0.0))
        pilot.decide_locally(snapshot())
        self.assertEqual(pilot.decision.action, 'forward')
        cmd = pilot.command(snapshot(0.05, front_center_left=0.03, front_left=0.04,
                                     front_center_right=0.04))
        self.assertEqual(cmd.source, 'safety')
        self.assertEqual(cmd.v, 0.0)
        self.assertNotEqual(cmd.w, 0.0)

    def test_bump_triggers_reflex(self):
        pilot = LayaPilot((3.0, 0.0))
        pilot.decide_locally(snapshot())
        bumped = Snapshot(0.1, 0.0, 0.0, 0.0, snapshot().ir, frozenset(['bump_front_center']))
        self.assertEqual(pilot.command(bumped).source, 'reflex')
        self.assertTrue(pilot.wants_decision)


if __name__ == '__main__':
    unittest.main()
