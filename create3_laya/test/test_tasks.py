"""Tests for prompt-to-action: clause splitting, numbers, interpretation, missions, ROS
message conversion. A scripted fake stands in for the Laya server."""

import math
import os
import unittest
from types import SimpleNamespace as NS

from create3_laya import create3, ros_convert
from create3_laya.mission import MissionRunner, Step
from create3_laya.pilot import LayaPilot, Snapshot
from create3_laya.tasks import TaskInterpreter, extract_numbers, load_places, split_clauses

PLACES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                      'maze_places.json')


def choice(pick, options, p=0.9):
    rest = (1 - p) / max(1, len(options) - 1)
    return {'choice': pick, 'probabilities': {o: (p if o == pick else rest) for o in options}}


class ScriptedLaya:
    """Answers by clause text: {text: (skill, place, direction, pace[, destination])}."""

    def __init__(self, script, confidence=0.9):
        self.script, self.p = script, confidence

    def decide(self, state, questions):
        text = state['instruction']
        if 'timing' in questions:
            pick = 'after' if 'after' in text or 'done' in text else 'now'
            return {'answers': {'timing': choice(pick, questions['timing']['criteria'])}}
        skill, place, direction, pace, *rest = self.script[text]
        destination = rest[0] if rest else ('yes' if place else 'no')
        motion = rest[1] if len(rest) > 1 else ('turn' if skill == 'rotate' else 'move')
        target = rest[2] if len(rest) > 2 else 'other'
        q = questions
        return {'answers': {
            'object': choice(target, q['object']['criteria']),
            'handles': choice('yes' if skill == 'unsupported' else 'no',
                              q['handles']['criteria']),
            'destination': choice(destination, q['destination']['criteria']),
            'motion': choice(motion, q['motion']['criteria']),
            'skill': choice(skill, q['skill']['criteria'], self.p),
            'place': choice(place or 'no_place', q['place']['criteria'], self.p),
            'direction': choice(direction or 'unspecified', q['direction']['criteria']),
            'pace': choice(pace or 'normal', q['pace']['criteria']),
        }}


class Language(unittest.TestCase):
    def test_split_on_joiners_but_not_decimals_or_lead_ins(self):
        self.assertEqual(split_clauses('turn around, then drive forward 1.5 m, wait 2 seconds'),
                         ['turn around', 'drive forward 1.5 m', 'wait 2 seconds'])
        self.assertEqual(split_clauses('actually, go to the west room instead'),
                         ['actually, go to the west room instead'])
        self.assertEqual(split_clauses('go to the east hall and come back'),
                         ['go to the east hall', 'come back'])

    def test_numbers_in_robot_units(self):
        self.assertEqual(extract_numbers('reverse 30 cm')['metres'], 0.3)
        self.assertEqual(extract_numbers('spin right 45 degrees')['degrees'], 45.0)
        self.assertEqual(extract_numbers('turn around')['degrees'], 180.0)
        self.assertEqual(extract_numbers('wait 2 minutes')['seconds'], 120.0)
        self.assertEqual(extract_numbers('explore three rooms')['count'], 3)


class Interpreter(unittest.TestCase):
    def setUp(self):
        self.places = load_places(PLACES)

    def interpreter(self, script, confidence=0.9):
        return TaskInterpreter(self.places, ScriptedLaya(script, confidence), seed=0)

    def test_multi_step_task(self):
        plan = self.interpreter({
            'go to the east hall': ('go_to', 'east_hall', None, None),
            'come back to the dock': ('return_to_dock', None, None, None),
        }).interpret('go to the east hall, then come back to the dock')
        self.assertTrue(plan.runnable)
        self.assertEqual([s.kind for s in plan.steps], ['go_to', 'go_to', 'face', 'dock'])
        self.assertEqual(plan.steps[2].target, (0.2, 0.0))
        self.assertEqual(plan.steps[0].target, self.places['east_hall']['xy'])

    def test_odometry_moves(self):
        interpreter = self.interpreter({
            'go 10 meters forward': ('drive', None, 'forward', None, 'no', 'move'),
            'move 2 m to the left': ('drive', None, 'left', None, 'no', 'move'),
            'back up 10 cm': ('drive', None, 'backward', None, 'no', 'move'),
            'go 2 meters backward': ('drive', None, 'backward', None, 'no', 'move')})

        def plan(text):
            return [(s.kind, round(s.amount, 3)) for s in interpreter.interpret(text).steps]
        self.assertEqual(plan('go 10 meters forward'), [('drive', 10.0)])
        self.assertEqual(plan('move 2 m to the left'),
                         [('rotate', round(math.pi / 2, 3)), ('drive', 2.0)])
        self.assertEqual(plan('back up 10 cm'), [('drive', -0.1)])
        # Past the Create 3's reverse limit: turn around and drive.
        self.assertEqual(plan('go 2 meters backward'),
                         [('rotate', round(math.pi, 3)), ('drive', 2.0)])

    def test_a_distance_settles_an_unsure_drive_or_turn(self):
        unsure = TaskInterpreter(self.places, ScriptedLaya({
            'back up 10 centimeters': ('drive', None, 'backward', None, 'no', 'move'),
            'spin a bit': ('drive', None, None, None, 'no', 'turn')}, confidence=0.15))
        plan = unsure.interpret('back up 10 centimeters')
        self.assertTrue(plan.runnable)
        self.assertEqual(plan.clauses[0].resolved,
                         'a distance or an angle in the text says drive or turn')
        self.assertIsNotNone(unsure.interpret('spin a bit').clarification)

    def test_find_uses_object_answer_or_phrase(self):
        interpreter = self.interpreter({
            'go to where the pallets are stored': ('go_to', None, None, None, 'yes', 'move',
                                                   'pallet'),
            'find the red barrel': ('find', None, None, None, 'yes', 'move', 'other')})
        plan = interpreter.interpret('go to where the pallets are stored')
        step = plan.steps[0]
        self.assertEqual((step.kind, step.place), ('find', 'pallet'))
        self.assertIn('a pallet', step.queries)
        self.assertEqual(plan.clauses[0].resolved, 'no known place, but an object to look for')
        step = interpreter.interpret('find the red barrel').steps[0]
        self.assertEqual((step.place, step.queries), ('red barrel', ('a red barrel',)))

    def test_named_object_and_side_decide_what_to_find(self):
        interpreter = self.interpreter({
            # Laya took "pallet" for a forklift here; the word in the text settles it.
            'drive to the pallet before you': ('find', None, 'forward', None, 'yes', 'move',
                                               'forklift'),
            'find the forklift on your left': ('find', None, 'left', None, 'yes', 'move',
                                               'forklift')})
        step = interpreter.interpret('drive to the pallet before you').steps[0]
        self.assertEqual((step.place, step.bearing), ('pallet', 0.0))
        self.assertEqual(step.describe(), 'find the pallet ahead')
        step = interpreter.interpret('find the forklift on your left').steps[0]
        self.assertEqual((step.place, step.bearing), ('forklift', math.pi / 2))
        self.assertEqual(step.describe(), 'find the forklift on the left')

    def test_sides_of_the_robot_versus_of_a_reference(self):
        find = ('find', None, None, None, 'yes', 'move', 'pallet')
        phrases = {  # text -> (bearing, reference)
            'go to the pallet': (None, None),
            'go to the pallet right in front of you': (0.0, None),
            'go to the pallet ahead': (0.0, None),
            'go to the pallet behind you': (math.pi, None),
            'go to the pallet behind the shelves': (None, 'shelves'),
            'go to the pallet in front of the door': (None, 'door'),
        }
        interpreter = self.interpreter({text: find for text in phrases})
        for text, expected in phrases.items():
            step = interpreter.interpret(text).steps[0]
            self.assertEqual((step.place, step.bearing, step.near), ('pallet',) + expected, text)
        # Nothing named before the relation: the object after it is where to go.
        step = self.interpreter({'go behind the shelves': (
            'find', None, None, None, 'yes', 'move', 'barrel')}).interpret(
            'go behind the shelves').steps[0]
        self.assertEqual((step.place, step.near), ('shelf', None))

    def test_x_near_y_keeps_the_target_and_names_the_reference(self):
        # Laya's object answer may pick the reference ("barrel"); the text's order decides.
        step = self.interpreter({
            'drive to the pallet near the shelves with red barrels': (
                'find', None, None, None, 'yes', 'move', 'barrel'),
        }).interpret('drive to the pallet near the shelves with red barrels').steps[0]
        self.assertEqual((step.place, step.near), ('pallet', 'shelves with red barrels'))
        self.assertEqual(step.near_queries, ('shelves with red barrels', 'red barrels'))
        self.assertEqual(step.describe(), 'find the pallet near the shelves with red barrels')

    def test_explore_without_places_wanders_by_odometry(self):
        interpreter = TaskInterpreter({'start': {'xy': (0, 0), 'description': 'start'}},
                                      ScriptedLaya({'explore': ('explore', None, None, None)}),
                                      seed=0)
        steps = interpreter.interpret('explore').steps
        self.assertEqual({s.kind for s in steps}, {'go_relative'})
        self.assertEqual(len(steps), 3)

    def test_unsure_skill_defers_to_a_sure_place(self):
        class UnsureSkill(ScriptedLaya):
            def decide(self, state, questions):
                response = super().decide(state, questions)
                answers = response['answers']
                if 'skill' in answers:
                    answers['skill'] = choice(answers['skill']['choice'],
                                              questions['skill']['criteria'], p=0.2)
                return response

        depot = {'start': {'xy': (1.0, 2.0), 'description': 'start'},
                 'dock': {'xy': (0.4, 2.0), 'face': (1.2, 2.0), 'description': 'dock'}}
        interpreter = TaskInterpreter(depot, UnsureSkill({
            'come back to where you started': ('return_to_dock', 'start', None, None, 'no'),
            'head back to the charger': ('stop', 'dock', None, None, 'no'),
            'do the thing': ('stop', None, None, None, 'no')}), seed=0)
        plan = interpreter.interpret('come back to where you started')
        self.assertEqual([(s.kind, s.place) for s in plan.steps], [('go_to', 'start')])
        self.assertEqual(plan.clauses[0].resolved,
                         'unsure of the action, but sure of the place, so go there')
        plan = interpreter.interpret('head back to the charger')
        self.assertEqual([s.kind for s in plan.steps], ['go_to', 'face', 'dock'])
        self.assertIsNotNone(interpreter.interpret('do the thing').clarification)

    def test_place_options_stay_within_calibration(self):
        places = dict(self.places, start={'xy': (0, 0), 'description': 'start'})
        offered = TaskInterpreter(places, None).offered_places()
        self.assertEqual(len(offered), 9)
        self.assertNotIn('start', offered)
        self.assertTrue(offered['dock'].endswith('where the robot started'))
        depot = {'start': {'xy': (0, 0), 'description': 'start'},
                 'dock': {'xy': (0, 0), 'description': 'dock'}}
        depot.update({f'seen_{i}': {'xy': (i, 0), 'description': f'seen {i}'} for i in range(9)})
        offered = TaskInterpreter(depot, None).offered_places()
        self.assertEqual(len(offered), 9)
        self.assertIn('dock', offered)
        self.assertNotIn('seen_0', offered)       # oldest sighting goes first
        self.assertIn('seen_8', offered)

    def test_back_to_back_dock_requests_dock_once(self):
        plan = self.interpreter({
            'go back to the charging station': ('return_to_dock', None, None, None),
            'dock': ('return_to_dock', None, None, None),
        }).interpret('go back to the charging station and dock')
        self.assertEqual([s.kind for s in plan.steps], ['go_to', 'face', 'dock'])

    def test_rotate_uses_direction_and_number(self):
        plan = self.interpreter({'spin right 45 degrees': ('rotate', None, 'right', None)}) \
            .interpret('spin right 45 degrees')
        self.assertAlmostEqual(plan.steps[0].amount, -math.radians(45))

    def test_destination_answer_overrides_a_turn(self):
        plan = self.interpreter({
            'take me to the bottom left corner': ('rotate', 'south_west_corner', 'left', None,
                                                  'yes')}).interpret(
            'take me to the bottom left corner')
        self.assertEqual(plan.steps[0].kind, 'go_to')
        self.assertEqual(plan.clauses[0].resolved, 'names a destination place, so go there')

    def test_backward_rotate_becomes_drive(self):
        interpreter = self.interpreter({'back up a little': ('rotate', None, 'backward', None,
                                                             'no', 'move')})
        plan = interpreter.interpret('back up a little')
        self.assertEqual(plan.steps[0].kind, 'drive')
        self.assertLess(plan.steps[0].amount, 0)
        interpreter = self.interpreter({'turn around': ('rotate', None, 'backward', None,
                                                        'no', 'move')})
        self.assertEqual(interpreter.interpret('turn around').steps[0].kind, 'rotate')

    def test_unsupported_is_refused(self):
        plan = self.interpreter({'pick up the cup': ('unsupported', None, None, None)},
                                confidence=0.2).interpret('pick up the cup')
        self.assertFalse(plan.runnable)
        self.assertIn("can't", plan.refusal)

    def test_unsure_asks_instead_of_acting(self):
        plan = self.interpreter({'go somewhere': ('go_to', 'east_hall', None, None)},
                                confidence=0.15).interpret('go somewhere')
        self.assertFalse(plan.runnable)
        self.assertIsNotNone(plan.clarification)

    def test_timing_when_busy(self):
        script = {'when you are done, go to the south room': ('go_to', 'south_room', None, None)}
        plan = self.interpreter(script).interpret('when you are done, go to the south room',
                                                  busy=True)
        self.assertEqual(plan.timing, 'after')


class Missions(unittest.TestCase):
    def test_rotate_then_drive_then_wait(self):
        pilot = LayaPilot()
        runner = MissionRunner(pilot)
        x = y = yaw = t = 0.0
        runner.start([Step('rotate', amount=math.pi / 2), Step('drive', amount=0.5),
                      Step('wait', amount=1.0)], Snapshot(t, x, y, yaw, {}))
        ir = {}
        while runner.active and t < 30:
            cmd = runner.command(Snapshot(t, x, y, yaw, ir))
            yaw += cmd.w * 0.02
            x += cmd.v * math.cos(yaw) * 0.02
            y += cmd.v * math.sin(yaw) * 0.02
            t += 0.02
        self.assertEqual(runner.status, 'done')
        self.assertAlmostEqual(yaw, math.pi / 2, delta=0.05)
        self.assertAlmostEqual(y, 0.5, delta=0.03)

    def test_drive_holds_its_line(self):
        runner = MissionRunner(LayaPilot())
        x = y = yaw = t = 0.0
        clear = {s.name: create3.range_to_intensity(1.0) for s in create3.IR_SENSORS}
        runner.start([Step('drive', amount=3.0)], Snapshot(t, x, y, yaw, clear))
        while runner.active and t < 60:
            cmd = runner.command(Snapshot(t, x, y, yaw, clear))
            yaw += (cmd.w + 0.05) * 0.02          # one wheel slips: the robot veers left
            x += cmd.v * math.cos(yaw) * 0.02
            y += cmd.v * math.sin(yaw) * 0.02
            t += 0.02
        self.assertEqual(runner.status, 'done')
        self.assertAlmostEqual(x, 3.0, delta=0.02)
        self.assertLess(abs(y), 0.03)
        self.assertLess(t, 20.0)

    def test_relative_goal_is_fixed_when_the_step_starts(self):
        runner = MissionRunner(LayaPilot())
        runner.start([Step('go_relative', offset=(10.0, 2.0))],
                     Snapshot(0, 1.0, 1.0, math.pi / 2, {}))
        x, y = runner.steps[0].target
        self.assertAlmostEqual(x, -1.0)
        self.assertAlmostEqual(y, 11.0)
        self.assertTrue(runner.needs_decisions())

    def test_backup_limit_ends_a_reverse_without_a_reflex(self):
        runner = MissionRunner(LayaPilot())
        runner.start([Step('drive', amount=-0.4)], Snapshot(0, 0, 0, 0, {}))
        cmd = runner.command(Snapshot(0.1, -0.05, 0, 0, {}, frozenset({'backup_limit'})))
        self.assertEqual(runner.status, 'done')
        self.assertEqual(cmd.source, 'step')
        self.assertIn('15 cm', runner.events[-1].detail)

    def test_external_action_handler(self):
        calls = []
        runner = MissionRunner(LayaPilot(), action_handlers={
            'dock': lambda step, snap, first: calls.append(first) or (
                'done' if len(calls) > 2 else 'running')})
        runner.start([Step('dock')], Snapshot(0, 0, 0, 0, {}))
        self.assertEqual(runner.command(Snapshot(0.0, 0, 0, 0, {})).source, 'external')
        for i in range(1, 5):
            runner.command(Snapshot(i * 0.1, 0, 0, 0, {}))
        self.assertEqual(runner.status, 'done')
        self.assertEqual(calls[0], True)

    def test_interrupt_hook_and_handled_go_steps(self):
        interrupted = []
        runner = MissionRunner(LayaPilot(), on_interrupt=interrupted.append,
                               action_handlers={'go_to': lambda step, snap, first: 'running'})
        runner.start([Step('go_to', target=(5.0, 0.0))], Snapshot(0, 0, 0, 0, {}))
        self.assertFalse(runner.needs_decisions())
        runner.cancel(Snapshot(1, 0, 0, 0, {}), 'stop')
        self.assertEqual([s.kind for s in interrupted], ['go_to'])

    def test_find_without_camera_fails_clearly(self):
        runner = MissionRunner(LayaPilot())
        runner.start([Step('find', place='pallet')], Snapshot(0, 0, 0, 0, {}))
        runner.command(Snapshot(0.1, 0, 0, 0, {}))
        self.assertEqual(runner.status, 'failed')
        self.assertIn('camera', runner.events[-1].detail)

    def test_face_turns_toward_a_point(self):
        runner = MissionRunner(LayaPilot())
        yaw, t = math.pi, 0.0                      # facing away from the dock at +x
        runner.start([Step('face', target=(1.0, 0.0))], Snapshot(t, 0.0, 0.0, yaw, {}))
        while runner.active and t < 20:
            yaw += runner.command(Snapshot(t, 0.0, 0.0, yaw, {})).w * 0.02
            t += 0.02
        self.assertEqual(runner.status, 'done')
        self.assertLess(abs(math.atan2(math.sin(yaw), math.cos(yaw))), 0.06)

    def test_replacing_a_mission_is_logged(self):
        runner = MissionRunner(LayaPilot())
        runner.start([Step('wait', amount=5)], Snapshot(0, 0, 0, 0, {}))
        runner.start([Step('wait', amount=1)], Snapshot(1, 0, 0, 0, {}))
        self.assertIn('replaced', [e.status for e in runner.events])
        self.assertEqual(runner.mission, 2)


class RosConversions(unittest.TestCase):
    def test_odom_to_map_applies_spawn_pose(self):
        pose = NS(position=NS(x=1.0, y=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
        x, y, yaw = ros_convert.odom_to_map(pose, (2.0, 3.0, math.pi / 2))
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 4.0)
        self.assertAlmostEqual(yaw, math.pi / 2)

    def test_gyro_odometry_follows_the_true_heading(self):
        # Drive a 2 m square. The wheels over-read every turn by 7% (as in the depot), so raw
        # odometry ends far from the start; the gyro heading keeps the fused pose on it.
        fused = ros_convert.GyroOdometry()
        true = [0.0, 0.0, 0.0]
        wheel = [0.0, 0.0, 0.0]

        def tick(forward, turn):
            true[2] += turn
            wheel[2] += turn * 1.07
            for pose in (true, wheel):
                pose[0] += forward * math.cos(pose[2])
                pose[1] += forward * math.sin(pose[2])
            return fused.update(tuple(wheel), true[2] + 0.3)   # gyro has its own zero
        fused.update((0.0, 0.0, 0.0), 0.3)
        for _ in range(4):
            for _ in range(100):
                tick(0.02, 0.0)
            for _ in range(50):
                pose = tick(0.0, math.pi / 100)
        self.assertGreater(math.hypot(wheel[0], wheel[1]), 0.4)
        self.assertLess(math.hypot(pose[0], pose[1]), 0.02)
        self.assertAlmostEqual(math.cos(pose[2]), 1.0, places=4)

    def test_hazards_and_ir_vector(self):
        msg = NS(detections=[NS(type=1, header=NS(frame_id='bump_left')),
                             NS(type=2, header=NS(frame_id='base_link')),
                             NS(type=0, header=NS(frame_id='base_link'))])
        self.assertEqual(ros_convert.hazards(msg),
                         frozenset({'bump_left', 'cliff', 'backup_limit'}))
        vec = NS(readings=[NS(header=NS(frame_id='ir_intensity_front_left'), value=900),
                           NS(header=NS(frame_id=''), value=12)])
        self.assertEqual(ros_convert.ir_vector_readings(vec), {'front_left': 900})

    def test_scan_points_in_the_robot_frame(self):
        # The TurtleBot 4's lidar: 4 cm back, turned 90 deg left, so scan angle -90 deg is ahead.
        n = 8
        ranges = [float('inf')] * n
        ranges[2] = 0.30                     # angle_min + 2 * pi/4 = -pi/2 in the scan frame
        ranges[6] = 5.0                      # beyond `near`
        scan = NS(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / n,
                  range_min=0.164)
        points = ros_convert.scan_points(scan, (-0.04, 0.0, math.pi / 2))
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0][0], 0.26)
        self.assertAlmostEqual(points[0][1], 0.0)


if __name__ == '__main__':
    unittest.main()
