"""Tests for the camera side: geometry, memory, and the find behavior."""

import math
import unittest

from create3_laya.pilot import Snapshot
from create3_laya.vision import Camera, FindBehavior, SemanticMemory, Sighting, box_range, locate


def flat_depth(width, height, value):
    return [[value] * width for _ in range(height)]


class Geometry(unittest.TestCase):
    def test_bearing_is_left_positive(self):
        cam = Camera(fx=200.0, cx=160.0, width=320)
        self.assertAlmostEqual(cam.bearing(160), 0.0)
        self.assertGreater(cam.bearing(60), 0.0)      # left half of the image
        self.assertAlmostEqual(cam.bearing(360), -math.atan2(200, 200))

    def test_box_range_ignores_missing_returns(self):
        depth = flat_depth(40, 40, 2.5)
        depth[20][20] = float('nan')
        depth[21][21] = 0.0
        self.assertAlmostEqual(box_range(depth, (10, 10, 30, 30)), 2.5)
        self.assertIsNone(box_range(flat_depth(40, 40, float('inf')), (10, 10, 30, 30)))

    def test_locate_places_sighting_in_map(self):
        cam = Camera(fx=200.0, cx=160.0, width=320)
        s = locate((1.0, 2.0, math.pi / 2), cam, (150, 100, 170, 140),
                   flat_depth(320, 240, 3.0), 'pallet', 0.6, 5.0)
        self.assertAlmostEqual(s.x, 1.0, places=5)
        self.assertAlmostEqual(s.y, 5.0, places=5)


class Memory(unittest.TestCase):
    def test_keeps_best_sighting_and_offers_places(self):
        memory = SemanticMemory()
        memory.add(Sighting('pallet', 0.3, 1.0, 1.0, 0))
        memory.add(Sighting('pallet', 0.6, 4.0, 2.0, 1))
        memory.add(Sighting('pallet', 0.4, 9.0, 9.0, 2))
        self.assertEqual(memory.places()['seen_pallet']['xy'], (4.0, 2.0))


def pallet(score, x, y):
    return Sighting('pallet', score, x, y, 0.0, distance=math.hypot(x, y))


class Find(unittest.TestCase):
    def run_find(self, offers=(), memory=None, prefer=None, nav_result='done', near=None):
        """offers: {tick: [sightings]}, ticks of 0.05 s. Returns (result, navigation calls,
        turn rates)."""
        calls, turns = [], []

        def navigate(target, first):
            calls.append((target, first))
            return nav_result if len(calls) > 2 else 'running'

        find = FindBehavior('pallet', navigate, memory or SemanticMemory(), prefer=prefer,
                            near=near, max_hops=1)
        offers = dict(offers)
        yaw, t, result = 0.0, 0.0, None
        for i in range(2000):
            for s in offers.get(i, []):
                find.offer(s)
            result = find.step(Snapshot(t, 0.0, 0.0, yaw, {}))
            if isinstance(result, tuple) and result[0] in ('done', 'failed'):
                break
            if isinstance(result, tuple):
                turns.append(result[1].w)
                yaw += result[1].w * 0.05
            t += 0.05
        return result, calls, turns

    def test_approaches_with_standoff_when_seen(self):
        result, calls, _ = self.run_find({10: [pallet(0.5, 3.0, 0.0)]})
        self.assertEqual(result[0], 'done')
        target, first = calls[0]
        self.assertTrue(first)
        self.assertAlmostEqual(target[0], 2.2, places=5)

    def test_the_one_in_front_beats_a_better_score_behind(self):
        memory = SemanticMemory()
        memory.add(pallet(0.6, -2.0, 0.0))         # seen earlier, behind the robot
        result, calls, _ = self.run_find({3: [pallet(0.6, -2.0, 0.0), pallet(0.3, 3.0, 0.5)]},
                                         memory=memory)
        self.assertEqual(result[0], 'done')
        self.assertGreater(calls[0][0][0], 1.5)     # heading for the one ahead
        self.assertIn('3.0 m away', result[1])

    def test_on_the_left_looks_left_before_choosing(self):
        result, calls, turns = self.run_find(
            {1: [pallet(0.6, 3.0, 0.0)], 40: [pallet(0.3, 0.0, 3.0)]}, prefer=math.pi / 2)
        self.assertGreater(turns[0], 0)              # turns left first
        self.assertGreater(calls[0][0][1], 1.5)     # and goes to the one on the left
        self.assertLess(abs(calls[0][0][0]), 0.5)

    def test_before_you_never_takes_a_closer_one_behind(self):
        # Turned around at a pallet: the one now behind is close, the one ahead 4.5 m off.
        memory = SemanticMemory()
        memory.add(pallet(0.3, -0.3, 1.2))
        offers = {5: [pallet(0.3, 4.5, 0.3)]}
        result, calls, _ = self.run_find(offers, memory=memory, prefer=0.0)
        self.assertGreater(calls[0][0][0], 3.0)
        self.assertIn('the pallet ahead 4.5 m away', result[1])
        # With no direction asked, the close one wins.
        memory = SemanticMemory()
        memory.add(pallet(0.3, -0.3, 1.2))
        _, calls, _ = self.run_find(offers, memory=memory)
        self.assertLess(calls[0][0][0], 0.5)

    def test_looks_again_further_that_way_before_settling_for_another(self):
        # Turned around at a pallet: none ahead within the camera's reach at first.
        memory = SemanticMemory()
        memory.add(pallet(0.3, -0.3, 1.2))
        result, calls, _ = self.run_find({450: [pallet(0.3, 4.5, 0.3)]}, memory=memory,
                                         prefer=0.0)
        self.assertAlmostEqual(calls[0][0][0], 2.5)     # moved on ahead and looked again
        self.assertAlmostEqual(calls[0][0][1], 0.0)
        self.assertGreater(calls[-1][0][0], 3.0)
        self.assertIn('the pallet ahead', result[1])

    def test_none_on_that_side_takes_another_and_says_so(self):
        result, calls, turns = self.run_find({1: [pallet(0.5, 3.0, 0.0)]}, prefer=math.pi / 2)
        self.assertEqual(result[0], 'done')
        self.assertAlmostEqual(calls[0][0][1], 2.5)     # looked again further left
        self.assertGreater(calls[-1][0][0], 1.5)
        self.assertGreater(sum(abs(w) for w in turns) * 0.05, 12.0)  # two looks all round
        self.assertIn('there was none on the left', result[1])

    def test_a_weak_first_glimpse_does_not_decide(self):
        # The first frame finds a faint pallet 5 m off; the next frames a clear one at 2 m.
        result, calls, turns = self.run_find({2: [pallet(0.26, 5.0, 0.5)],
                                              20: [pallet(0.42, 2.0, -0.3)]})
        self.assertLess(calls[0][0][0], 1.5)        # standing off the 2 m one
        self.assertEqual(set(turns[:20]), {0.0})    # holding still while frames come in

    def test_x_near_y_takes_the_one_by_the_reference(self):
        shelves = Sighting('shelves', 0.4, 4.5, 4.5, 0.0, distance=6.4)
        result, calls, _ = self.run_find(
            {2: [pallet(0.5, 2.0, 0.0), pallet(0.3, 4.0, 3.8), shelves]}, near='shelves')
        self.assertGreater(calls[0][0][1], 2.5)     # the pallet by the shelves, not the near one
        self.assertIn('from the shelves', result[1])

    def test_x_near_y_looks_again_from_by_the_reference(self):
        # The shelves are in sight, the pallet by them only once the robot is closer.
        shelves = Sighting('shelves', 0.4, 4.5, 4.5, 0.0, distance=6.4)
        result, calls, _ = self.run_find(
            {2: [pallet(0.5, -3.0, 0.0), shelves], 450: [pallet(0.3, 4.0, 3.8)]},
            near='shelves')
        self.assertGreater(calls[0][0][0], 3.0)        # went over to the shelves first
        self.assertGreater(calls[-1][0][1], 2.5)       # then to the pallet by them
        self.assertIn('0.9 m from the shelves', result[1])

    def test_x_near_y_measures_to_the_middle_of_a_long_reference(self):
        # Shelves along x = 2 from y = 0 to 4, and a lone false match far off. The pallet
        # facing their middle, not the one just past their end (nearer the end's barrels).
        shelves = [Sighting('shelves', 0.4, 2.0, i * 0.5, 0.0) for i in range(9)]
        stray = Sighting('shelves', 0.3, -5.0, -5.0, 0.0)
        result, calls, _ = self.run_find(
            {2: shelves + [stray, pallet(0.3, -0.8, 2.0), pallet(0.3, 1.5, -2.0),
                           pallet(0.3, -5.5, -5.0)]}, near='shelves')
        self.assertGreater(calls[0][0][1], 1.0)
        self.assertLess(calls[0][0][0], 0.0)
        self.assertIn('2.8 m from the shelves', result[1])

    def test_x_near_y_looks_all_round_before_choosing(self):
        # The shelves' near end is seen first, next to a pallet; the rest of them later.
        offers = {2: [Sighting('shelves', 0.4, 2.0, y, 0.0) for y in (0.0, 0.5)]
                  + [pallet(0.3, 1.5, -2.0), pallet(0.3, -0.8, 3.0)],
                  200: [Sighting('shelves', 0.4, 2.0, 0.5 * i, 0.0) for i in range(2, 9)]}
        _, calls, turns = self.run_find(offers, near='shelves')
        self.assertGreater(sum(abs(w) for w in turns) * 0.05, 6.0)
        self.assertGreater(calls[0][0][1], 1.0)      # the one facing the shelves' middle

    def test_x_near_y_skips_what_is_on_y(self):
        # A crate on the shelves reads as a weak pallet right among the barrels.
        shelves = [Sighting('shelves', 0.4, 2.0, i * 0.5, 0.0) for i in range(9)]
        result, calls, _ = self.run_find(
            {2: shelves + [pallet(0.3, 2.1, 2.0), pallet(0.3, -0.8, 2.0)]}, near='shelves')
        self.assertLess(calls[-1][0][0], 0.0)        # the one in front of them
        self.assertIn('2.8 m from the shelves', result[1])

    def test_a_weak_sighting_counts_once_seen_again_in_place(self):
        result, _, _ = self.run_find({2: [pallet(0.2, 2.0, 0.0)], 10: [pallet(0.2, 4.0, 3.0)]})
        self.assertEqual(result[0], 'failed')        # two weak ones, in different places
        result, calls, _ = self.run_find({2: [pallet(0.2, 2.0, 0.0)],
                                          10: [pallet(0.21, 2.3, 0.1)]})
        self.assertEqual(result[0], 'done')
        self.assertLess(calls[0][0][0], 2.0)

    def test_x_near_y_without_seeing_y_says_so(self):
        result, _, _ = self.run_find({2: [pallet(0.5, 2.0, 0.0)]}, near='shelves')
        self.assertEqual(result[0], 'done')
        self.assertIn("didn't see the shelves", result[1])

    def test_falls_back_to_where_one_was_seen_before(self):
        memory = SemanticMemory()
        memory.add(pallet(0.5, 4.0, 4.0))          # off to the side: look around first
        result, calls, turns = self.run_find(memory=memory)
        self.assertEqual(result[0], 'done')
        self.assertIn('seen earlier', result[1])
        self.assertGreater(calls[0][0][0], 3.0)
        self.assertGreater(sum(abs(w) for w in turns) * 0.05, 6.0)   # after a full turn

    def test_one_seen_earlier_right_in_front_counts(self):
        # Standing at a pallet the camera can no longer make out, then "the pallet before you".
        memory = SemanticMemory()
        memory.add(pallet(0.25, 1.0, 0.0))
        memory.add(pallet(0.5, -4.0, 1.0))
        result, calls, turns = self.run_find({30: [pallet(0.5, -4.0, 1.0)]}, memory=memory)
        self.assertIn('1.0 m away', result[1])
        self.assertEqual(set(turns), {0.0})         # decided without turning around

    def test_scans_a_full_turn_then_hops_then_gives_up(self):
        result, calls, _ = self.run_find()
        self.assertEqual(result[0], 'failed')
        self.assertEqual(len({c[0] for c in calls}), 1)   # exactly one hop target
        self.assertIn('no pallet seen', result[1])


if __name__ == '__main__':
    unittest.main()
