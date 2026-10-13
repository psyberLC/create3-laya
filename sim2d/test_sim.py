"""End-to-end checks of the 2D simulator with the offline mock (no Laya server needed).

  python sim2d/test_sim.py
"""

import math
import sys
import unittest
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_sim import Simulation, parse_schedule  # noqa: E402


def make_args(**overrides):
    args = dict(task=[], interactive=False, goal=None, driver='code',
                driving_model='create3-driving', task_model='typed-decisions', mock=True,
                url=None, start=(0.0, 0.0, 0.0), world=str(HERE / 'worlds' / 'maze.sdf'),
                max_time=400.0, period=0.5, min_confidence=0.3, seed=3, out='', quiet=True)
    args.update(overrides)
    return Namespace(**args)


class Simulator(unittest.TestCase):
    def test_every_place_is_reached_without_collisions(self):
        sim = Simulation(make_args())
        for name, place in sim.places.items():
            with self.subTest(place=name):
                sim.drive_to(place['xy'])
                self.assertTrue(sim.run(), f'{name} timed out')
                x, y = place['xy']
                self.assertLess(math.hypot(sim.robot.x - x, sim.robot.y - y), 0.3)
        self.assertEqual(sim.robot.collisions, 0)

    def test_two_part_task_with_interrupt(self):
        sim = Simulation(make_args())
        sim.run(parse_schedule(['go to the east hall, then come back to the dock',
                                '@20 stop']))
        self.assertEqual(sim.runner.status, 'stopped')
        self.assertEqual([t['action'] for t in sim.tasks], ['started', 'stopped the robot'])

    def test_model_driver_path(self):
        sim = Simulation(make_args(driver='laya'))
        sim.run(parse_schedule(['go to the east hall']))
        self.assertEqual(sim.runner.status, 'done')
        sources = {d['source'] for d in sim.decisions}
        self.assertIn('laya', sources)
        self.assertTrue(all(s == 'laya' or s.startswith('fallback') for s in sources))

    def test_motion_primitives(self):
        sim = Simulation(make_args())
        sim.run(parse_schedule(['turn around, then drive forward 1 meter']))
        self.assertEqual(sim.runner.status, 'done')
        self.assertAlmostEqual(abs(sim.robot.yaw), math.pi, delta=0.08)
        self.assertAlmostEqual(sim.robot.x, -1.0, delta=0.05)


if __name__ == '__main__':
    unittest.main()
