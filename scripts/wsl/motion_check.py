#!/usr/bin/env python3
"""Measure how the robot really moves for plain-English commands (simulation ground truth).

With a simulation running (the launcher, or KEEP_OPEN=1 scripts/wsl/gazebo_test.sh ...):

  python3 scripts/wsl/motion_check.py "drive forward 1 meter" "turn left 90 degrees"

Each task is sent on /task like any other. For every step it prints where the robot started
and ended according to create3_sim's sim_ground_truth_pose: distance along the heading it had
when the step began, sideways offset, worst sideways deviation on the way, heading change.
"""

import argparse
import json
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Checker:
    def __init__(self, node):
        self.node = node
        self.track = []                   # (sim t, x, y, yaw)
        self.events = []
        self.pub = node.create_publisher(String, 'task', 10)
        node.create_subscription(Odometry, 'sim_ground_truth_pose', self._pose,
                                 qos_profile_sensor_data)
        node.create_subscription(String, 'task_status', self._status, 50)

    def _pose(self, msg):
        p, s = msg.pose.pose, msg.header.stamp
        self.track.append((s.sec + s.nanosec * 1e-9, p.position.x, p.position.y,
                           yaw_of(p.orientation)))

    def _status(self, msg):
        event = json.loads(msg.data)
        if event['event'] != 'heartbeat':
            event['at'] = len(self.track)          # index of the latest pose when it arrived
            self.events.append(event)

    def spin_until(self, done, timeout):
        end = time.time() + timeout
        while time.time() < end and not done():
            rclpy.spin_once(self.node, timeout_sec=0.1)
        return done()

    def run(self, text, timeout):
        if not self.spin_until(lambda: self.track and self.pub.get_subscription_count(), 30):
            return print('  no ground-truth pose or no task listener; is a simulation running?')
        first = len(self.events)
        self.pub.publish(String(data=text))
        if not self.spin_until(lambda: any(e['event'] == 'task' and e.get('text') == text
                                           for e in self.events[first:]), 60):
            return print('  no answer from the Laya node')
        task = next(e for e in self.events[first:] if e['event'] == 'task')
        print(f"  {task['outcome']}: {task['summary']}")
        mission = task.get('mission')
        if mission is None:
            return
        self.spin_until(lambda: any(e['event'] == 'mission' and e['mission'] == mission
                                    for e in self.events[first:]), timeout)
        steps = [e for e in self.events[first:] if e['event'] == 'step'
                 and e['mission'] == mission]
        starts = {e['step']: e for e in steps if e['status'] == 'started'}
        for e in steps:
            if e['status'] == 'started' or e['step'] not in starts:
                continue
            s = starts[e['step']]
            self.report(s.get('detail', ''), self.track[max(0, s['at'] - 1):e['at']], e)
        end = next((e for e in self.events[first:] if e['event'] == 'mission'
                    and e['mission'] == mission), None)
        print(f"  task {end['status'] if end else 'did not finish in time'}")

    @staticmethod
    def report(what, poses, end):
        if len(poses) < 2:
            return print(f'    {what}: no poses recorded')
        t0, x0, y0, th0 = poses[0]
        t1, x1, y1, th1 = poses[-1]
        c, s = math.cos(th0), math.sin(th0)

        def along(p):
            return (p[1] - x0) * c + (p[2] - y0) * s

        def side(p):
            return -(p[1] - x0) * s + (p[2] - y0) * c
        worst = max((side(p) for p in poses), key=abs)
        detail = f" ({end['detail']})" if end.get('detail') else ''
        print(f'    {what}: {end["status"]}{detail} in {t1 - t0:.1f} s sim\n'
              f'      start ({x0:+.2f}, {y0:+.2f}) heading {math.degrees(th0):+.0f} deg -> '
              f'end ({x1:+.2f}, {y1:+.2f}) heading {math.degrees(th1):+.0f} deg\n'
              f'      forward {along(poses[-1]):+.2f} m, sideways {side(poses[-1]):+.2f} m '
              f'(worst {worst:+.2f} m on the way), turned {math.degrees(wrap(th1 - th0)):+.1f} deg')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('tasks', nargs='+')
    p.add_argument('--timeout', type=float, default=300.0, help='wall seconds per task')
    args = p.parse_args()
    rclpy.init()
    node = rclpy.create_node('motion_check')
    checker = Checker(node)
    for text in args.tasks:
        print(f'>>> {text}', flush=True)
        checker.run(text, args.timeout)
        sys.stdout.flush()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
