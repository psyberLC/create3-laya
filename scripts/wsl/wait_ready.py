#!/usr/bin/env python3
"""Wait until the Laya task node reports it can drive (rclpy only, no ROS daemon).

The node publishes a heartbeat on /task_status every 2 s with whether it has a pose and IR
readings; this exits 0 once one says both, else 1 after the timeout.

  python3 wait_ready.py [--timeout 240]
"""

import argparse
import json
import sys
import time

import rclpy
from std_msgs.msg import String


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--timeout', type=float, default=240.0)
    p.add_argument('--status', default='/task_status')
    p.add_argument('--min-ir', type=int, default=1)
    p.add_argument('--nav2', action='store_true',
                   help="also wait for Nav2's bt_navigator to be lifecycle-active")
    args = p.parse_args()

    rclpy.init()
    node = rclpy.create_node('laya_wait_ready')
    nav = nav_state = None
    if args.nav2:
        # The action server exists long before Nav2 accepts goals: bt_navigator only turns
        # lifecycle-active once SLAM's map and the costmaps are up.
        from lifecycle_msgs.srv import GetState
        nav = node.create_client(GetState, '/bt_navigator/get_state')
        nav_state = {'future': None, 'label': '', 'request': GetState.Request()}
    latest = {}
    node.create_subscription(
        String, args.status,
        lambda m: latest.update(json.loads(m.data)) if '"heartbeat"' in m.data else None, 10)
    start, said = time.time(), set()
    ok = False
    while time.time() - start < args.timeout:
        tick = time.time()
        rclpy.spin_once(node, timeout_sec=0.2)
        time.sleep(max(0.0, 0.05 - (time.time() - tick)))   # spin_once can return at once
        for key, label in (('event', 'task node'), ('pose', 'pose'), ('ir_sensors', 'IR')):
            if latest.get(key) and label not in said:
                extra = f" ({latest.get('pose_source')})" if key == 'pose' else ''
                print(f'  {label} ready after {time.time() - start:.0f} s{extra}', flush=True)
                said.add(label)
        if nav is not None and 'Nav2' not in said and nav.service_is_ready():
            future = nav_state['future']
            if future is None:
                nav_state['future'] = nav.call_async(nav_state['request'])
            elif future.done():
                nav_state['label'] = future.result().current_state.label
                nav_state['future'] = None
                if nav_state['label'] == 'active':
                    print(f'  Nav2 active after {time.time() - start:.0f} s', flush=True)
                    said.add('Nav2')
        if latest.get('pose') and latest.get('ir_sensors', 0) >= args.min_ir and (
                nav is None or 'Nav2' in said):
            ok = True
            break
    if not ok:
        wanted = (('task node', 'pose') + (('IR',) if args.min_ir else ())
                  + (('Nav2',) if nav is not None else ()))
        missing = [label for label in wanted if label not in said]
        print(f'  still missing after {args.timeout:.0f} s: {", ".join(missing)}', flush=True)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
