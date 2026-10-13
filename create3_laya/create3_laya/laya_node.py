"""ROS 2 node: tell the Create 3 what to do in plain English; Laya interprets it and the robot
carries it out in create3_sim (Gazebo) or on a real Create 3.

Topics (relative, so they follow the robot namespace):
  sub  task                 std_msgs/String     e.g. "go to the east hall, then come back"
  sub  odom, hazard_detection, ir_intensity, _internal/ir_intensity_<sensor>, dock_status
  pub  cmd_vel              geometry_msgs/TwistStamped   (only while a task runs)
  pub  task_status          std_msgs/String     JSON: interpretations and step progress
Actions used: dock, undock (irobot_create_msgs).

Laya calls run on worker threads; the 20 Hz control loop never waits on the network.
"""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import rclpy
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import (DockStatus, HazardDetectionVector, IrIntensity,
                                    IrIntensityVector)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String

from . import create3, ros_convert
from .laya_client import LayaClient, LayaError
from .mission import MissionRunner, Step
from .pilot import LayaPilot, Snapshot
from .planner import GridPlanner
from .tasks import DOCK_DESCRIPTION, START_DESCRIPTION, TaskInterpreter, load_places
from .vision import FindBehavior, SemanticMemory
from .world import Box, World

STALE_IR = 0.5          # s; older IR readings count as missing
STALE_HAZARD = 0.3      # s
CODE_PERIOD = 0.2       # s between code-driver decisions


def _share(package, *parts):
    try:
        return os.path.join(get_package_share_directory(package), *parts)
    except PackageNotFoundError:
        return ''


class LayaCreate3(Node):
    def __init__(self):
        super().__init__('laya_create3')
        p = self.declare_parameter
        self.spawn = (p('spawn_x', 0.0).value, p('spawn_y', 0.0).value,
                      p('spawn_yaw', 0.0).value)
        url = p('laya_url', '').value or None
        task_model = p('task_model', 'typed-decisions').value
        driving_model = p('driving_model', 'create3-driving').value
        driver = p('driver', 'code').value
        world_sdf = p('world_sdf', '').value or _share('irobot_create_gz_bringup', 'worlds',
                                                       'maze.sdf')
        places_file = p('places_file', '').value or _share('create3_laya', 'config',
                                                           'maze_places.json')
        self.period = p('decision_period', 0.5).value
        rate = p('control_rate', 20.0).value
        min_confidence = p('min_confidence', 0.3).value
        # 'odom' (wheel odometry from the spawn pose; drifts, as on a real Create 3) or
        # 'ground_truth' (create3_sim's sim_ground_truth_pose; simulation only).
        self.pose_source = p('pose_source', 'odom').value
        if self.pose_source not in ('odom', 'ground_truth'):
            raise ValueError(f"pose_source must be 'odom' or 'ground_truth', "
                             f'not {self.pose_source!r}')

        dock_in_map = p('dock_in_map', True).value

        planner = None
        if world_sdf and world_sdf != 'none' and os.path.exists(world_sdf):
            world = World.from_sdf(world_sdf)
            if world.boxes:
                if dock_in_map:
                    # create3_spawn puts the dock 0.157 m ahead of the robot, facing it; its
                    # collision box (4 x 13 cm) sits a further 4.5 cm out.
                    sx, sy, syaw = self.spawn
                    d = 0.157 + 0.045
                    world.boxes.append(Box('standard_dock', sx + d * math.cos(syaw),
                                           sy + d * math.sin(syaw), syaw, 0.04, 0.13))
                planner = GridPlanner(world)
        if planner is None and world_sdf != 'none':
            self.get_logger().warn(f'no local geometry in {world_sdf!r}: driving straight at '
                                   'goals without a route')
        # Generous: the first request after Gazebo or the camera model crowded Laya out of GPU
        # memory took 17 s while its weights paged back in (later ones: 0.2 s). Tasks are
        # interpreted on their own thread, so waiting doesn't stall driving.
        self.task_client = LayaClient(url, task_model, timeout=30.0)
        self.drive_client = LayaClient(url, driving_model, timeout=1.0)
        self.pilot = LayaPilot(planner=planner, driver=driver, min_confidence=min_confidence)
        handlers = {'dock': lambda step, snap, first: self._run_action('dock', first),
                    'undock': lambda step, snap, first: self._run_action('undock', first)}

        # Optional: Nav2 for go steps, and a camera for 'find' (TurtleBot 4).
        self.nav = self.eyes = self.finder = None
        self.memory = SemanticMemory()
        if p('nav_backend', 'pilot').value == 'nav2':
            from .ros_extras import Nav2Navigator
            self.nav = Nav2Navigator(self, frame=p('nav_frame', 'odom').value)
            handlers['go_to'] = handlers['go_relative'] = \
                lambda step, snap, first: self._navigate(step.target, first, snap)
        # The lidar's (x, y, yaw) on the robot: the TurtleBot 4's RPLIDAR sits 4 cm back, turned
        # 90 deg left (turtlebot4_description), so a scan's 0 deg is the robot's left.
        self.lidar_mount = tuple(p('lidar_mount', [-0.04, 0.0, math.pi / 2]).value)
        scan_topic = p('scan_topic', 'scan').value
        if p('camera', False).value:
            from .ros_extras import CameraEyes
            self.eyes = CameraEyes(
                self, p('vision_url', '').value or None,
                image=p('image_topic', 'oakd/rgb/preview/image_raw').value,
                depth=p('depth_topic', 'oakd/rgb/preview/depth').value,
                info=p('camera_info_topic', 'oakd/rgb/preview/camera_info').value,
                scan=scan_topic, scan_yaw=self.lidar_mount[2])
            handlers['find'] = self._find
        self.runner = MissionRunner(self.pilot, on_interrupt=lambda step: self._cancel_actions(),
                                    action_handlers=handlers)

        # Named places: the catalog ('none' for worlds without one, like the depot), plus
        # 'start' and 'dock' from the spawn pose, plus whatever the camera has seen.
        self.base_places = {} if places_file == 'none' else load_places(places_file)
        sx, sy, syaw = self.spawn
        self.base_places.setdefault('start', {'xy': (sx, sy), 'description': START_DESCRIPTION})
        # The robot spawns docked facing the dock; undocking backs it away from it.
        self.base_places.setdefault('dock', {
            'xy': (sx - 0.6 * math.cos(syaw), sy - 0.6 * math.sin(syaw)),
            'face': (sx + 0.2 * math.cos(syaw), sy + 0.2 * math.sin(syaw)),
            'description': DOCK_DESCRIPTION})
        self.interpreter = TaskInterpreter(dict(self.base_places), self.task_client)
        self.task_pool = ThreadPoolExecutor(max_workers=1)
        self.drive_pool = ThreadPoolExecutor(max_workers=1)

        self.pose = None
        self._snap = None
        self.ir, self.ir_stamp = {}, {}
        self.hazards, self.hazard_stamp = frozenset(), -math.inf
        self.docked = False
        self.tasks = []              # (text, busy, future) in arrival order
        self.queue = []              # step lists waiting for the current mission
        self.drive_query = None      # (query, future, start)
        self.next_decision = 0.0
        self.event_count = 0
        self.was_driving = False
        self.last_status = 'idle'
        self.actions = {'dock': (ActionClient(self, Dock, 'dock'), Dock.Goal()),
                        'undock': (ActionClient(self, Undock, 'undock'), Undock.Goal())}
        self.action_state = {}

        sensor_qos = qos_profile_sensor_data
        if self.pose_source == 'ground_truth':
            # Simulation only (create3_sim's pose_republisher), world frame already.
            self.create_subscription(
                Odometry, 'sim_ground_truth_pose',
                lambda msg: setattr(self, 'pose', ros_convert.odom_to_map(msg.pose.pose,
                                                                         (0.0, 0.0, 0.0))),
                sensor_qos)
        else:
            self.create_subscription(Odometry, 'odom', self._on_odom, sensor_qos)
            # Gyro-aided dead reckoning (see ros_convert.GyroOdometry).
            self.imu_yaw, self.odometry = None, ros_convert.GyroOdometry()
            self.create_subscription(Imu, 'imu', self._on_imu, sensor_qos)
        self.create_subscription(HazardDetectionVector, 'hazard_detection', self._on_hazards,
                                 sensor_qos)
        self.create_subscription(IrIntensityVector, 'ir_intensity', self._on_ir_vector,
                                 sensor_qos)
        for sensor in create3.IR_SENSORS:
            # Per-sensor topics carry the sensor identity that create3_sim's vector loses.
            self.create_subscription(
                IrIntensity, f'_internal/ir_intensity_{sensor.name}',
                lambda msg, name=sensor.name: self._store_ir({name: int(msg.value)}), sensor_qos)
        if p('ir_from_lidar', False).value:
            # A simulation that leaves the IR sensors out (each costs a GPU render that the
            # simulation waits for) reads them off the lidar, so the IR speed limits and
            # obstacle stops still work; a missing reading would count as a wall at 8 cm.
            self.create_subscription(
                LaserScan, scan_topic,
                lambda msg: self._store_ir(create3.ir_from_points(
                    ros_convert.scan_points(msg, self.lidar_mount))), sensor_qos)
        self.create_subscription(DockStatus, 'dock_status', self._on_dock_status, sensor_qos)
        self.create_subscription(String, 'task', self._on_task, 10)
        self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', sensor_qos)
        self.status_pub = self.create_publisher(String, 'task_status', 10)
        self.create_timer(1.0 / rate, self._tick)
        self.create_timer(2.0, self._heartbeat)
        self.get_logger().info(
            f'ready: driver={driver}, pose={self.pose_source}, '
            f'Laya at {self.task_client.url} (tasks: {task_model}'
            f'{", driving: " + driving_model if driver == "laya" else ""}); '
            f'send tasks on {self.resolve_topic_name("task")}')

    # -- sensors ---------------------------------------------------------------------

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg):
        odom = ros_convert.odom_to_map(msg.pose.pose, self.spawn)
        self.pose = self.odometry.update(odom, self.imu_yaw)

    def _on_imu(self, msg):
        if msg.orientation_covariance[0] >= 0.0:   # -1 marks "no orientation estimate"
            self.imu_yaw = ros_convert.yaw_from_quaternion(msg.orientation)

    def _on_hazards(self, msg):
        self.hazards, self.hazard_stamp = ros_convert.hazards(msg), self._now()

    def _on_ir_vector(self, msg):
        self._store_ir(ros_convert.ir_vector_readings(msg))

    def _store_ir(self, readings):
        now = self._now()
        for name, value in readings.items():
            self.ir[name], self.ir_stamp[name] = value, now

    def _on_dock_status(self, msg):
        self.docked = bool(msg.is_docked)

    def _snapshot(self, now):
        if self.pose is None:
            return None
        ir = {n: v for n, v in self.ir.items() if now - self.ir_stamp[n] <= STALE_IR}
        hazards = self.hazards if now - self.hazard_stamp <= STALE_HAZARD else frozenset()
        return Snapshot(now, *self.pose, ir, hazards)

    # -- tasks -----------------------------------------------------------------------

    def _on_task(self, msg):
        text = msg.data.strip()
        if not text:
            return
        busy = self.runner.active or bool(self.queue)
        # Things the camera has seen become places ("go back to the pallets").
        self.interpreter.places = {**self.base_places, **self.memory.places()}
        self.tasks.append((text, busy, self.task_pool.submit(self.interpreter.interpret, text,
                                                             None, busy)))
        self.get_logger().info(f'task received: "{text}"')

    def _collect_tasks(self, snap):
        while self.tasks and self.tasks[0][2].done():
            text, busy, future = self.tasks.pop(0)
            try:
                plan = future.result()
            except LayaError as e:
                self._status('error', text=text, detail=f'Laya unavailable: {e}')
                continue
            self._apply_plan(text, plan, snap)

    def _apply_plan(self, text, plan, snap):
        busy = self.runner.active or bool(self.queue)
        if not plan.runnable:
            outcome = 'refused' if plan.refusal else 'needs clarification'
        elif plan.stops:
            self._cancel_actions()
            self.runner.cancel(snap, f'"{text}"')
            self.queue.clear()
            outcome = 'stopped'
        else:
            steps = list(plan.steps)
            if busy and plan.timing == 'after':
                self.queue.append(steps)
                outcome = 'queued'
            else:
                if self.docked and steps[0].kind not in ('dock', 'undock'):
                    steps.insert(0, Step('undock', clause='leave the dock first'))
                self._cancel_actions()
                self.queue.clear()
                self.runner.start(steps, snap)
                outcome = 'replaced current task' if busy else 'started'
        started = outcome in ('started', 'replaced current task')
        self._status('task', text=text, outcome=outcome, summary=plan.summary(),
                     mission=self.runner.mission if started else None,
                     steps=[s.describe() for s in plan.steps], timing=plan.timing,
                     clauses=[{'text': c.text, 'skill': c.skill,
                               'skill_confidence': round(c.skill_confidence, 3),
                               'place': c.place, 'place_confidence': round(c.place_confidence, 3)}
                              for c in plan.clauses])
        self._check_mission()      # after the plan, so listeners know the mission its steps join
        self.get_logger().info(f'"{text}" -> {outcome}: {plan.summary()}')

    # -- control loop ----------------------------------------------------------------

    def _tick(self):
        now = self._now()
        snap = self._snapshot(now)
        if snap is None:
            return
        self._snap = snap
        if self.eyes is not None:
            self._collect_sightings()
        self._collect_tasks(snap)
        self._decide(snap, now)
        command = self.runner.command(snap)
        self._check_mission()
        if not self.runner.active and self.queue:
            steps = self.queue.pop(0)
            if self.docked and steps[0].kind not in ('dock', 'undock'):
                steps.insert(0, Step('undock', clause='leave the dock first'))
            self.runner.start(steps, snap)
        self._report_events()
        # Publish only while this node drives; 'external' means an action server (dock,
        # undock) or Nav2 is driving, and our zeros would fight it. One zero after we stop.
        driving = self.runner.active and command.source != 'external'
        if driving or self.was_driving:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'base_link'
            msg.twist.linear.x = float(command.v) if driving else 0.0
            msg.twist.angular.z = float(command.w) if driving else 0.0
            self.cmd_pub.publish(msg)
        self.was_driving = driving

    def _decide(self, snap, now):
        if not self.runner.needs_decisions():
            self.drive_query = None
            return
        if not self.pilot.uses_model:
            if now >= self.next_decision or self.pilot.wants_decision:
                self.pilot.decide_locally(snap)
                self.next_decision = now + CODE_PERIOD
            return
        if self.drive_query is None and (now >= self.next_decision or self.pilot.wants_decision):
            query = self.runner.prepare(snap)
            future = self.drive_pool.submit(self.drive_client.decide, query.state,
                                            query.questions)
            self.drive_query = (query, future, now)
            self.next_decision = now + self.period
        if self.drive_query is not None and self.drive_query[1].done():
            query, future, start = self.drive_query
            self.drive_query = None
            response, error = None, None
            try:
                response = future.result()
            except LayaError as e:
                error = e
            try:
                self.runner.apply(query, now, response, error, now - start)
            except LayaError as e:
                self.get_logger().error(f'driving model failed, stopping: {e}')
                self.runner.cancel(snap, f'driving model failed: {e}')

    # -- Create 3 actions (dock / undock) --------------------------------------------

    def _run_action(self, name, first):
        client, goal = self.actions[name]
        if first:
            if not client.server_is_ready():
                self.get_logger().error(f'{name} action server not available')
                return 'failed'
            self.action_state[name] = {'goal': client.send_goal_async(goal), 'result': None,
                                       'handle': None}
            return 'running'
        state = self.action_state.get(name)
        if state is None:
            return 'failed'
        if 'retry_at' in state:
            if self._now() < state['retry_at']:
                return 'running'
            result = self._run_action(name, True)
            if name in self.action_state:
                self.action_state[name]['tries'] = state['tries']
            return result
        if state['result'] is None:
            if not state['goal'].done():
                return 'running'
            handle = state['goal'].result()
            if not handle.accepted:
                return self._action_failed(name)
            state['handle'], state['result'] = handle, handle.get_result_async()
            return 'running'
        if not state['result'].done():
            return 'running'
        docked = state['result'].result().result.is_docked
        if docked == (name == 'dock'):
            self.action_state.pop(name, None)
            return 'done'
        return self._action_failed(name)

    def _action_failed(self, name, retries=2):
        """A task that replaces one mid-undock finds the robot partly off the dock, or the
        Create 3 still cancelling that undock (it refused the new goal, and the task failed).
        Off the dock is all an undock needs; otherwise try again a second later."""
        state = self.action_state.pop(name, None) or {}
        if name != 'undock':
            return 'failed'
        if not self.docked:
            return 'done'
        tries = state.get('tries', 0)
        if tries >= retries:
            return 'failed'
        self.action_state[name] = {'retry_at': self._now() + 1.0, 'tries': tries + 1}
        return 'running'

    def _cancel_actions(self):
        for state in self.action_state.values():
            if state.get('handle') is not None:
                state['handle'].cancel_goal_async()
        self.action_state.clear()
        if self.nav is not None:
            self.nav.cancel()

    # -- Nav2 and the camera (TurtleBot 4) -------------------------------------------

    def _to_odom(self, x, y):
        """Our map frame -> the frame anchored where the robot started: odom, and also the
        map SLAM builds (it starts aligned with odom)."""
        sx, sy, syaw = self.spawn
        dx, dy = x - sx, y - sy
        c, s = math.cos(syaw), math.sin(syaw)
        return c * dx + s * dy, -s * dx + c * dy

    def _navigate(self, target, first, snap=None):
        if self.nav is None:
            return 'failed', 'no navigation backend for this step'
        snap = snap or self._snap
        to_frame = self._to_odom if self.nav.frame in ('odom', 'map') else (lambda x, y: (x, y))
        return self.nav.navigate(target, first, (snap.x, snap.y), to_frame)

    def _find(self, step, snap, first):
        if self.nav is None:
            return 'failed', 'finding objects needs Nav2 (launch with nav_backend:=nav2)'
        if first:
            # It looks before choosing; sightings from before this search count as well.
            self.finder = FindBehavior(step.place, lambda target, f: self._navigate(target, f),
                                       self.memory, prefer=step.bearing, near=step.near)
            # Query -> label: the target's, plus the reference's for "X near Y".
            self.find_labels = {q: step.place for q in (step.queries or [f'a {step.place}'])}
            self.find_labels.update({q: step.near for q in step.near_queries})
            self.find_started = snap.t
        if self.eyes.error and snap.t - self.find_started > 5.0:
            return 'failed', f'vision server unavailable: {self.eyes.error}'
        self.eyes.request(snap.t, (snap.x, snap.y, snap.yaw), self.find_labels)
        chosen = self.finder.chosen
        result = self.finder.step(snap, self.eyes.open_bearing())
        if self.finder.chosen is not chosen:
            s, f = self.finder.chosen, self.finder
            self.get_logger().info(
                f'chose the {s.label} at ({s.x:.1f}, {s.y:.1f}), '
                f'{math.degrees(f.off(s)):.0f} deg off, out of {len(f.candidates)} sightings '
                f'({len(f.remembered)} from before)')
        return result

    def _collect_sightings(self):
        for s in self.eyes.collect():
            before = self.memory.sightings.get(s.label)
            self.memory.add(s)
            if self.finder is not None:
                self.finder.offer(s)
            if self.memory.sightings[s.label] is s and before is not s:
                self._status('seen', label=s.label, x=round(s.x, 2), y=round(s.y, 2),
                             score=round(s.score, 3), distance=round(s.distance, 2))
                self.get_logger().info(f'saw {s.label} ({s.score:.2f}) {s.distance:.1f} m away '
                                       f'at ({s.x:.1f}, {s.y:.1f})')

    # -- status ----------------------------------------------------------------------

    def _report_events(self):
        while self.event_count < len(self.runner.events):
            e = self.runner.events[self.event_count]
            self.event_count += 1
            self._status('step', mission=e.mission, step=e.step, status=e.status,
                         detail=e.detail)
            if e.status != 'started':
                self.get_logger().info(f'step {e.step + 1} {e.status}'
                                       + (f': {e.detail}' if e.detail else ''))

    def _check_mission(self):
        self._report_events()      # step results (with the reason) before the mission's end
        status = self.runner.status
        if status != self.last_status and status in ('done', 'failed', 'stopped'):
            self._status('mission', mission=self.runner.mission, status=status)
        self.last_status = status

    def _heartbeat(self):
        """Lets tools see that the node is alive and has what it needs to drive."""
        self._status('heartbeat', pose=self.pose is not None, ir_sensors=len(self.ir),
                     busy=self.runner.active, pose_source=self.pose_source)

    def _status(self, kind, **fields):
        self.status_pub.publish(String(data=json.dumps({'event': kind, 't': self._now(),
                                                        **fields})))

    def stop(self):
        self._cancel_actions()
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)
        self.task_pool.shutdown(wait=False, cancel_futures=True)
        self.drive_pool.shutdown(wait=False, cancel_futures=True)
        if self.eyes is not None:
            self.eyes.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LayaCreate3()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
