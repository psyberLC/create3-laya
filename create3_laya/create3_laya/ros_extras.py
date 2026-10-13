"""ROS pieces for robots with more than the Create 3's own sensors (TurtleBot 4):
Nav2 navigation and the OAK-D camera + lidar. Imported only when the node is configured
for them, so the plain Create 3 setup needs neither Nav2 nor OpenCV.
"""

import math
from concurrent.futures import ThreadPoolExecutor

from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from .vision import Camera, VisionClient, VisionError, locate

SUCCEEDED = 4                      # action_msgs/GoalStatus.STATUS_SUCCEEDED


class Nav2Navigator:
    """NavigateToPose goals in `frame` (default 'odom'; Nav2 transforms them into its map)."""

    def __init__(self, node, frame='odom', action='navigate_to_pose'):
        from nav2_msgs.action import NavigateToPose
        self.node = node
        self.frame = frame
        self.Goal = NavigateToPose.Goal
        self.client = ActionClient(node, NavigateToPose, action)
        self.goal_future = self.result_future = self.handle = None

    def navigate(self, target, first, pose=None, to_frame=lambda x, y: (x, y)):
        """target: map (x, y). Returns 'running', 'done' or ('failed', detail)."""
        if first:
            self.cancel()
            if not self.client.server_is_ready():
                return 'failed', 'Nav2 is not running (navigate_to_pose unavailable)'
            goal = self.Goal()
            goal.pose.header.frame_id = self.frame
            goal.pose.header.stamp = self.node.get_clock().now().to_msg()
            x, y = to_frame(*target)
            goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
            yaw = 0.0
            if pose is not None:                # facing the way it travels, in the goal's frame
                px, py = to_frame(*pose)
                yaw = math.atan2(y - py, x - px)
            goal.pose.pose.orientation.z = math.sin(yaw / 2)
            goal.pose.pose.orientation.w = math.cos(yaw / 2)
            self.goal_future = self.client.send_goal_async(goal)
            self.result_future = self.handle = None
            return 'running'
        if self.goal_future is None:
            return 'failed', 'no navigation goal was sent'
        if self.result_future is None:
            if not self.goal_future.done():
                return 'running'
            self.handle = self.goal_future.result()
            if not self.handle.accepted:
                return 'failed', 'Nav2 rejected the goal'
            self.result_future = self.handle.get_result_async()
            return 'running'
        if not self.result_future.done():
            return 'running'
        status = self.result_future.result().status
        self.goal_future = self.result_future = self.handle = None
        return 'done' if status == SUCCEEDED else ('failed', f'Nav2 ended with status {status}')

    def cancel(self):
        if self.handle is not None:
            self.handle.cancel_goal_async()
        self.goal_future = self.result_future = self.handle = None


class CameraEyes:
    """Latest OAK-D frames + lidar scan, and detection requests to the local vision server."""

    def __init__(self, node, vision_url=None, image='oakd/rgb/preview/image_raw',
                 depth='oakd/rgb/preview/depth', info='oakd/rgb/preview/camera_info',
                 scan='scan', scan_yaw=0.0, period=0.4, min_score=0.18):
        import numpy  # noqa: F401  (fail early, with a clear error, if missing)
        self.node = node
        self.client = VisionClient(vision_url)
        self.period = period
        self.min_score = min_score
        self.scan_yaw = scan_yaw          # the lidar's mounting angle on the robot
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.image = self.depth = self.camera = self.scan = None
        self.pending = None               # (future, pose, depth, camera, label, t)
        self.last_request = -math.inf
        self.error = None
        node.create_subscription(Image, image, self._on_image, qos_profile_sensor_data)
        node.create_subscription(Image, depth, self._on_depth, qos_profile_sensor_data)
        node.create_subscription(CameraInfo, info, self._on_info, qos_profile_sensor_data)
        node.create_subscription(LaserScan, scan, self._on_scan, qos_profile_sensor_data)

    # -- sensor callbacks ------------------------------------------------------------

    def _on_image(self, msg):
        self.image = msg

    def _on_depth(self, msg):
        self.depth = msg

    def _on_info(self, msg):
        if msg.k[0] > 0:
            self.camera = Camera(fx=msg.k[0], cx=msg.k[2], width=msg.width)

    def _on_scan(self, msg):
        self.scan = msg

    @property
    def ready(self):
        return None not in (self.image, self.depth, self.camera)

    # -- detection -------------------------------------------------------------------

    def request(self, now, pose, labels):
        """Send the latest frame for detection if the previous answer is in and it's time.
        `labels`: detector query -> label its sightings get (the target and, for "X near Y",
        the reference share one frame)."""
        if self.pending is not None or not self.ready or now - self.last_request < self.period:
            return
        jpeg = _jpeg(self.image)
        depth = _depth_array(self.depth)
        future = self.pool.submit(self.client.detect, jpeg, list(labels), self.min_score)
        self.pending = (future, pose, depth, self.camera, dict(labels), now)
        self.last_request = now

    def collect(self):
        """Sightings from a finished request (possibly none)."""
        if self.pending is None or not self.pending[0].done():
            return []
        future, pose, depth, camera, labels, t = self.pending
        self.pending = None
        try:
            result = future.result()
        except VisionError as e:
            self.error = str(e)
            return []
        self.error = None
        sightings = []
        sx = depth.shape[1] / max(1, result['width'])
        sy = depth.shape[0] / max(1, result['height'])
        for d in result['detections']:
            x0, y0, x1, y1 = d['box']
            box = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
            label = labels.get(d.get('query'), next(iter(labels.values())))
            s = locate(pose, Camera(camera.fx * sx, camera.cx * sx, depth.shape[1]), box, depth,
                       label, d['score'], t)
            if s is not None:
                sightings.append(s)
        return sightings

    def open_bearing(self, window=math.radians(30)):
        """Bearing (rad, robot frame) of the most open direction in the lidar scan."""
        scan = self.scan
        if scan is None or not scan.ranges:
            return 0.0
        n = len(scan.ranges)
        clean = [r if math.isfinite(r) and r > scan.range_min else scan.range_max
                 for r in scan.ranges]
        half = max(1, int(window / max(scan.angle_increment, 1e-6) / 2))
        best, best_score = 0.0, -1.0
        for i in range(0, n, max(1, half // 2)):
            span = [clean[(i + k) % n] for k in range(-half, half + 1)]
            score = min(span)
            if score > best_score:
                best_score, best = score, scan.angle_min + i * scan.angle_increment
        best += self.scan_yaw              # scan frame -> robot frame
        return math.atan2(math.sin(best), math.cos(best))

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def _array(msg, dtype, channels):
    """sensor_msgs/Image -> numpy array (height, width[, channels]), dropping row padding."""
    import numpy as np
    data = np.frombuffer(bytes(msg.data), dtype=dtype)
    rows = data.reshape(msg.height, msg.step // np.dtype(dtype).itemsize)
    pixels = rows[:, :msg.width * channels]
    return pixels.reshape(msg.height, msg.width, channels) if channels > 1 else pixels


def _jpeg(msg):
    import cv2
    import numpy as np
    channels = 4 if msg.encoding in ('rgba8', 'bgra8') else 3
    img = _array(msg, np.uint8, channels)
    if msg.encoding.startswith('rgb'):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR if channels == 4 else cv2.COLOR_RGB2BGR)
    elif channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise VisionError('could not encode the camera frame')
    return buf.tobytes()


def _depth_array(msg):
    import numpy as np
    if msg.encoding == '16UC1':
        return _array(msg, np.uint16, 1).astype(np.float32) / 1000.0
    return _array(msg, np.float32, 1).copy()
