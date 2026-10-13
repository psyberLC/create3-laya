"""Gazebo's clock to ROS /clock, at most `rate` times per simulated second.

Gazebo publishes its clock every physics step (333 Hz with the 3 ms step), and the bridge
passed each one on to every node using sim time: 58 subscriptions in the TurtleBot 4 depot,
18,000 deliveries a second that kept each idle node at ~12% CPU. Control loops here run at
20-60 Hz, so 100 Hz of sim time is enough. Reads Gazebo's /clock directly (gz-transport),
which also saves the bridge hop.

  ros2 run create3_laya clock_relay --ros-args -p rate:=100.0
"""

import threading

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


class ClockRelay(Node):
    def __init__(self):
        super().__init__('clock_relay')
        self.period = 1.0 / self.declare_parameter('rate', 100.0).value
        self.topic = self.declare_parameter('gz_topic', '/clock').value
        self.pub = self.create_publisher(
            Clock, '/clock', QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.last = None
        self.lock = threading.Lock()
        from gz.msgs10.clock_pb2 import Clock as GzClock
        from gz.transport13 import Node as GzNode
        self.gz = GzNode()              # keep: the subscription lives as long as this does
        if not self.gz.subscribe(GzClock, self.topic, self._on_gz_clock):
            raise RuntimeError(f'could not subscribe to Gazebo topic {self.topic}')
        self.get_logger().info(f'relaying {self.topic} at up to {1.0 / self.period:g} Hz')

    def _on_gz_clock(self, msg):       # gz-transport's thread
        t = msg.sim.sec + msg.sim.nsec * 1e-9
        with self.lock:
            # A time behind the last one is a world reset: pass it on at once.
            if self.last is not None and 0.0 <= t - self.last < self.period:
                return
            self.last = t
        self.pub.publish(Clock(clock=Time(sec=msg.sim.sec, nanosec=msg.sim.nsec)))


def main():
    rclpy.init()
    node = ClockRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
