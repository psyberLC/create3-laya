"""Send one plain-English task to the robot and follow it until it ends.

  ros2 run create3_laya laya_task "go to the east hall, then come back to the dock"
  ros2 run create3_laya laya_task "explore two rooms" --ros-args -r __ns:=/robot1
  ros2 run create3_laya laya_task --console     # one task per stdin line, status as JSON lines

The console mode is how the Windows launcher talks to a simulator: each stdin line is sent as a
task, every task_status message is printed as "@@<json>", and end of input (or "quit") exits.
"""

import json
import queue
import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TaskSender(Node):
    def __init__(self, text):
        super().__init__('laya_task')
        self.text = text
        self.sent = False
        self.finished = False
        self.mission = None
        self.pub = self.create_publisher(String, 'task', 10)
        self.create_subscription(String, 'task_status', self._on_status, 10)
        self.create_timer(0.5, self._send)

    def _send(self):
        if not self.sent and self.pub.get_subscription_count() > 0:
            self.pub.publish(String(data=self.text))
            self.sent = True

    def _on_status(self, msg):
        status = json.loads(msg.data)
        event = status['event']
        if event == 'task' and status.get('text') == self.text and self.mission is None:
            print(f"{status['outcome']}: {status['summary']}", flush=True)
            self.mission = status.get('mission')
            self.finished = self.mission is None      # refused, queued, stopped, unclear
        elif event == 'step' and status['mission'] == self.mission:
            detail = f" - {status['detail']}" if status.get('detail') else ''
            print(f"  step {status['step'] + 1} {status['status']}{detail}", flush=True)
            self.finished = status['status'] == 'replaced'
        elif event == 'mission' and status['mission'] == self.mission:
            print(f"task {status['status']}", flush=True)
            self.finished = True
        elif event == 'error' and status.get('text') == self.text:
            print(status.get('detail'), flush=True)
            self.finished = True


class TaskConsole(Node):
    """Tasks from stdin, every status message to stdout as '@@<json>'."""

    def __init__(self):
        super().__init__('laya_task_console')
        self.finished = False
        self.pending = queue.Queue()
        self.pub = self.create_publisher(String, 'task', 10)
        self.create_subscription(String, 'task_status',
                                 lambda msg: print('@@' + msg.data, flush=True), 10)
        self.create_timer(0.2, self._send)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in sys.stdin:
            text = line.strip()
            if text.lower() in ('quit', 'exit'):
                break
            if text:
                self.pending.put(text)
        self.finished = True

    def _send(self):
        while self.pub.get_subscription_count() > 0 and not self.pending.empty():
            self.pub.publish(String(data=self.pending.get()))


def main(args=None):
    words = [w for w in rclpy.utilities.remove_ros_args(sys.argv)[1:] if w.strip()]
    if not words:
        print(__doc__)
        return 2
    rclpy.init(args=args)
    node = TaskConsole() if words == ['--console'] else TaskSender(' '.join(words))
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
