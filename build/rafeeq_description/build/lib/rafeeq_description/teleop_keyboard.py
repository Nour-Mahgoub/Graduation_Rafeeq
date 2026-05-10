#!/usr/bin/env python3
import curses
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear = 0.0
        self.angular = 0.0
        self.speed = 0.5
        self.turn = 1.0
        # Publish at 10 Hz regardless of key events
        self.create_timer(0.1, self._publish)

    def _publish(self):
        msg = Twist()
        msg.linear.x = self.linear
        msg.angular.z = self.angular
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TeleopKeyboard()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    def run(stdscr):
        stdscr.nodelay(True)
        stdscr.clear()
        for i, line in enumerate([
            'Rafeeq Keyboard Teleop',
            '----------------------',
            '  w / s  :  forward / backward',
            '  a / d  :  turn left / right',
            '  w+a / w+d : move and turn simultaneously',
            '  space  :  stop',
            '  + / -  :  increase / decrease speed',
            '  q      :  quit',
        ]):
            stdscr.addstr(i, 0, line)

        pressed = set()

        while True:
            key = stdscr.getch()

            if key == ord('q'):
                node.linear, node.angular = 0.0, 0.0
                break
            elif key == ord(' '):
                node.linear, node.angular = 0.0, 0.0
            elif key == ord('+') or key == ord('='):
                node.speed = min(node.speed + 0.1, 2.0)
                node.turn = min(node.turn + 0.1, 3.0)
            elif key == ord('-'):
                node.speed = max(node.speed - 0.1, 0.1)
                node.turn = max(node.turn - 0.1, 0.1)
            elif key == ord('w'):
                node.linear = node.speed
            elif key == ord('s'):
                node.linear = -node.speed
            elif key == ord('a'):
                node.angular = node.turn
            elif key == ord('d'):
                node.angular = -node.turn
            elif key == -1:
                # No key pressed — keep last velocity (robot keeps moving)
                pass

            try:
                stdscr.addstr(9, 0,
                    f'  speed={node.speed:.1f}  turn={node.turn:.1f}'
                    f'  |  linear={node.linear:.2f}  angular={node.angular:.2f}   ')
            except Exception:
                pass
            stdscr.refresh()
            time.sleep(0.02)

    curses.wrapper(run)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
