#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Keep only the front 180 degrees: -pi/2 (left) to pi/2 (right)
ANGLE_MIN_KEEP = -math.pi / 2
ANGLE_MAX_KEEP = math.pi / 2


class ScanRelay(Node):
    def __init__(self):
        super().__init__('scan_relay')
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        self.sub = self.create_subscription(LaserScan, '/scan_raw', self.callback, 10)

    def callback(self, msg):
        # Fix Gazebo's scoped frame_id
        msg.header.frame_id = 'Lidar_Link'

        # Find the index range that falls within [-pi/2, pi/2]
        n = len(msg.ranges)
        idx_start = round((ANGLE_MIN_KEEP - msg.angle_min) / msg.angle_increment)
        idx_end = round((ANGLE_MAX_KEEP - msg.angle_min) / msg.angle_increment)
        idx_start = max(0, min(idx_start, n - 1))
        idx_end = max(0, min(idx_end, n - 1))

        out = LaserScan()
        out.header = msg.header
        out.angle_min = msg.angle_min + idx_start * msg.angle_increment
        out.angle_max = msg.angle_min + idx_end * msg.angle_increment
        out.angle_increment = msg.angle_increment
        out.time_increment = msg.time_increment
        out.scan_time = msg.scan_time
        out.range_min = msg.range_min
        out.range_max = msg.range_max
        out.ranges = list(msg.ranges[idx_start:idx_end + 1])
        if msg.intensities:
            out.intensities = list(msg.intensities[idx_start:idx_end + 1])

        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanRelay()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
