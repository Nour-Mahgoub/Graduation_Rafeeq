#!/usr/bin/env python3
"""
SmartWheel Serial Bridge Node
Reads CSV serial data from Arduino and publishes:
  - /odom          (nav_msgs/Odometry)
  - /imu/data      (sensor_msgs/Imu)
  - /ultrasonic    (sensor_msgs/Range)
  - /tf            (odom -> base_link transform)

Usage:
  ros2 run serial_node serial_bridge --ros-args -p port:=/dev/ttyUSB0
"""

import rclpy
from rclpy.node import Node
import serial
import math
import re

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Range
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster


# ─── ROBOT PHYSICAL PARAMETERS ───────────────────────────────────────────────
WHEEL_RADIUS     = 0.085   # metres — actual motor wheel radius
WHEEL_SEPARATION = 0.34    # metres — distance between left and right motor wheels
ENCODER_CPR      = 360     # encoder counts per revolution


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__('serial_bridge')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('port',     '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout',  1.0)

        port     = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        timeout  = self.get_parameter('timeout').value

        # ── Serial connection ────────────────────────────────────────────────
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            self.get_logger().info(f'Serial connected on {port} at {baudrate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise

        # ── Publishers ───────────────────────────────────────────────────────
        self.odom_pub  = self.create_publisher(Odometry, '/odom',       10)
        self.imu_pub   = self.create_publisher(Imu,      '/imu/data',   10)
        self.range_pub = self.create_publisher(Range,    '/ultrasonic', 10)

        # ── TF Broadcaster ───────────────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Odometry state ───────────────────────────────────────────────────
        self.prev_left_ticks  = None
        self.prev_right_ticks = None
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0
        self.last_stamp = None

        # ── Main read timer (matches Arduino 20 Hz publish rate) ─────────────
        self.timer = self.create_timer(0.05, self.read_serial)

        self.get_logger().info('SmartWheel serial bridge node started')

    # ─── SERIAL READ CALLBACK ────────────────────────────────────────────────
    def read_serial(self):
        if not self.ser.in_waiting:
            return

        try:
            line = self.ser.readline().decode('utf-8').strip()
        except UnicodeDecodeError:
            return

        if not line:
            return

        data = self.parse_line(line)
        if data is None:
            self.get_logger().warn(f'Could not parse line: {line}')
            return

        now = self.get_clock().now().to_msg()

        self.publish_imu(data, now)
        self.publish_range(data, now)
        self.publish_odom(data, now)

    # ─── PARSER ──────────────────────────────────────────────────────────────
    def parse_line(self, line):
        """
        Parse: L<>,R<>,AX<>,AY<>,AZ<>,GX<>,GY<>,GZ<>,US<>
        Returns dict or None on failure.
        """
        try:
            pattern = (
                r'L(-?\d+),R(-?\d+),'
                r'AX(-?[\d.]+),AY(-?[\d.]+),AZ(-?[\d.]+),'
                r'GX(-?[\d.]+),GY(-?[\d.]+),GZ(-?[\d.]+),'
                r'US(-?[\d.]+)'
            )
            m = re.match(pattern, line)
            if not m:
                return None
            return {
                'left_ticks':  int(m.group(1)),
                'right_ticks': int(m.group(2)),
                'ax': float(m.group(3)),
                'ay': float(m.group(4)),
                'az': float(m.group(5)),
                'gx': float(m.group(6)),
                'gy': float(m.group(7)),
                'gz': float(m.group(8)),
                'us': float(m.group(9)),
            }
        except Exception:
            return None

    # ─── ODOMETRY ────────────────────────────────────────────────────────────
    def publish_odom(self, data, stamp):
        left_ticks  = data['left_ticks']
        right_ticks = data['right_ticks']

        if self.prev_left_ticks is None:
            self.prev_left_ticks  = left_ticks
            self.prev_right_ticks = right_ticks
            self.last_stamp = stamp
            return

        # Tick deltas → wheel arc distances
        d_left  = (left_ticks  - self.prev_left_ticks)  / ENCODER_CPR * 2 * math.pi * WHEEL_RADIUS
        d_right = (right_ticks - self.prev_right_ticks) / ENCODER_CPR * 2 * math.pi * WHEEL_RADIUS
        self.prev_left_ticks  = left_ticks
        self.prev_right_ticks = right_ticks

        # Differential drive kinematics
        d_center = (d_left + d_right) / 2.0
        d_yaw    = (d_right - d_left) / WHEEL_SEPARATION

        self.yaw += d_yaw
        self.x   += d_center * math.cos(self.yaw)
        self.y   += d_center * math.sin(self.yaw)

        # Time delta for velocity estimate
        dt = 0.05  # nominal 20 Hz; good enough for velocity
        if self.last_stamp is not None:
            dt_ns = (stamp.sec - self.last_stamp.sec) * 1e9 + (stamp.nanosec - self.last_stamp.nanosec)
            if dt_ns > 0:
                dt = dt_ns / 1e9
        self.last_stamp = stamp

        q = self.yaw_to_quaternion(self.yaw)

        # ── Publish /odom ────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'

        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = q

        odom.twist.twist.linear.x  = d_center / dt
        odom.twist.twist.angular.z = d_yaw    / dt

        self.odom_pub.publish(odom)

        # ── Broadcast odom → base_link TF ────────────────────────────────────
        tf = TransformStamped()
        tf.header.stamp            = stamp
        tf.header.frame_id         = 'odom'
        tf.child_frame_id          = 'base_link'
        tf.transform.translation.x = self.x
        tf.transform.translation.y = self.y
        tf.transform.translation.z = 0.0
        tf.transform.rotation      = q

        self.tf_broadcaster.sendTransform(tf)

    # ─── IMU ─────────────────────────────────────────────────────────────────
    def publish_imu(self, data, stamp):
        imu = Imu()
        imu.header.stamp    = stamp
        imu.header.frame_id = 'imu_link'

        imu.linear_acceleration.x = data['ax']
        imu.linear_acceleration.y = data['ay']
        imu.linear_acceleration.z = data['az']

        imu.angular_velocity.x = data['gx']
        imu.angular_velocity.y = data['gy']
        imu.angular_velocity.z = data['gz']

        # Raw MPU6050 — orientation unknown, tell robot_localization to ignore it
        imu.orientation_covariance[0] = -1.0

        imu.linear_acceleration_covariance[0] = 0.01
        imu.linear_acceleration_covariance[4] = 0.01
        imu.linear_acceleration_covariance[8] = 0.01

        imu.angular_velocity_covariance[0] = 0.005
        imu.angular_velocity_covariance[4] = 0.005
        imu.angular_velocity_covariance[8] = 0.005

        self.imu_pub.publish(imu)

    # ─── ULTRASONIC ──────────────────────────────────────────────────────────
    def publish_range(self, data, stamp):
        if data['us'] < 0:
            return

        rng = Range()
        rng.header.stamp    = stamp
        rng.header.frame_id = 'ultrasonic_link'
        rng.radiation_type  = Range.ULTRASOUND
        rng.field_of_view   = 0.26   # ~15 degrees
        rng.min_range       = 0.05   # 5 cm
        rng.max_range       = 2.0    # 200 cm
        rng.range           = data['us'] / 100.0  # cm → metres

        self.range_pub.publish(rng)

    # ─── HELPERS ─────────────────────────────────────────────────────────────
    def yaw_to_quaternion(self, yaw) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def destroy_node(self):
        if self.ser.is_open:
            self.ser.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SerialBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
