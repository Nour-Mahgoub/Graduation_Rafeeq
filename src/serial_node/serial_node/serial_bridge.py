#!/usr/bin/env python3
"""
SmartWheel Serial Bridge Node
Bidirectional communication between ROS 2 and Arduino.

Arduino → RPi (20Hz CSV):
  L<ticks>,R<ticks>,AX<m/s²>,AY<m/s²>,AZ<m/s²>,GX<rad/s>,GY<rad/s>,GZ<rad/s>,US<cm>

RPi → Arduino (on cmd_vel):
  V<linear_x>,W<angular_z>\n

Publishes:
  - /odom          (nav_msgs/Odometry)
  - /imu/data      (sensor_msgs/Imu)
  - /ultrasonic    (sensor_msgs/Range)
  - /tf            (odom → base_footprint transform)

Subscribes:
  - /cmd_vel       (geometry_msgs/Twist)

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
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster


# ─── ROBOT PHYSICAL PARAMETERS ───────────────────────────────────────────────
WHEEL_RADIUS     = 0.085   # metres — measure your actual hoverboard wheel radius
WHEEL_SEPARATION = 0.34    # metres — measure axle centre to axle centre
ENCODER_CPR      = 90      # 6 × 15 pole pairs (standard hoverboard hub motor)
                            # Verify: count rotor magnets ÷ 2 = pole pairs


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

        # ── Subscriber ───────────────────────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

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

    # ─── CMD_VEL CALLBACK ────────────────────────────────────────────────────
    def cmd_vel_callback(self, msg: Twist):
        """
        Receive /cmd_vel from Nav2 and forward to Arduino as:
        V<linear_x>,W<angular_z>\n
        Arduino parses this and converts to hoverboard ESC commands.
        """
        linear  = msg.linear.x
        angular = msg.angular.z
        cmd = f'V{linear:.4f},W{angular:.4f}\n'
        try:
            self.ser.write(cmd.encode('utf-8'))
        except serial.SerialException as e:
            self.get_logger().error(f'Serial write failed: {e}')

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
        Arduino sends cumulative long ticks for L and R.
        IMU values must already be in SI units (m/s² and rad/s) from Arduino.
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
                'ax': float(m.group(3)),   # m/s² — converted on Arduino
                'ay': float(m.group(4)),
                'az': float(m.group(5)),
                'gx': float(m.group(6)),   # rad/s — converted on Arduino
                'gy': float(m.group(7)),
                'gz': float(m.group(8)),
                'us': float(m.group(9)),   # cm
            }
        except Exception:
            return None

    # ─── ODOMETRY ────────────────────────────────────────────────────────────
    def publish_odom(self, data, stamp):
        left_ticks  = data['left_ticks']
        right_ticks = data['right_ticks']

        # First reading — just initialise, no delta yet
        if self.prev_left_ticks is None:
            self.prev_left_ticks  = left_ticks
            self.prev_right_ticks = right_ticks
            self.last_stamp = stamp
            return

        # Tick deltas → wheel arc distances (metres)
        d_left  = ((left_ticks  - self.prev_left_ticks)  / ENCODER_CPR) * 2.0 * math.pi * WHEEL_RADIUS
        d_right = ((right_ticks - self.prev_right_ticks) / ENCODER_CPR) * 2.0 * math.pi * WHEEL_RADIUS
        self.prev_left_ticks  = left_ticks
        self.prev_right_ticks = right_ticks

        # Differential drive kinematics
        d_center = (d_left + d_right) / 2.0
        d_yaw    = (d_right - d_left) / WHEEL_SEPARATION

        self.yaw += d_yaw
        self.x   += d_center * math.cos(self.yaw)
        self.y   += d_center * math.sin(self.yaw)

        # Time delta for velocity
        dt = 0.05  # fallback 20 Hz
        if self.last_stamp is not None:
            dt_ns = (
                (stamp.sec - self.last_stamp.sec) * 1_000_000_000
                + (stamp.nanosec - self.last_stamp.nanosec)
            )
            if dt_ns > 0:
                dt = dt_ns / 1e9
        self.last_stamp = stamp

        q = self.yaw_to_quaternion(self.yaw)

        # ── Publish /odom ────────────────────────────────────────────────────
        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_footprint'

        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = q

        # Pose covariance (diagonal) — tune after real hardware testing
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw

        odom.twist.twist.linear.x  = d_center / dt
        odom.twist.twist.angular.z = d_yaw    / dt

        # Twist covariance
        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[35] = 0.05

        self.odom_pub.publish(odom)

        # ── Broadcast odom → base_footprint TF ──────────────────────────────
        tf = TransformStamped()
        tf.header.stamp            = stamp
        tf.header.frame_id         = 'odom'
        tf.child_frame_id          = 'base_footprint'
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

        # Values already in SI units — converted by Arduino before sending
        imu.linear_acceleration.x = data['ax']
        imu.linear_acceleration.y = data['ay']
        imu.linear_acceleration.z = data['az']

        imu.angular_velocity.x = data['gx']
        imu.angular_velocity.y = data['gy']
        imu.angular_velocity.z = data['gz']

        # Raw MPU6050 — no orientation estimate, tell robot_localization to ignore
        imu.orientation_covariance[0] = -1.0

        # Covariance diagonals — tune after calibration
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
        rng.field_of_view   = 0.26   # ~15 degrees in radians
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
        # Send stop command before shutting down
        try:
            self.ser.write(b'V0.0000,W0.0000\n')
        except Exception:
            pass
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