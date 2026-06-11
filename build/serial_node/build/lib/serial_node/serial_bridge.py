# # #!/usr/bin/env python3
# # """
# # SmartWheel Serial Bridge Node
# # Bidirectional communication between ROS 2 and Arduino (RAFEEQ / Team-06).

# # Arduino → RPi (20 Hz CSV):
# #   L<rpm>,R<rpm>,AX<m/s²>,AY<m/s²>,AZ<m/s²>,GX<rad/s>,GY<rad/s>,GZ<rad/s>,BV<V>,BT<°C>

# #   L / R   – left / right wheel RPM from hoverboard Hall sensors
# #               (sign convention: positive = forward, right motor already
# #                sign-corrected on the Arduino side)
# #   AX…GZ  – MPU-6050 values already converted to SI units on Arduino
# #   BV      – battery voltage decoded from ESC (volts)
# #   BT      – ESC board temperature (°C)

# # RPi → Arduino (on cmd_vel):
# #   V<linear_x>,W<angular_z>\n

# # Publishes:
# #   /odom          (nav_msgs/Odometry)
# #   /imu/data      (sensor_msgs/Imu)
# #   /battery       (sensor_msgs/BatteryState)
# #   /tf            (odom → base_footprint transform)

# # Subscribes:
# #   /cmd_vel       (geometry_msgs/Twist)

# # Usage:
# #   ros2 run serial_node serial_bridge --ros-args -p port:=/dev/ttyUSB0
# # """

# # import rclpy
# # from rclpy.node import Node
# # import serial
# # import math
# # import re

# # from nav_msgs.msg import Odometry
# # from sensor_msgs.msg import Imu, BatteryState
# # from geometry_msgs.msg import Twist, TransformStamped, Quaternion
# # from tf2_ros import TransformBroadcaster
# # from std_msgs.msg import String


# # # ─── ROBOT PHYSICAL PARAMETERS (locked — from Arduino firmware) ──────────────
# # WHEEL_RADIUS     = 0.130   # metres  (r = 13 cm loaded, matches Arduino)
# # WHEEL_SEPARATION = 0.348   # metres  (d = 34.8 cm axle-to-axle, matches Arduino)

# # # RPM → rad/s conversion factor  (ω = RPM × 2π / 60)
# # RPM_TO_RAD_S = 2.0 * math.pi / 60.0


# # class SerialBridgeNode(Node):

# #     def __init__(self):
# #         super().__init__('serial_bridge')

# #         # ── Parameters ──────────────────────────────────────────────────────
# #         self.declare_parameter('port',     '/dev/ttyACM0')
# #         self.declare_parameter('baudrate', 115200)
# #         self.declare_parameter('timeout',  1.0)

# #         port     = self.get_parameter('port').value
# #         baudrate = self.get_parameter('baudrate').value
# #         timeout  = self.get_parameter('timeout').value

# #         # ── Serial connection ────────────────────────────────────────────────
# #         try:
# #             self.ser = serial.Serial(port, baudrate, timeout=timeout)
# #             self.get_logger().info(f'Serial connected on {port} at {baudrate} baud')
# #         except serial.SerialException as e:
# #             self.get_logger().error(f'Failed to open serial port: {e}')
# #             raise

# #         # ── Publishers ───────────────────────────────────────────────────────
# #         self.odom_pub    = self.create_publisher(Odometry,      '/odom',     10)
# #         self.imu_pub     = self.create_publisher(Imu,           '/imu/data', 10)
# #         self.battery_pub = self.create_publisher(BatteryState,  '/battery',  10)
# #         self.arduino_cmd_pub = self.create_publisher(String, '/arduino_cmd', 10)

# #         # ── Subscriber ───────────────────────────────────────────────────────
# #         self.cmd_vel_sub = self.create_subscription(
# #             Twist, '/cmd_vel', self.cmd_vel_callback, 10)

# #         # ── TF Broadcaster ───────────────────────────────────────────────────
# #         self.tf_broadcaster = TransformBroadcaster(self)

# #         # ── Odometry state ───────────────────────────────────────────────────
# #         self.x          = 0.0
# #         self.y          = 0.0
# #         self.yaw        = 0.0
# #         self.last_stamp = None   # ROS time of previous packet

# #         # ── Main read timer (matches Arduino 20 Hz publish rate) ─────────────
# #         self.timer = self.create_timer(0.05, self.read_serial)

# #         self.get_logger().info('SmartWheel serial bridge node started (RPM mode)')

# #     # ─── CMD_VEL CALLBACK ────────────────────────────────────────────────────
# #     # def cmd_vel_callback(self, msg: Twist):
# #     #     """
# #     #     Forward /cmd_vel to Arduino as:  V<linear_x>,W<angular_z>\n
# #     #     """
# #     #     cmd = f'V{msg.linear.x:.4f},W{msg.angular.z:.4f}\n'
# #     #     try:
# #     #         self.ser.write(cmd.encode('utf-8'))
# #     #     except serial.SerialException as e:
# #     #         self.get_logger().error(f'Serial write failed: {e}')
# #     def cmd_vel_callback(self, msg: Twist):
# #         cmd = f'V{msg.linear.x:.4f},W{msg.angular.z:.4f}\n'
# #         try:
# #             self.ser.write(cmd.encode('utf-8'))
# #             arduino_cmd_msg = String()
# #             arduino_cmd_msg.data = cmd.strip()
# #             self.arduino_cmd_pub.publish(arduino_cmd_msg)
# #         except serial.SerialException as e:
# #             self.get_logger().error(f'Serial write failed: {e}')

# #     # ─── SERIAL READ CALLBACK ────────────────────────────────────────────────
# #     def read_serial(self):
# #         if not self.ser.in_waiting:
# #             return

# #         try:
# #             line = self.ser.readline().decode('utf-8').strip()
# #         except UnicodeDecodeError:
# #             return

# #         if not line:
# #             return

# #         # Discard the Arduino startup banners
# #         if line.startswith('RAFEEQ') or line.startswith('Waiting'):
# #             return

# #         data = self.parse_line(line)
# #         if data is None:
# #             self.get_logger().warn(f'Could not parse line: {line}')
# #             return

# #         now = self.get_clock().now().to_msg()

# #         self.publish_imu(data, now)
# #         self.publish_battery(data, now)
# #         self.publish_odom(data, now)

# #     # ─── PARSER ──────────────────────────────────────────────────────────────
# #     def parse_line(self, line: str):
# #         """
# #         Parse Arduino telemetry line:
# #           L<rpm>,R<rpm>,AX<f>,AY<f>,AZ<f>,GX<f>,GY<f>,GZ<f>,BV<f>,BT<f>

# #         Returns a dict or None on failure.

# #         RPM sign convention (already applied on Arduino):
# #           positive RPM = forward motion for both wheels.
# #         IMU values are already in SI units (m/s² and rad/s).
# #         """
# #         try:
# #             pattern = (
# #                 r'L(-?[\d.]+),R(-?[\d.]+),'
# #                 r'AX(-?[\d.]+),AY(-?[\d.]+),AZ(-?[\d.]+),'
# #                 r'GX(-?[\d.]+),GY(-?[\d.]+),GZ(-?[\d.]+),'
# #                 r'BV(-?[\d.]+),BT(-?[\d.]+)'
# #             )
# #             m = re.match(pattern, line)
# #             if not m:
# #                 return None
# #             return {
# #                 'rpm_left':  float(m.group(1)),   # positive = forward
# #                 'rpm_right': float(m.group(2)),   # positive = forward
# #                 'ax': float(m.group(3)),           # m/s²
# #                 'ay': float(m.group(4)),
# #                 'az': float(m.group(5)),
# #                 'gx': float(m.group(6)),           # rad/s
# #                 'gy': float(m.group(7)),
# #                 'gz': float(m.group(8)),
# #                 'bat_volt': float(m.group(9)),     # volts
# #                 'bat_temp': float(m.group(10)),    # °C
# #             }
# #         except Exception:
# #             return None

# #     # ─── ODOMETRY ────────────────────────────────────────────────────────────
# #     def publish_odom(self, data: dict, stamp):
# #         """
# #         Differential-drive odometry from wheel RPMs.

# #         Kinematics:
# #           ω_wheel  = RPM × 2π / 60            [rad/s]
# #           v_wheel  = ω_wheel × WHEEL_RADIUS   [m/s]
# #           v_linear = (v_L + v_R) / 2
# #           v_angular= (v_R - v_L) / WHEEL_SEPARATION

# #         Pose integration (Euler, dt from wall clock):
# #           yaw += v_angular × dt
# #           x   += v_linear  × cos(yaw) × dt
# #           y   += v_linear  × sin(yaw) × dt
# #         """
# #         now = stamp

# #         # Compute dt
# #         if self.last_stamp is None:
# #             self.last_stamp = now
# #             return   # need at least two stamps to integrate

# #         dt_ns = (
# #             (now.sec - self.last_stamp.sec) * 1_000_000_000
# #             + (now.nanosec - self.last_stamp.nanosec)
# #         )
# #         dt = dt_ns / 1e9 if dt_ns > 0 else 0.05   # fallback 20 Hz
# #         self.last_stamp = now

# #         # RPM → linear wheel velocity [m/s]
# #         v_left  = data['rpm_left']  * RPM_TO_RAD_S * WHEEL_RADIUS
# #         v_right = data['rpm_right'] * RPM_TO_RAD_S * WHEEL_RADIUS

# #         # Robot velocity in body frame
# #         v_linear  = (v_left + v_right) / 2.0
# #         v_angular = (v_right - v_left) / WHEEL_SEPARATION

# #         # Integrate pose
# #         self.yaw += v_angular * dt
# #         self.x   += v_linear * math.cos(self.yaw) * dt
# #         self.y   += v_linear * math.sin(self.yaw) * dt

# #         q = self.yaw_to_quaternion(self.yaw)

# #         # ── Publish /odom ────────────────────────────────────────────────────
# #         odom = Odometry()
# #         odom.header.stamp    = stamp
# #         odom.header.frame_id = 'odom'
# #         odom.child_frame_id  = 'base_footprint'

# #         odom.pose.pose.position.x  = self.x
# #         odom.pose.pose.position.y  = self.y
# #         odom.pose.pose.position.z  = 0.0
# #         odom.pose.pose.orientation = q

# #         # Pose covariance diagonal — tune after real hardware testing
# #         odom.pose.covariance[0]  = 0.01   # x
# #         odom.pose.covariance[7]  = 0.01   # y
# #         odom.pose.covariance[35] = 0.05   # yaw

# #         odom.twist.twist.linear.x  = v_linear
# #         odom.twist.twist.angular.z = v_angular

# #         # Twist covariance diagonal
# #         odom.twist.covariance[0]  = 0.01
# #         odom.twist.covariance[35] = 0.05

# #         self.odom_pub.publish(odom)

# #         # ── Broadcast odom → base_footprint TF ──────────────────────────────
# #         tf = TransformStamped()
# #         tf.header.stamp            = stamp
# #         tf.header.frame_id         = 'odom'
# #         tf.child_frame_id          = 'base_footprint'
# #         tf.transform.translation.x = self.x
# #         tf.transform.translation.y = self.y
# #         tf.transform.translation.z = 0.0
# #         tf.transform.rotation      = q

# #         self.tf_broadcaster.sendTransform(tf)

# #     # ─── IMU ─────────────────────────────────────────────────────────────────
# #     def publish_imu(self, data: dict, stamp):
# #         imu = Imu()
# #         imu.header.stamp    = stamp
# #         imu.header.frame_id = 'imu_link'

# #         # Already in SI units — converted by Arduino before sending
# #         imu.linear_acceleration.x = data['ax']
# #         imu.linear_acceleration.y = data['ay']
# #         imu.linear_acceleration.z = data['az']

# #         imu.angular_velocity.x = data['gx']
# #         imu.angular_velocity.y = data['gy']
# #         imu.angular_velocity.z = data['gz']

# #         # Raw MPU-6050: no orientation estimate
# #         # Setting [0] = -1 tells robot_localization to ignore orientation
# #         imu.orientation_covariance[0] = -1.0

# #         # Covariance diagonals — tune after calibration
# #         imu.linear_acceleration_covariance[0] = 0.01
# #         imu.linear_acceleration_covariance[4] = 0.01
# #         imu.linear_acceleration_covariance[8] = 0.01

# #         imu.angular_velocity_covariance[0] = 0.005
# #         imu.angular_velocity_covariance[4] = 0.005
# #         imu.angular_velocity_covariance[8] = 0.005

# #         self.imu_pub.publish(imu)

# #     # ─── BATTERY ─────────────────────────────────────────────────────────────
# #     def publish_battery(self, data: dict, stamp):
# #         """
# #         Publish ESC battery voltage and board temperature.
# #         BatteryState.voltage      → volts decoded from ESC (raw × 0.02663)
# #         BatteryState.temperature  → board temp °C (raw × 100 / 1024)

# #         POWER_SUPPLY_STATUS_UNKNOWN is used because the ESC does not
# #         report charge/discharge state.
# #         """
# #         bat = BatteryState()
# #         bat.header.stamp    = stamp
# #         bat.header.frame_id = 'base_link'

# #         bat.voltage     = data['bat_volt']    # V
# #         bat.temperature = data['bat_temp']    # °C
# #         bat.current     = float('nan')        # not available from ESC
# #         bat.charge      = float('nan')
# #         bat.capacity    = float('nan')
# #         bat.design_capacity = float('nan')
# #         bat.percentage  = float('nan')

# #         bat.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
# #         bat.power_supply_health     = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
# #         bat.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
# #         bat.present = True

# #         self.battery_pub.publish(bat)

# #     # ─── HELPERS ─────────────────────────────────────────────────────────────
# #     @staticmethod
# #     def yaw_to_quaternion(yaw: float) -> Quaternion:
# #         q = Quaternion()
# #         q.x = 0.0
# #         q.y = 0.0
# #         q.z = math.sin(yaw / 2.0)
# #         q.w = math.cos(yaw / 2.0)
# #         return q

# #     def destroy_node(self):
# #         # Send stop command before shutting down
# #         try:
# #             self.ser.write(b'V0.0000,W0.0000\n')
# #         except Exception:
# #             pass
# #         if self.ser.is_open:
# #             self.ser.close()
# #         super().destroy_node()


# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = SerialBridgeNode()
# #     try:
# #         rclpy.spin(node)
# #     except KeyboardInterrupt:
# #         pass
# #     finally:
# #         node.destroy_node()
# #         rclpy.shutdown()


# # if __name__ == '__main__':
# #     main()


# #!/usr/bin/env python3
# """
# SmartWheel Serial Bridge Node
# Bidirectional communication between ROS 2 and Arduino (RAFEEQ / Team-06).

# Arduino → RPi (20 Hz CSV):
#   L<rpm>,R<rpm>,AX<m/s²>,AY<m/s²>,AZ<m/s²>,GX<rad/s>,GY<rad/s>,GZ<rad/s>,BV<V>,BT<°C>

# RPi → Arduino (on cmd_vel):
#   V<linear_x>,W<angular_z>\n

# Publishes:
#   /odom          (nav_msgs/Odometry)
#   /imu/data      (sensor_msgs/Imu)
#   /battery       (sensor_msgs/BatteryState)
#   /arduino_cmd   (std_msgs/String)  — echoes every cmd sent to Arduino
#   /tf            (odom → base_footprint transform)

# Subscribes:
#   /cmd_vel       (geometry_msgs/Twist)

# Key design: serial reading runs in a dedicated daemon thread so it never
# blocks the ROS executor, and cmd_vel writes happen immediately on the
# callback thread — both directions are fully parallel.

# Usage:
#   ros2 run serial_node serial_bridge --ros-args -p port:=/dev/ttyACM0
# """

# import threading
# import re
# import math

# import rclpy
# from rclpy.node import Node
# import serial

# from nav_msgs.msg import Odometry
# from sensor_msgs.msg import Imu, BatteryState
# from geometry_msgs.msg import Twist, TransformStamped, Quaternion
# from std_msgs.msg import String
# from tf2_ros import TransformBroadcaster


# # ─── ROBOT PHYSICAL PARAMETERS (locked — matches Arduino firmware) ────────────
# WHEEL_RADIUS     = 0.130   # metres
# WHEEL_SEPARATION = 0.348   # metres
# RPM_TO_RAD_S     = 2.0 * math.pi / 60.0

# # Arduino startup lines to silently discard (tuple for startswith)
# _IGNORE_PREFIXES = ('RAFEEQ', 'Waiting', 'ESC', 'V')


# class SerialBridgeNode(Node):

#     def __init__(self):
#         super().__init__('serial_bridge')

#         # ── Parameters ──────────────────────────────────────────────────────
#         self.declare_parameter('port',     '/dev/ttyACM0')
#         self.declare_parameter('baudrate', 115200)
#         self.declare_parameter('timeout',  1.0)

#         port     = self.get_parameter('port').value
#         baudrate = self.get_parameter('baudrate').value
#         timeout  = self.get_parameter('timeout').value
#          # DTR low on open prevents Arduino auto-reset, but may cause issues on some platf

#         # ── Serial connection ────────────────────────────────────────────────
#         try:
#             self.ser = serial.Serial()
#             self.ser.port     = port
#             self.ser.baudrate = baudrate
#             self.ser.timeout  = timeout
#             self.ser.dtr      = False    # must be set before open()
#             self.ser.rts      = False    # belt and braces — RTS can also trigger reset
#             self.ser.open()
#             self.get_logger().info(f'Serial connected on {port} at {baudrate} baud')
#         except serial.SerialException as e:
#             self.get_logger().error(f'Failed to open serial port: {e}')
#             raise

#         # Lock to guard serial writes (reader thread + cmd_vel callback)
#         self._write_lock = threading.Lock()

#         # ── Publishers ───────────────────────────────────────────────────────
#         self.odom_pub        = self.create_publisher(Odometry,     '/odom',         10)
#         self.imu_pub         = self.create_publisher(Imu,          '/imu/data',     10)
#         self.battery_pub     = self.create_publisher(BatteryState, '/battery',      10)
#         self.arduino_cmd_pub = self.create_publisher(String,       '/arduino_cmd',  10)

#         # ── Subscriber ───────────────────────────────────────────────────────
#         self.cmd_vel_sub = self.create_subscription(
#             Twist, '/cmd_vel', self.cmd_vel_callback, 10)

#         # ── TF Broadcaster ───────────────────────────────────────────────────
#         self.tf_broadcaster = TransformBroadcaster(self)

#         # ── Odometry state ───────────────────────────────────────────────────
#         self.x          = 0.0
#         self.y          = 0.0
#         self.yaw        = 0.0
#         self.last_stamp = None

#         # ── Reader thread ────────────────────────────────────────────────────
#         # Runs continuously and independently of the ROS executor so that
#         # cmd_vel writes and serial reads are truly parallel.
#         self._stop_event = threading.Event()
#         self._reader_thread = threading.Thread(
#             target=self._reader_loop, daemon=True)
#         self._reader_thread.start()

#         self.get_logger().info('SmartWheel serial bridge started — reader thread running')

#     # ─── CMD_VEL CALLBACK (ROS executor thread) ──────────────────────────────
#     def cmd_vel_callback(self, msg: Twist):
#         """Write velocity command to Arduino immediately when /cmd_vel arrives."""
#         cmd = f'V{msg.linear.x:.4f},W{msg.angular.z:.4f}\n'
#         with self._write_lock:
#             try:
#                 self.ser.write(cmd.encode('utf-8'))
#             except serial.SerialException as e:
#                 self.get_logger().error(f'Serial write failed: {e}')
#                 return

#         # Publish what was sent (strip trailing newline for readability)
#         out = String()
#         out.data = cmd.strip()
#         self.arduino_cmd_pub.publish(out)

#     # ─── READER LOOP (dedicated daemon thread) ───────────────────────────────
#     def _reader_loop(self):
#         """
#         Runs in its own thread. Blocks on readline() so it consumes every
#         line from the Arduino as soon as it arrives — no timer drift.
#         """
#         while not self._stop_event.is_set():
#             try:
#                 raw = self.ser.readline()
#             except serial.SerialException as e:
#                 if not self._stop_event.is_set():
#                     self.get_logger().error(f'Serial read error: {e}')
#                 break

#             try:
#                 line = raw.decode('utf-8').strip()
#             except UnicodeDecodeError:
#                 continue

#             if not line:
#                 continue

#             # Discard Arduino startup banners and echoed commands
#             if line.startswith(_IGNORE_PREFIXES):
#                 continue

#             data = self._parse_line(line)
#             if data is None:
#                 self.get_logger().warn(f'Could not parse line: {line}')
#                 continue

#             now = self.get_clock().now().to_msg()
#             self._publish_imu(data, now)
#             self._publish_battery(data, now)
#             self._publish_odom(data, now)

#     # ─── PARSER ──────────────────────────────────────────────────────────────
#     _PATTERN = re.compile(
#         r'L(-?[\d.]+),R(-?[\d.]+),'
#         r'AX(-?[\d.]+),AY(-?[\d.]+),AZ(-?[\d.]+),'
#         r'GX(-?[\d.]+),GY(-?[\d.]+),GZ(-?[\d.]+),'
#         r'BV(-?[\d.]+),BT(-?[\d.]+)'
#     )

#     def _parse_line(self, line: str):
#         m = self._PATTERN.match(line)
#         if not m:
#             return None
#         return {
#             'rpm_left':  float(m.group(1)),
#             'rpm_right': float(m.group(2)),
#             'ax': float(m.group(3)),
#             'ay': float(m.group(4)),
#             'az': float(m.group(5)),
#             'gx': float(m.group(6)),
#             'gy': float(m.group(7)),
#             'gz': float(m.group(8)),
#             'bat_volt': float(m.group(9)),
#             'bat_temp': float(m.group(10)),
#         }

#     # ─── ODOMETRY ────────────────────────────────────────────────────────────
#     def _publish_odom(self, data: dict, stamp):
#         now = stamp

#         if self.last_stamp is None:
#             self.last_stamp = now
#             return

#         dt_ns = (
#             (now.sec - self.last_stamp.sec) * 1_000_000_000
#             + (now.nanosec - self.last_stamp.nanosec)
#         )
#         dt = dt_ns / 1e9 if dt_ns > 0 else 0.05
#         self.last_stamp = now

#         v_left  = data['rpm_left']  * RPM_TO_RAD_S * WHEEL_RADIUS
#         v_right = data['rpm_right'] * RPM_TO_RAD_S * WHEEL_RADIUS

#         v_linear  = (v_left + v_right) / 2.0
#         v_angular = (v_right - v_left) / WHEEL_SEPARATION

#         self.yaw += v_angular * dt
#         self.x   += v_linear * math.cos(self.yaw) * dt
#         self.y   += v_linear * math.sin(self.yaw) * dt

#         q = self._yaw_to_quaternion(self.yaw)

#         odom = Odometry()
#         odom.header.stamp    = stamp
#         odom.header.frame_id = 'odom'
#         odom.child_frame_id  = 'base_footprint'

#         odom.pose.pose.position.x  = self.x
#         odom.pose.pose.position.y  = self.y
#         odom.pose.pose.position.z  = 0.0
#         odom.pose.pose.orientation = q

#         odom.pose.covariance[0]  = 0.01
#         odom.pose.covariance[7]  = 0.01
#         odom.pose.covariance[35] = 0.05

#         odom.twist.twist.linear.x  = v_linear
#         odom.twist.twist.angular.z = v_angular

#         odom.twist.covariance[0]  = 0.01
#         odom.twist.covariance[35] = 0.05

#         self.odom_pub.publish(odom)

#         tf = TransformStamped()
#         tf.header.stamp            = stamp
#         tf.header.frame_id         = 'odom'
#         tf.child_frame_id          = 'base_footprint'
#         tf.transform.translation.x = self.x
#         tf.transform.translation.y = self.y
#         tf.transform.translation.z = 0.0
#         tf.transform.rotation      = q

#         self.tf_broadcaster.sendTransform(tf)

#     # ─── IMU ─────────────────────────────────────────────────────────────────
#     def _publish_imu(self, data: dict, stamp):
#         imu = Imu()
#         imu.header.stamp    = stamp
#         imu.header.frame_id = 'imu_link'

#         imu.linear_acceleration.x = data['ax']
#         imu.linear_acceleration.y = data['ay']
#         imu.linear_acceleration.z = data['az']

#         imu.angular_velocity.x = data['gx']
#         imu.angular_velocity.y = data['gy']
#         imu.angular_velocity.z = data['gz']

#         imu.orientation_covariance[0] = -1.0  # no orientation estimate

#         imu.linear_acceleration_covariance[0] = 0.01
#         imu.linear_acceleration_covariance[4] = 0.01
#         imu.linear_acceleration_covariance[8] = 0.01

#         imu.angular_velocity_covariance[0] = 0.005
#         imu.angular_velocity_covariance[4] = 0.005
#         imu.angular_velocity_covariance[8] = 0.005

#         self.imu_pub.publish(imu)

#     # ─── BATTERY ─────────────────────────────────────────────────────────────
#     def _publish_battery(self, data: dict, stamp):
#         bat = BatteryState()
#         bat.header.stamp    = stamp
#         bat.header.frame_id = 'base_link'

#         bat.voltage     = data['bat_volt']
#         bat.temperature = data['bat_temp']
#         bat.current     = float('nan')
#         bat.charge      = float('nan')
#         bat.capacity    = float('nan')
#         bat.design_capacity = float('nan')
#         bat.percentage  = float('nan')

#         bat.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
#         bat.power_supply_health     = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
#         bat.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
#         bat.present = True

#         self.battery_pub.publish(bat)

#     # ─── HELPERS ─────────────────────────────────────────────────────────────
#     @staticmethod
#     def _yaw_to_quaternion(yaw: float) -> Quaternion:
#         q = Quaternion()
#         q.x = 0.0
#         q.y = 0.0
#         q.z = math.sin(yaw / 2.0)
#         q.w = math.cos(yaw / 2.0)
#         return q

#     def destroy_node(self):
#         self._stop_event.set()
#         try:
#             with self._write_lock:
#                 self.ser.write(b'V0.0000,W0.0000\n')
#         except Exception:
#             pass
#         if self.ser.is_open:
#             self.ser.close()
#         self._reader_thread.join(timeout=1.0)
#         super().destroy_node()


# def main(args=None):
#     rclpy.init(args=args)
#     node = SerialBridgeNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()



#!/usr/bin/env python3
"""
SmartWheel Serial Bridge Node
Bidirectional communication between ROS 2 and Arduino (RAFEEQ / Team-06).

Arduino → RPi (20 Hz CSV):
  L<rpm>,R<rpm>,AX<m/s²>,AY<m/s²>,AZ<m/s²>,GX<rad/s>,GY<rad/s>,GZ<rad/s>,BV<V>,BT<°C>

RPi → Arduino (on cmd_vel):
  V<linear_x>,W<angular_z>\n

Publishes:
  /odom          (nav_msgs/Odometry)
  /imu/data      (sensor_msgs/Imu)
  /battery       (sensor_msgs/BatteryState)
  /arduino_cmd   (std_msgs/String)  — echoes every cmd sent to Arduino
  /tf            (odom → base_footprint transform)

Subscribes:
  /cmd_vel          (geometry_msgs/Twist)
  /navigation_goal  (std_msgs/String)
    Accepted values:
      move_forward  → V0.5000,W0.0000  for 3 s then stop
      move_backward → V-0.5000,W0.0000 for 3 s then stop
      turn_left     → V0.0000,W-2.0000 for 3 s then stop
      turn_right    → V0.0000,W2.0000  for 3 s then stop
      stop          → V0.0000,W0.0000  immediately

Usage:
  ros2 run serial_node serial_bridge --ros-args -p port:=/dev/ttyACM0
"""

import threading
import re
import math

import rclpy
from rclpy.node import Node
import serial

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, BatteryState
from geometry_msgs.msg import Twist, TransformStamped, Quaternion
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


# ─── ROBOT PHYSICAL PARAMETERS (locked — matches Arduino firmware) ────────────
WHEEL_RADIUS     = 0.130   # metres
WHEEL_SEPARATION = 0.348   # metres
RPM_TO_RAD_S     = 2.0 * math.pi / 60.0

# Duration (seconds) to hold a navigation_goal command before auto-stopping
NAV_GOAL_DURATION = 3.0

# Velocity map for /navigation_goal commands  (linear_x, angular_z)
_NAV_GOAL_VELOCITIES = {
    'move_forward':  ( 0.5,  0.0),
    'move_backward': (-0.5,  0.0),
    'turn_right':    ( 0.0,  2.0),
    'turn_left':     ( 0.0, -2.0),
    'stop':          ( 0.0,  0.0),
}

# Arduino startup lines to silently discard
_IGNORE_PREFIXES = ('RAFEEQ', 'Waiting', 'ESC', 'V')


class SerialBridgeNode(Node):

    def __init__(self):
        super().__init__('serial_bridge')

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter('port',     '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout',  1.0)

        port     = self.get_parameter('port').value
        baudrate = self.get_parameter('baudrate').value
        timeout  = self.get_parameter('timeout').value

        # ── Serial connection ────────────────────────────────────────────────
        try:
            self.ser = serial.Serial()
            self.ser.port     = port
            self.ser.baudrate = baudrate
            self.ser.timeout  = timeout
            self.ser.dtr      = False   # prevent Arduino auto-reset on open
            self.ser.rts      = False
            self.ser.open()
            self.get_logger().info(f'Serial connected on {port} at {baudrate} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            raise

        # Lock to guard serial writes (reader thread + callbacks)
        self._write_lock = threading.Lock()

        # Timer handle for the auto-stop after a navigation_goal command.
        # Protected by _motion_lock so concurrent goals don't race.
        self._stop_timer   = None
        self._motion_lock  = threading.Lock()

        # ── Publishers ───────────────────────────────────────────────────────
        self.odom_pub        = self.create_publisher(Odometry,     '/odom',        10)
        self.imu_pub         = self.create_publisher(Imu,          '/imu/data',    10)
        self.battery_pub     = self.create_publisher(BatteryState, '/battery',     10)
        self.arduino_cmd_pub = self.create_publisher(String,       '/arduino_cmd', 10)

        # ── Subscribers ──────────────────────────────────────────────────────
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        self.nav_goal_sub = self.create_subscription(
            String, '/navigation_goal', self.navigation_goal_callback, 10)

        # ── TF Broadcaster ───────────────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── Odometry state ───────────────────────────────────────────────────
        self.x          = 0.0
        self.y          = 0.0
        self.yaw        = 0.0
        self.last_stamp = None

        # ── Reader thread ────────────────────────────────────────────────────
        self._stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self.get_logger().info('SmartWheel serial bridge started — reader thread running')

    # ─── CMD_VEL CALLBACK ────────────────────────────────────────────────────
    def cmd_vel_callback(self, msg: Twist):
        """Write velocity command to Arduino immediately when /cmd_vel arrives."""
        cmd = f'V{msg.linear.x:.4f},W{msg.angular.z:.4f}\n'
        self._serial_write(cmd)

    # ─── NAVIGATION_GOAL CALLBACK ────────────────────────────────────────────
    def navigation_goal_callback(self, msg: String):
        """
        Handle discrete navigation commands published to /navigation_goal.

        For every command except 'stop', the corresponding velocity is sent
        to the Arduino for NAV_GOAL_DURATION seconds, after which a stop
        command (V0.0,W0.0) is sent automatically.

        A new command arriving before the timer fires cancels the previous
        pending stop and immediately applies the new velocity.
        """
        goal = msg.data.strip().lower()

        if goal not in _NAV_GOAL_VELOCITIES:
            self.get_logger().warn(
                f'Unknown navigation_goal "{goal}". '
                f'Valid: {list(_NAV_GOAL_VELOCITIES.keys())}')
            return

        linear_x, angular_z = _NAV_GOAL_VELOCITIES[goal]
        cmd = f'V{linear_x:.4f},W{angular_z:.4f}\n'

        with self._motion_lock:
            # Cancel any pending auto-stop from a previous goal
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None

            # Send the new velocity
            self._serial_write(cmd)
            self.get_logger().info(
                f'navigation_goal "{goal}" → {cmd.strip()}')

            if goal == 'stop':
                # 'stop' takes effect immediately — no auto-stop timer needed
                return

            # Schedule an auto-stop after NAV_GOAL_DURATION seconds
            self._stop_timer = threading.Timer(
                NAV_GOAL_DURATION, self._auto_stop)
            self._stop_timer.daemon = True
            self._stop_timer.start()

    # ─── AUTO-STOP (called by threading.Timer) ────────────────────────────────
    def _auto_stop(self):
        """Send a stop command after the timed motion window expires."""
        with self._motion_lock:
            self._stop_timer = None   # timer has fired; clear the handle

        stop_cmd = 'V0.0000,W0.0000\n'
        self._serial_write(stop_cmd)
        self.get_logger().info('navigation_goal auto-stop sent')

    # ─── SERIAL WRITE HELPER ─────────────────────────────────────────────────
    def _serial_write(self, cmd: str):
        """Thread-safe serial write + /arduino_cmd echo."""
        with self._write_lock:
            try:
                self.ser.write(cmd.encode('utf-8'))
            except serial.SerialException as e:
                self.get_logger().error(f'Serial write failed: {e}')
                return

        out = String()
        out.data = cmd.strip()
        self.arduino_cmd_pub.publish(out)

    # ─── READER LOOP (dedicated daemon thread) ───────────────────────────────
    def _reader_loop(self):
        while not self._stop_event.is_set():
            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                if not self._stop_event.is_set():
                    self.get_logger().error(f'Serial read error: {e}')
                break

            try:
                line = raw.decode('utf-8').strip()
            except UnicodeDecodeError:
                continue

            if not line:
                continue

            if line.startswith(_IGNORE_PREFIXES):
                continue

            data = self._parse_line(line)
            if data is None:
                self.get_logger().warn(f'Could not parse line: {line}')
                continue

            now = self.get_clock().now().to_msg()
            self._publish_imu(data, now)
            self._publish_battery(data, now)
            self._publish_odom(data, now)

    # ─── PARSER ──────────────────────────────────────────────────────────────
    _PATTERN = re.compile(
        r'L(-?[\d.]+),R(-?[\d.]+),'
        r'AX(-?[\d.]+),AY(-?[\d.]+),AZ(-?[\d.]+),'
        r'GX(-?[\d.]+),GY(-?[\d.]+),GZ(-?[\d.]+),'
        r'BV(-?[\d.]+),BT(-?[\d.]+)'
    )

    def _parse_line(self, line: str):
        m = self._PATTERN.match(line)
        if not m:
            return None
        return {
            'rpm_left':  float(m.group(1)),
            'rpm_right': float(m.group(2)),
            'ax': float(m.group(3)),
            'ay': float(m.group(4)),
            'az': float(m.group(5)),
            'gx': float(m.group(6)),
            'gy': float(m.group(7)),
            'gz': float(m.group(8)),
            'bat_volt': float(m.group(9)),
            'bat_temp': float(m.group(10)),
        }

    # ─── ODOMETRY ────────────────────────────────────────────────────────────
    def _publish_odom(self, data: dict, stamp):
        now = stamp

        if self.last_stamp is None:
            self.last_stamp = now
            return

        dt_ns = (
            (now.sec - self.last_stamp.sec) * 1_000_000_000
            + (now.nanosec - self.last_stamp.nanosec)
        )
        dt = dt_ns / 1e9 if dt_ns > 0 else 0.05
        self.last_stamp = now

        v_left  = data['rpm_left']  * RPM_TO_RAD_S * WHEEL_RADIUS
        v_right = data['rpm_right'] * RPM_TO_RAD_S * WHEEL_RADIUS

        v_linear  = (v_left + v_right) / 2.0
        v_angular = (v_right - v_left) / WHEEL_SEPARATION

        self.yaw += v_angular * dt
        self.x   += v_linear * math.cos(self.yaw) * dt
        self.y   += v_linear * math.sin(self.yaw) * dt

        q = self._yaw_to_quaternion(self.yaw)

        odom = Odometry()
        odom.header.stamp    = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_footprint'

        odom.pose.pose.position.x  = self.x
        odom.pose.pose.position.y  = self.y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = q

        odom.pose.covariance[0]  = 0.01
        odom.pose.covariance[7]  = 0.01
        odom.pose.covariance[35] = 0.05

        odom.twist.twist.linear.x  = v_linear
        odom.twist.twist.angular.z = v_angular

        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[35] = 0.05

        self.odom_pub.publish(odom)

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
    def _publish_imu(self, data: dict, stamp):
        imu = Imu()
        imu.header.stamp    = stamp
        imu.header.frame_id = 'imu_link'

        imu.linear_acceleration.x = data['ax']
        imu.linear_acceleration.y = data['ay']
        imu.linear_acceleration.z = data['az']

        imu.angular_velocity.x = data['gx']
        imu.angular_velocity.y = data['gy']
        imu.angular_velocity.z = data['gz']

        imu.orientation_covariance[0] = -1.0  # no orientation estimate

        imu.linear_acceleration_covariance[0] = 0.01
        imu.linear_acceleration_covariance[4] = 0.01
        imu.linear_acceleration_covariance[8] = 0.01

        imu.angular_velocity_covariance[0] = 0.005
        imu.angular_velocity_covariance[4] = 0.005
        imu.angular_velocity_covariance[8] = 0.005

        self.imu_pub.publish(imu)

    # ─── BATTERY ─────────────────────────────────────────────────────────────
    def _publish_battery(self, data: dict, stamp):
        bat = BatteryState()
        bat.header.stamp    = stamp
        bat.header.frame_id = 'base_link'

        bat.voltage     = data['bat_volt']
        bat.temperature = data['bat_temp']
        bat.current     = float('nan')
        bat.charge      = float('nan')
        bat.capacity    = float('nan')
        bat.design_capacity = float('nan')
        bat.percentage  = float('nan')

        bat.power_supply_status     = BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        bat.power_supply_health     = BatteryState.POWER_SUPPLY_HEALTH_UNKNOWN
        bat.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LION
        bat.present = True

        self.battery_pub.publish(bat)

    # ─── HELPERS ─────────────────────────────────────────────────────────────
    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> Quaternion:
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def destroy_node(self):
        # Cancel any pending motion timer
        with self._motion_lock:
            if self._stop_timer is not None:
                self._stop_timer.cancel()
                self._stop_timer = None

        self._stop_event.set()
        try:
            with self._write_lock:
                self.ser.write(b'V0.0000,W0.0000\n')
        except Exception:
            pass
        if self.ser.is_open:
            self.ser.close()
        self._reader_thread.join(timeout=1.0)
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