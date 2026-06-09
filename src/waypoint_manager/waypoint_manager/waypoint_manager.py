import os
import math
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory

class WaypointManagerNode(Node):
    def __init__(self):
        super().__init__('waypoint_manager')

        # Load waypoints from yaml
        pkg_share = get_package_share_directory('waypoint_manager')
        yaml_path = os.path.join(pkg_share, 'config', 'waypoints.yaml')

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            self.waypoints = data['waypoints']

        self.get_logger().info(f'Loaded waypoints: {list(self.waypoints.keys())}')

        self.sub = self.create_subscription(
            String,
            '/navigation_goal',
            self.location_callback,
            10
        )

        self.pub = self.create_publisher(
            PoseStamped,
            '/goal_pose',
            10
        )

        self.get_logger().info('Waypoint Manager ready')

    def yaw_to_quaternion(self, yaw):
        # Convert yaw angle to quaternion (z, w only for 2D)
        return {
            'qx': 0.0,
            'qy': 0.0,
            'qz': math.sin(yaw / 2.0),
            'qw': math.cos(yaw / 2.0)
        }

    # def location_callback(self, msg):
    #     raw = msg.data.lower().strip()
    #     self.get_logger().info(f'Received: "{raw}"')

    #     if raw.startswith('go_to_'):
    #         location = raw.replace('go_to_', '').replace('_', ' ')
    #     else:
    #         location = raw.replace('_', ' ')

    #     self.get_logger().info(f'Parsed location: "{location}"')

    #     if location not in self.waypoints:
    #         self.get_logger().warn(f'Unknown location: "{location}" — available: {list(self.waypoints.keys())}')
    #         return

    #     wp = self.waypoints[location]
    #     q = self.yaw_to_quaternion(wp['yaw'])

    #     goal = PoseStamped()
    #     goal.header.frame_id = 'map'
    #     goal.header.stamp = self.get_clock().now().to_msg()
    #     goal.pose.position.x = float(wp['x'])
    #     goal.pose.position.y = float(wp['y'])
    #     goal.pose.position.z = 0.0
    #     goal.pose.orientation.x = q['qx']
    #     goal.pose.orientation.y = q['qy']
    #     goal.pose.orientation.z = q['qz']
    #     goal.pose.orientation.w = q['qw']

    #     self.pub.publish(goal)
    #     self.get_logger().info(f'Sent goal for: "{location}"')
    def location_callback(self, msg):
        raw = msg.data.lower().strip()
        self.get_logger().info(f'Received: "{raw}"')

        if raw.startswith('go_to_'):
            location = raw.replace('go_to_', '').replace('_', ' ')
        else:
            location = raw.replace('_', ' ')

        self.get_logger().info(f'Parsed location: "{location}"')

        if location not in self.waypoints:
            self.get_logger().warn(f'Unknown location: "{location}" — available: {list(self.waypoints.keys())}')
            return

        wp = self.waypoints[location]

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(wp['x'])
        goal.pose.position.y = float(wp['y'])
        goal.pose.position.z = float(wp['z'])
        goal.pose.orientation.x = float(wp['qx'])
        goal.pose.orientation.y = float(wp['qy'])
        goal.pose.orientation.z = float(wp['qz'])
        goal.pose.orientation.w = float(wp['qw'])

        self.pub.publish(goal)
        self.get_logger().info(f'Sent goal for: "{location}" at ({wp["x"]}, {wp["y"]})')


def main(args=None):
    rclpy.init(args=args)
    node = WaypointManagerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()