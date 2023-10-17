import rclpy
from std_srvs.srv import Empty  # Correct import statement
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import random
import time
from os import system

class RobotControllerNode:
    def __init__(self):
        rclpy.init()
        self.node = rclpy.create_node("robot_controller_node")
        self.cmd_vel_publisher = self.node.create_publisher(Twist, '/cmd_vel', 2000)
        self.odom_subscription = self.node.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        #self.scan_subscription = self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.odom_data = None
        self.scan_data = None

    def odom_callback(self, msg):
        system('clear')
        self.odom_data = msg
        self.node.get_logger().info("Received Odometry data:")
        self.node.get_logger().info(f"Header: {msg.header}")
        self.node.get_logger().info(f"Position: ({msg.pose.pose.position.x}, {msg.pose.pose.position.y}, {msg.pose.pose.position.z})")
        self.node.get_logger().info(f"Orientation: ({msg.pose.pose.orientation.x}, {msg.pose.pose.orientation.y}, {msg.pose.pose.orientation.z}, {msg.pose.pose.orientation.w})")

    def scan_callback(self, msg):
        self.scan_data = msg
        self.node.get_logger().info("Received LaserScan data:")
        self.node.get_logger().info(f"Header: {msg.header}")
        self.node.get_logger().info(f"Ranges: {msg.ranges}")

    def control_and_publish(self):
        # Generate random velocities
        random_linear_vel = 1.0
        random_angular_vel = 0.1

        # Publish data for cmd_vel
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = random_linear_vel
        cmd_vel_msg.angular.z = random_angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)


def main():
    robot_controller_node = RobotControllerNode()

    # Control and publish data
    robot_controller_node.control_and_publish()

    rclpy.spin(robot_controller_node.node)

if __name__ == '__main__':
    main()
