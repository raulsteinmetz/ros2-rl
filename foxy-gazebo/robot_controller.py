from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.odom_data = 0
        self.scan_data = 0

    def odom_callback(self, msg):
        self.odom_data = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def control_and_publish(self, linear_vel=0.2, angular_vel=0.1):
        if self.scan_data is None:
            return

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)
