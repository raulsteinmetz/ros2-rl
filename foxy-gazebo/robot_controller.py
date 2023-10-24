# ROS 2 Python client library
import rclpy

# ROS 2 message and service data types
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from rclpy.qos import QoSProfile

# ROS 2 node class
from rclpy.node import Node

# Other utilities
from rclpy.duration import Duration
from rclpy.time import Time
import numpy as np


class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # Publishers, Subscribers, and Clients
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Usually, simulation environments provide a service to reset the world or robot
        self.reset_client = self.create_client(Empty, '/reset_simulation')  # Adjust the service name

        self.odom_data = None
        self.scan_data = None

    def odom_callback(self, msg):
        self.get_logger().info('Odometry data received')
        self.odom_data = msg

    def scan_callback(self, msg):
        self.get_logger().info('Scan data received')
        self.scan_data = msg

    def control_and_publish(self, linear_vel=0.2, angular_vel=0.1):
        if self.scan_data is None or self.odom_data is None:
            return
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def step(self, action):
        """
        Execute action, observe the next state and reward.
        Args:
        - action: Tuple with (linear_vel, angular_vel)

        Returns:
        - next_state: Your observation space, based on odom and scan data
        - reward: Reward based on the action taken
        - done: Whether the episode is finished
        - info: Additional information, if needed
        """
        done = False
        info = {}
        reward = 0

        # Perform the action
        self.control_and_publish(action[0], action[1])

        # Here, you would usually have a slight delay or wait for the next observation to be ready.
        # Execute callbacks and spin the node for a moment to update data.
        rclpy.spin_once(self, timeout_sec=1.0)

        # Get the next state
        # For simplicity, let's say it's just the position for now. It could be more complex.
        try:
            next_state = np.array([self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y])
        except:
            next_state = np.zeros(5)

        # print(next_state)

        # Calculate reward
        # This is highly specific to your task. For example, did the robot reach a goal? Did it crash?
        reward = self.calculate_reward()

        # Check if the episode is done
        # This could be based on a goal being reached, a crash, or some other criterion.
        done = self.is_episode_done()

        return next_state, reward, done, info

    def reset_simulation(self):
        """Resets the simulation to start a new episode."""

        # In a typical setup, there's a service call that triggers the simulation environment to reset.
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting again...')
        
        self.reset_client.call_async(req)

        # Wait for the reset to actually happen. You may need to reinitialize some variables.
        while not self.is_reset_complete():
            # You could also add a timeout here for safety.
            pass  # Maybe add a slight delay here

        # Once reset, you might return an initial observation
        self.control_and_publish(0.1, 0.1)


        try:
            initial_state = np.array([self.odom_data.pose.pose.position.x, self.odom_data.pose.pose.position.y])
        except:
            initial_state = np.zeros(5) # might need to ajust this
        return initial_state

    def calculate_reward(self):
        """
        Calculate the reward for the current step.
        This method needs to be filled out based on what constitutes a reward in your environment.
        """
        # This is just a placeholder. Your actual reward calculation could be much more complex.
        reward = 0
        return reward

    def is_episode_done(self):
        """
        Determines if the episode is done based on your criteria (e.g., robot reached a goal or crashed).
        """
        # Placeholder: your actual condition for an episode being over could be different.
        done = False
        return done

    def is_reset_complete(self):
        """
        Check whether the reset is complete. This might be checking certain state variables or 
        waiting for a specific signal or callback from the simulation environment.
        """
        # Placeholder: your actual check for whether reset is complete could be different.
        reset_complete = True  # This is where you’d check if the reset is indeed complete.
        return reset_complete
