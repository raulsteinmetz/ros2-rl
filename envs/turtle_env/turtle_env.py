import rclpy
import random
import math
import numpy as np
import torch as T
from time import sleep
from os import system
from datetime import datetime
from matplotlib import pyplot as plt
from rclpy.node import Node
from std_srvs.srv import Empty 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Point, Quaternion
from envs.turtle_env.target import generate_target_sdf
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetEntityState, SetEntityState
from torch.utils.tensorboard import SummaryWriter

REACH_TRESHOLD = 0.3
LIDAR_DISCRETIZATION = 10
LIDAR_MAX_RANGE = 3.5
COLISION_TRESHOLD = 0.19

class Env(Node):
    """
    A class representing the environment in which the TurtleBot operates. It is a ROS node 
    that interacts with the Gazebo simulation environment.
    """
    def __init__(self):
        """
        Initialize the environment node.
        """
        super().__init__("trainer_node")
        # Setup publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.reset_client = self.create_client(Empty, '/reset_simulation')
        self.get_entity_state_client = self.create_client(GetEntityState, '/demo/get_entity_state')
        self.set_entity_state_client = self.create_client(SetEntityState, '/demo/set_entity_state')

        self.reset_info()
        self.init_properties()

    def reset_info(self):
        """
        Reset the internal state information of the environment.
        """
        self.odom_data = None
        self.scan_data = None

    def init_properties(self):
        """
        Initialize the properties of the environment.
        """
        self.num_states = 14
        self.num_actions = 2
        self.action_upper_bound = .25
        self.action_lower_bound = -.25

    def odom_callback(self, msg):
        """
        Callback function for odometry data.

        :param msg: The odometry data message.
        """
        self.odom_data = msg

    def scan_callback(self, msg):
        """
        Callback function for LIDAR scan data.

        :param msg: The LIDAR scan data message.
        """
        self.scan_data = msg

    def get_state(self, linear_vel, angular_vel):
        """
        Get the current state of the environment.

        :param linear_vel: Current linear velocity of the robot.
        :param angular_vel: Current angular velocity of the robot.
        :return: The current state of the environment, robot's x and y coordinates, and LIDAR data.
        """
        self.reset_info()
        rclpy.spin_once(self, timeout_sec=0.5)
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        turtle_x = self.odom_data.pose.pose.position.x
        turtle_y = self.odom_data.pose.pose.position.y

        # Convert quaternion to Euler angles
        q = self.odom_data.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

        angle_to_target = math.atan2(self.target_y - turtle_y, self.target_x - turtle_x) - yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))

        distance_to_target = math.sqrt((self.target_x - turtle_x) ** 2 + (self.target_y - turtle_y) ** 2)

        lidar_readings = self.scan_data.ranges
        num_samples = LIDAR_DISCRETIZATION
        step = (len(lidar_readings) - 1) // (num_samples - 1)
        lidar = [lidar_readings[i * step] if lidar_readings[i * step] != float('inf') else LIDAR_MAX_RANGE for i in range(num_samples)]

        state = lidar + [distance_to_target, angle_to_target, linear_vel, angular_vel]

        return state, turtle_x, turtle_y, lidar



    def reset_simulation(self, stage):
        """
        Reset the simulation environment for a new episode.

        :param stage: The stage of training or simulation scenario.
        :return: The initial state of the environment after reset.
        """
        # Resetting the environment
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting again...')
        
        self.reset_client.call_async(req)
        self.despawn_target_mark()

        self.spawn_target_in_environment(stage)
        
        self.publish_vel(0.0, 0.0)

        # Wait for fresh sensor reads to avoid last episode's data
        self.scan_data = None
        self.odom_data = None
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)
            sleep(0.1)

        state, _, _, _ = self.get_state(0, 0)
        return state

    def publish_vel(self, linear_vel, angular_vel):
        """
        Publish velocity commands to the robot.

        :param linear_vel: Linear velocity command.
        :param angular_vel: Angular velocity command.
        """
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def get_reward(self, turtle_x, turtle_y, target_x, target_y, lidar_32, steps, max_steps):
        """
        Calculate the reward based on the current state of the environment.

        :param turtle_x: Current x-coordinate of the robot.
        :param turtle_y: Current y-coordinate of the robot.
        :param target_x: x-coordinate of the target.
        :param target_y: y-coordinate of the target.
        :param lidar_32: LIDAR sensor data.
        :param steps: Current step count in the episode.
        :param max_steps: Maximum number of steps in an episode.
        :return: Reward for the current state and a boolean indicating if the episode is done.
        """
        reward = 0
        done = False

        # Calculate distance to the target
        distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2) 

        # Determine reward and done state based on proximity to target and collision data
        if distance < REACH_TRESHOLD:
            done = True
            reward = 100
        elif np.min(lidar_32) < COLISION_TRESHOLD:
            done = True
            reward = -10
        elif steps >= (max_steps - 1):
            done = True
            reward = -10

        return reward, done



    def spawn_target_in_environment(self, stage):
        """
        Spawn a target in the environment at a random location depending on the stage of training.

        :param stage: The stage of training, which determines the spawning strategy for the target.
        """
        # Wait for spawn service to be available
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        # Generate a random position for the target mark based on the stage
        # Each stage has different target spawning strategies
        self.target_x, self.target_y = self.generate_random_target_position(stage)
        fixed_z = 0.01  # Fixed z coordinate, just above ground level

        request = SpawnEntity.Request()
        request.name = 'target_mark'
        request.xml = generate_target_sdf(self.target_x, self.target_y, fixed_z)

        # Calling the service twice due to a bug in Gazebo
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_spawn_result(future, fixed_z)

        sleep(0.5)
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def despawn_target_mark(self):
        """
        Despawn the target mark from the environment.
        """
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DeleteEntity service not available, waiting again...')

        request = DeleteEntity.Request()
        request.name = 'target_mark'
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        self.handle_despawn_result(future)

    def step(self, action, step, max_steps_per_episode, discrete, stage):
        """
        Execute a step in the environment based on the given action.

        :param action: The action to be executed.
        :param step: Current step count in the episode.
        :param max_steps_per_episode: Maximum number of steps in an episode.
        :param discrete: Flag indicating if the action space is discrete.
        :param stage: Current stage of the training.
        :return: Reward for the step, a boolean indicating if the episode is done, and the next state.
        """
        rclpy.spin_once(self, timeout_sec=0.5)
        self.publish_action(action, discrete)

        rclpy.spin_once(self, timeout_sec=0.5)

        if discrete:
            state_, turtle_x, turtle_y, lidar32 = self.get_state(action, action)
        else:
            state_, turtle_x, turtle_y, lidar32 = self.get_state(*self.process_continuous_action(action))

        reward, done = self.get_reward(turtle_x, turtle_y, self.target_x, self.target_y, lidar32, step, max_steps_per_episode)

        return reward, done, state_
    
    def generate_random_target_position(self, stage):
        """
        Generate a random position for the target based on the training stage.

        :param stage: The stage of training.
        :return: A tuple (x, y) representing the target's position.
        """
                # generate random position for the target mark
        if stage == 1:
            self.target_x = random.uniform(-1.90, 1.90)  # Adjust the range to fit your environment
            self.target_y = random.uniform(-1.90, 1.90)
        elif stage == 2:
            tgt_positions = [(-1.7, -1.7), (1.7, -1.7), (-1.7, 1.7), (1.7, 1.7),
                             (0, 1.5), (1.5, 0), (-1.5, 0), (0, -1.5),
                             (1, 1.7), (-1, 1.7), (1, -1.7), (-1, -1.7),
                             (1.7, 1), (1.7, -1), (-1.7, 1), (-1.7, -1),
                             (1.7, -0.5), (1.7, 0.5), (-1.7, 0.5), (-1.7, -0.5),
                             (0.5, -1.7), (-0.5, -1.7), (0.5, 1.7), (-0.5, 1.7)]
            self.target_x, self.target_y = random.choice(tgt_positions)
        elif stage == 3:
            area = np.random.randint(0, 3)
            if area == 0:
                self.target_x = random.uniform(-1.8, -1.9)
                self.target_y = random.uniform(-1.9, 1.9)
            elif area == 1:
                self.target_x = random.uniform(-1.9, 1.9)
                self.target_y = random.uniform(-1.8, -1.9)
            elif area == 2:
                self.target_x = random.uniform(1.9, 1.1)
                self.target_y = random.uniform(0.4, 1.1)
        elif stage == 4:
            area = np.random.randint(0, 3)
            if area == 0:
                self.target_x = random.uniform(1.8, 1.9)
                self.target_y = random.uniform(-1.8, -1.9)
            elif area == 1:
                self.target_x = random.uniform(-1.8, -1.9)
                self.target_y = random.uniform(-1.9, 1.9)
            elif area == 2:
                self.target_x = random.uniform(1.9, 1.1)
                self.target_y = random.uniform(0.4, 1.1)

        return self.target_x, self.target_y

    def handle_spawn_result(self, future, fixed_z):
        """
        Handle the result of a spawn entity request.

        :param future: The future object returned from the spawn entity service.
        :param fixed_z: The fixed z-coordinate of the spawned entity.
        """
        if future.result() is not None:
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={self.target_x}, y={self.target_y}, z={fixed_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")

    def handle_despawn_result(self, future):
        """
        Handle the result of a despawn entity request.

        :param future: The future object returned from the delete entity service.
        """
        if future.result() is not None and future.result().success:
            self.get_logger().info("Mark deleted successfully.")
        else:
            self.get_logger().info("No mark to delete or deletion failed.")

    def publish_action(self, action, discrete):
        """
        Publish the robot's velocity based on the given action.

        :param action: The action to be executed.
        :param discrete: Flag indicating if the action space is discrete.
        """
        if discrete:
            # Handle discrete actions
            velocities = [(0.1, -.8), (0.1, -.4), (0.1, 0.0), (0.1, .4), (0.1, .8)]
            linear_vel, angular_vel = velocities[action]
        else:
            # Handle continuous actions
            linear_vel = np.abs(float(action[0]))
            angular_vel = float(action[1]) * 2

        self.publish_vel(linear_vel, angular_vel)

    def process_continuous_action(self, action):
        """
        Process a continuous action into linear and angular velocities.

        :param action: The continuous action array.
        :return: A tuple of linear and angular velocities.
        """
        linear_vel = np.abs(float(action[0]))
        angular_vel = float(action[1]) * 2
        return linear_vel, angular_vel
