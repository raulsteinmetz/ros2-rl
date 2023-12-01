import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty 
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
import random
import math
from time import sleep
from os import system
import numpy as np
from turtle_env.target import generate_target_sdf
from matplotlib import pyplot as plt


REACH_TRESHOLD = 0.3
LIDAR_DISCRETIZATION = 10
LIDAR_MAX_RANGE = 3.5
COLISION_TRESHOLD = 0.19


class Env(Node):
    def __init__(self):
        super().__init__("trainer_node")
        # publishers and subscribers
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1) # controlling turtlebot
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1) # getting odometry info
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1) # reading turtlebot lidar
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity') # spawning entities in gazebo env
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity') # deleting entities in gazebo env
        self.reset_client = self.create_client(Empty, '/reset_simulation') # resetting simulation

        # internal state
        self.reset_info()
        self.init_properties()

    def reset_info(self):
        # Internal state
        self.odom_data = None
        self.scan_data = None



    def init_properties(self):
        self.num_states = 14
        self.num_actions = 2
        self.action_upper_bound = .25
        self. action_lower_bound = -.25

    def odom_callback(self, msg):
        self.odom_data = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def get_state(self, linear_vel, angular_vel):
        # reset the environment infos, for control purposes
        self.reset_info()
        
        # read scan and odon
        rclpy.spin_once(self, timeout_sec=0.5)
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        
        # robot position
        turtle_x = self.odom_data.pose.pose.position.x
        turtle_y = self.odom_data.pose.pose.position.y


        # getting orientation for angle between robot and target calculation
        q_x = self.odom_data.pose.pose.orientation.x
        q_y = self.odom_data.pose.pose.orientation.y
        q_z = self.odom_data.pose.pose.orientation.z
        q_w = self.odom_data.pose.pose.orientation.w

        # converting quaternion to euler angles (yaw)
        siny_cosp = 2 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # calculating the angle to the target
        angle_to_target = math.atan2(self.target_y - turtle_y, self.target_x - turtle_x) - yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))

        # calculating the euclidean distance to the target
        distance_to_target = math.sqrt((self.target_x - turtle_x) ** 2 + (self.target_y - turtle_y) ** 2)

        # lidar readings
        lidar_readings = self.scan_data.ranges
        num_samples = LIDAR_DISCRETIZATION # discretize 360 reads to 10
        step = (len(lidar_readings) - 1) // (num_samples - 1)
        lidar = [lidar_readings[i * step] for i in range(num_samples)]

        # replace inf distance (no read) for 3.5 (virtual maximum lidar range)
        lidar = [x if x != float('inf') else 3 for x in lidar]

        # state_array
        state = lidar + [distance_to_target, angle_to_target] + [linear_vel, angular_vel]

        return state, turtle_x, turtle_y, lidar



    def reset_simulation(self, stage):
        # resetting env
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting again...')
        
        self.reset_client.call_async(req)
        self.despawn_target_mark()

        self.spawn_target_in_environment(stage)
        
        self.publish_vel(0.0, 0.0)

        # waiting for reads, fixes a bug where the read indicates the last episode's colision
        self.scan_data = None
        self.odom_data = None
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)
            sleep(0.1)

        state, _, _, _, = self.get_state(0, 0)

        return state
    
    
    def publish_vel(self, linear_vel, angular_vel):
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def get_reward(self, turtle_x, turtle_y, target_x, target_y, lidar_32, steps, max_steps):
        reward = 0
        done = False

        # distance to target
        distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2) 

        if distance < REACH_TRESHOLD: # close enough
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
        # wait for spawn service to be available
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # generate random position for the target mark
        if stage == 1:
            self.target_x = random.uniform(-1.90, 1.90)  # Adjust the range to fit your environment
            self.target_y = random.uniform(-1.90, 1.90)
        elif stage == 2:
            area = np.random.randint(0, 4)
            if area == 0: 
                self.target_x = random.uniform(-1.90, -1.60)
                self.target_y = random.uniform(-1.90, -1.60) 
            elif area == 1:
                self.target_x = random.uniform(-1.90, -1.60)
                self.target_y = random.uniform(1.60, 1.90) 
            elif area == 2:
                self.target_x = random.uniform(1.60, 1.90)
                self.target_y = random.uniform(-1.90, -1.60)
            elif area == 3:
                self.target_x = random.uniform(1.60, 1.90)
                self.target_y = random.uniform(1.60, 1.90)



        fixed_z = 0.01  # fixed z coordinate, just above ground level

        request = SpawnEntity.Request()
        request.name = 'target_mark'  # Unique name for the new model
        request.xml = generate_target_sdf(self.target_x, self.target_y, fixed_z)  # Model XML with the random position

        # calling the service twice, fixes a bug where gazebo apparently "goes back in time" and despawns it
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={self.target_x}, y={self.target_y}, z={fixed_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")
        sleep(.5)
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

    def despawn_target_mark(self):
        # check if service is available
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DeleteEntity service not available, waiting again...')

        request = DeleteEntity.Request()
        request.name = 'target_mark'

        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().success:
            self.get_logger().info("Mark deleted successfully.")
        else: # in the first episode there won't be a target to despawn
            # self.get_logger().error("Failed to delete mark.")
            pass


    def step(self, action, step, max_steps_per_episode, discrete):
        rclpy.spin_once(self, timeout_sec=0.5)
        if discrete == True:
            if action == 0:
                self.publish_vel(0.1, -.8)
            elif action == 1:
                self.publish_vel(0.1, -.4)
            elif action == 2:
                self.publish_vel(0.1, 0.0)
            elif action == 3:
                self.publish_vel(0.1, .4)
            elif action == 4:
                self.publish_vel(0.1, .8)
        else:
            self.publish_vel(np.abs(float(action[0])), float(action[1]) * 2)

        rclpy.spin_once(self, timeout_sec=0.5)


        if discrete == True:
            state_, turtle_x, turtle_y, lidar32 = self.get_state(action, action) # passing action twice, so the dimentions of the network remain the same
        else:
            state_, turtle_x, turtle_y, lidar32 = self.get_state(np.abs(float(action[0])), float(action[1]) * 2)

        reward, done = self.get_reward(turtle_x, turtle_y, self.target_x, self.target_y, lidar32, step, max_steps_per_episode)

        return reward, done, state_


class Trainer():
    def __init__(self):
        self.env = Env()

    def train(self, agent, episodes, max_steps, load_models=True, stage=1, discrete=False):
        if load_models:
            agent.load_models()

        acum_rwds = []
        mov_avg_rwds = []
        
        N = 100 # Window size for moving average, e.g., 100 episodes

        best_moving_average = -np.inf

        for episode in range(episodes):
            step = 0
            done = False

            state = self.env.reset_simulation(stage)
            acum_reward = 0

            print('Episode: ', episode)

            while not done and step < max_steps:
                action = agent.choose_action(state)
                reward, done, state_ = self.env.step(action, step, max_steps, discrete)

                agent.remember(state, action, reward, state_, done)
                state = state_
                acum_reward += reward

                agent.learn()


                step += 1

            print("Episode * {} * Acumulated Reward is ==> {}".format(episode, acum_reward))
            acum_rwds.append(acum_reward)

             # Compute moving average
            if episode >= N-1:
                moving_avg = np.mean(acum_rwds[-N:])
                mov_avg_rwds.append(moving_avg)
            else:
                mov_avg_rwds.append(np.mean(acum_rwds[:episode+1]))

            if episode >= N-1:
                if mov_avg_rwds[-1] > best_moving_average:
                    best_moving_average = mov_avg_rwds[-1]
                    agent.save_models()
                    print("Saving best models with moving average reward {}...".format(best_moving_average))


            if episode == 1:
                agent.save_models()

            if episode % 50 == 0:
                # Plot raw rewards and moving average
                plt.plot(acum_rwds, alpha=0.5, label="Raw Reward" if episode == 0 else "")
                plt.plot(mov_avg_rwds, color='red', label="Moving Avg Reward" if episode == 0 else "")
                plt.xlabel("Episode")
                plt.ylabel("Acumulated Reward")
                plt.legend()  # Add legend to the plot
                plt.savefig('acum_rwds.png')

    def kill_env(self):
        self.env.destroy_node()



