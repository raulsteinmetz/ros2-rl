# env imports
import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty  # Service for pausing and unpausing the simulation4
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
import random

# neural network imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

# other
from util import *
from time import sleep
from os import system



class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")

        # Publishers, Subscribers, and Clients
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 1) # might have to ajust the buffers, do not know their influence just yet
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 1)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 1)

        # Clients to pause and unpause the Gazebo simulation
        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')
        
        # Client to spawn entities in Gazebo
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        # Client to delete entities in Gazebo
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')


        # Usually, simulation environments provide a service to reset the world or robot
        self.reset_client = self.create_client(Empty, '/reset_simulation')  # Adjust the service name

        # Internal state
        self.odom_data = None
        self.scan_data = None

        # target
        self.target_y = 0
        self.target_x = 0

        # Start the main RL control loop
        self.rl_control_loop()

    def odom_callback(self, msg):
        self.odom_data = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def get_state(self, linear_vel, angular_vel):
        self.scan_data = None
        self.odom_data = None
        rclpy.spin_once(self, timeout_sec=0.5)

        # Wait for scan and odom read:
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)

        # Turtlebot's position
        turtle_x = self.odom_data.pose.pose.position.x
        turtle_y = self.odom_data.pose.pose.position.y

        # Target's position
        target_x = self.target_x
        target_y = self.target_y

        # Extract the quaternion components
        orientation_q = self.odom_data.pose.pose.orientation
        q_x = orientation_q.x
        q_y = orientation_q.y
        q_z = orientation_q.z
        q_w = orientation_q.w

        # Convert quaternion to Euler angles (yaw)
        siny_cosp = 2 * (q_w * q_z + q_x * q_y)
        cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Calculate the angle to the target
        angle_to_target = math.atan2(target_y - turtle_y, target_x - turtle_x) - yaw
        angle_to_target = math.atan2(math.sin(angle_to_target), math.cos(angle_to_target))

        # Calculate the Euclidean distance to the target
        distance_to_target = math.sqrt((target_x - turtle_x) ** 2 + (target_y - turtle_y) ** 2)

        # Lidar readings
        lidar_readings = self.scan_data.ranges

        # Discretize the lidar readings to only 10
        num_samples = 10
        step = (len(lidar_readings) - 1) // (num_samples - 1)
        lidar_10 = [lidar_readings[i * step] for i in range(num_samples)]

        # Replace 'inf' with '3.5'
        lidar_10 = [x if x != float('inf') else 3 for x in lidar_10]

        # Construct the state array
        state = lidar_10 + [distance_to_target, angle_to_target] + [linear_vel, angular_vel]

        return state, turtle_x, turtle_y, target_x, target_y, lidar_10



    def reset_simulation(self):
        """Resets the simulation to start a new episode."""
    
        # In a typical setup, there's a service call that triggers the simulation environment to reset.
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting again...')
        self.reset_client.call_async(req)
        
        self.despawn_target_sphere()

        self.spawn_target_in_environment()
        
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = 0.0
        cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_publisher.publish(cmd_vel_msg)

        # unpause
        self.call_service_sync(self.unpause_simulation_client, Empty.Request())

        # give it some spins to update the lidar and not bug detect collision
        self.scan_data = None
        self.odom_data = None
    
        while self.scan_data is None or self.odom_data is None:
            rclpy.spin_once(self, timeout_sec=0.5)
            sleep(0.1)

        state, _, _, _, _, _ = self.get_state()

        return state
        

    def get_reward(self, turtle_x, turtle_y, target_x, target_y, lidar_32):
        reward = 0
        done = False

        # distance to target
        distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2) 

        if distance < 0.3:
            print('Episode ended with target reached')
            done = True
            reward = 100
        elif np.min(lidar_32) < 0.19:
            print('Episode ended with collision')
            done = True
            reward = -100

        return reward, done

        


    def rl_control_loop(self):
        num_states = 26
        num_actions = 2

        upper_bound = .25
        lower_bound = -.25

        # Learning rate for actor-critic models
        critic_lr = 0.001
        actor_lr = 0.001

        # Discount factor for future rewards
        gamma = 0.99
        # Used to update target networks
        tau = 0.001

        agent = Agent(num_states, num_actions, upper_bound, lower_bound, gamma, tau, critic_lr, actor_lr, 0.2)

        max_episodes = 5000  # for example
        max_steps_per_episode = 250  # for example

        acum_rwds = []
        mov_avg_rwds = []
        
        N = 100 # Window size for moving average, e.g., 100 episodes

        best_moving_average = -np.inf

        for episode in range(max_episodes):
            step = 0
            done = False


            state = self.reset_simulation()
            acum_reward = 0

            print('Episode: ', episode)

            while rclpy.ok() and not done and step < max_steps_per_episode:
                # pause sim
                rclpy.spin_once(self, timeout_sec=0.5)

                self.call_service_sync(self.pause_simulation_client, Empty.Request())

                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state), 0)

                system('clear')
                print('distance: ', state[24])
                print('angle: ', state[25])

                action = agent.policy(tf_prev_state)[0]

                cmd_vel_msg = Twist()
                cmd_vel_msg.linear.x = np.abs(float(action[0])) # only forward for now
                cmd_vel_msg.angular.z = float(action[1])
                self.cmd_vel_publisher.publish(cmd_vel_msg)
                rclpy.spin_once(self, timeout_sec=0.5)

                self.call_service_sync(self.unpause_simulation_client, Empty.Request())

                sleep(0.001)

                rclpy.spin_once(self, timeout_sec=0.5)

                state_, turtle_x, turtle_y, target_x, target_y, lidar32 = self.get_state()

                # pause again
                self.call_service_sync(self.pause_simulation_client, Empty.Request())

                # reward
                reward, done = self.get_reward(turtle_x, turtle_y, target_x, target_y, lidar32)

                agent.mem.record((state, action, reward, state_))
                state = state_
                acum_reward += reward

                agent.learn()
                agent.update_target()


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

            # Plot raw rewards and moving average
            plt.plot(acum_rwds, alpha=0.5, label="Raw Reward" if episode == 0 else "")
            plt.plot(mov_avg_rwds, color='red', label="Moving Avg Reward" if episode == 0 else "")
            plt.xlabel("Episode")
            plt.ylabel("Acumulated Reward")
            plt.legend()  # Add legend to the plot
            plt.savefig('acum_rwds.png')



    def call_service_sync(self, client, request):
        # Synchronous service call
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


    def spawn_target_in_environment(self):
        # Check if spawn_entity service is available
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # Generate random coordinates within a specific range for the sphere's position
        # Note: You should adjust the range to fit the environment of your simulation
        self.target_x = random.uniform(-1, 2)  # For example, within [-5.0, 5.0] range
        self.target_y = random.uniform(-1, 2)
        fixed_z = 0.1  # Fixed z coordinate

        # Dynamic SDF with the random position
        sphere_sdf = f"""
        <?xml version='1.0'?>
        <sdf version='1.6'>
          <model name='target_model'>
            <pose>{self.target_x} {self.target_y} {fixed_z} 0 0 0</pose>
            <link name='link'>
              <collision name='collision'>
                <geometry>
                  <sphere><radius>0.1</radius></sphere>
                </geometry>
              </collision>
              <visual name='visual'>
                <geometry>
                  <sphere><radius>0.1</radius></sphere>
                </geometry>
              </visual>
            </link>
          </model>
        </sdf>
        """

        request = SpawnEntity.Request()
        request.name = 'target_sphere'  # Unique name for the new model
        request.xml = sphere_sdf  # Model XML with the random position

        # Call the service
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)  # Wait for the response

        if future.result() is not None:
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={self.target_x}, y={self.target_y}, z={fixed_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")

        sleep(0.1) # waiting for it to spawn


    def despawn_target_sphere(self):
        # Check if delete_entity service is available
        while not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('DeleteEntity service not available, waiting again...')

        # Request to delete the target sphere
        request = DeleteEntity.Request()
        request.name = 'target_sphere'  # Name of the sphere entity to be deleted

        # Call the service
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)  # Wait for the response

        if future.result() is not None and future.result().success:
            self.get_logger().info("Entity deleted successfully.")
        else:
            self.get_logger().error("Failed to delete entity.")

        #sleep(1)


def main(args=None):
    rclpy.init(args=args)
    robot_controller_node = RobotControllerNode()
            
    rclpy.spin(robot_controller_node)

    robot_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
