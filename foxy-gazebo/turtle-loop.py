import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty  # Service for pausing and unpausing the simulation4
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
from gazebo_msgs.srv import DeleteEntity
import random

from time import sleep

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__("robot_controller_node")

        # Publishers, Subscribers, and Clients
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

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

        # Start the main RL control loop
        self.rl_control_loop()

    def odom_callback(self, msg):
        print('triggered odom')
        self.odom_data = msg

    def scan_callback(self, msg):
        print('triggered scan')
        self.scan_data = msg

    def reset_simulation(self):
        """Resets the simulation to start a new episode."""

        # In a typical setup, there's a service call that triggers the simulation environment to reset.
        req = Empty.Request()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Reset service not available, waiting again...')
        
        self.reset_client.call_async(req)
        self.despawn_target_sphere()
        self.spawn_target_in_environment()




    def rl_control_loop(self):
        max_episodes = 1000  # for example
        max_steps_per_episode = 4000  # for example

        for episode in range(max_episodes):
          step = 0
          done = False

          self.reset_simulation()

          print('Episode: ', episode)

          # Main loop for reinforcement learning control
          while rclpy.ok() and not done and step < max_steps_per_episode:
              # Pause the simulation
              self.call_service_sync(self.pause_simulation_client, Empty.Request())

              if self.scan_data is None or self.odom_data is None:
                  # If there's no data, there's nothing to do yet, just continue
                  self.call_service_sync(self.unpause_simulation_client, Empty.Request())
                  continue

              sleep(0.001)


              linear_vel = 0.5  # These are placeholders and should be decided by your RL algorithm
              angular_vel = 0.8

              cmd_vel_msg = Twist()
              cmd_vel_msg.linear.x = linear_vel
              cmd_vel_msg.angular.z = angular_vel

              # Publish the control command
              self.cmd_vel_publisher.publish(cmd_vel_msg)

              # Unpause the simulation for the next step to take effect
              self.call_service_sync(self.unpause_simulation_client, Empty.Request())

              # Here you may want to introduce a delay based on your control frequency
              rclpy.spin_once(self, timeout_sec=0.5)  # Adjust the timing as necessary

              step += 1

              # print('ODOM:\n\n')
              # print(self.odom_data)

              # print('SCAN:\n\n')
              # print(self.scan_data)

    def call_service_sync(self, client, request):
        # Synchronous service call
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def control_and_publish(self):
        if self.scan_data is None:
            # Don't do anything if no scan data has been received
            return
        
        # Logic for velocity command based on sensor readings can be added here
        linear_vel = 0.2
        angular_vel = 0.1

        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_vel
        cmd_vel_msg.angular.z = angular_vel
        self.cmd_vel_publisher.publish(cmd_vel_msg)

    def spawn_target_in_environment(self):
        # Check if spawn_entity service is available
        while not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        # Generate random coordinates within a specific range for the sphere's position
        # Note: You should adjust the range to fit the environment of your simulation
        random_x = random.uniform(-1, 2)  # For example, within [-5.0, 5.0] range
        random_y = random.uniform(-1, 2)
        random_z = random.uniform(0.2, 0.2)  # Assuming the ground level is at z=0 and you want the sphere above the ground

        # Dynamic SDF with the random position
        sphere_sdf = f"""
        <?xml version='1.0'?>
        <sdf version='1.6'>
          <model name='target_model'>
            <pose>{random_x} {random_y} {random_z} 0 0 0</pose>
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
            self.get_logger().info(f"Entity spawned successfully at coordinates: x={random_x}, y={random_y}, z={random_z}.")
        else:
            self.get_logger().error("Failed to spawn entity.")

        sleep(1) # waiting for it to spawn


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

        sleep(1)


def main(args=None):
    rclpy.init(args=args)
    robot_controller_node = RobotControllerNode()
            
    rclpy.spin(robot_controller_node)

    robot_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
