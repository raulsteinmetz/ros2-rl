import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty  # Service for pausing and unpausing the simulation
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SpawnEntity
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

        # Internal state
        self.odom_data = None
        self.scan_data = None

        # Spawn a target in the environment
        self.spawn_target_in_environment()

        # Start the main RL control loop
        self.rl_control_loop()

    def odom_callback(self, msg):
        self.odom_data = msg

    def scan_callback(self, msg):
        self.scan_data = msg

    def rl_control_loop(self):
        # Main loop for reinforcement learning control
        while rclpy.ok():
            # Pause the simulation
            self.call_service_sync(self.pause_simulation_client, Empty.Request())

            if self.scan_data is None or self.odom_data is None:
                # If there's no data, there's nothing to do yet, just continue
                self.call_service_sync(self.unpause_simulation_client, Empty.Request())
                continue

            sleep(0.001)

            print("spinned once")

            linear_vel = 0.2  # These are placeholders and should be decided by your RL algorithm
            angular_vel = 0.1

            cmd_vel_msg = Twist()
            cmd_vel_msg.linear.x = linear_vel
            cmd_vel_msg.angular.z = angular_vel

            # Publish the control command
            self.cmd_vel_publisher.publish(cmd_vel_msg)

            # Unpause the simulation for the next step to take effect
            self.call_service_sync(self.unpause_simulation_client, Empty.Request())

            # Here you may want to introduce a delay based on your control frequency
            rclpy.spin_once(self, timeout_sec=0.1)  # Adjust the timing as necessary

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
        
        # Request for spawning a sphere
        request = SpawnEntity.Request()
        sphere_sdf = """
        <?xml version='1.0'?>
        <sdf version='1.6'>
          <model name='target_model'>
            <pose>1 0 0 0 0 0</pose>
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
        request.name = 'target_sphere'  # Unique name for the new model
        request.xml = sphere_sdf  # Model XML

        # Call the service
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)  # Wait for the response

        if future.result() is not None:
            self.get_logger().info("Entity spawned successfully.")
        else:
            self.get_logger().error("Failed to spawn entity.")


def main(args=None):
    rclpy.init(args=args)
    robot_controller_node = RobotControllerNode()
    rclpy.spin(robot_controller_node)

    robot_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
