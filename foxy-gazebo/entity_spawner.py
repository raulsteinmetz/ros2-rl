import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity

class EntitySpawner(Node):
    def __init__(self):
        super().__init__('entity_spawner')

        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')

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
