import rclpy
from rclpy.node import Node
from std_srvs.srv import Empty

class SimulationManager(Node):
    def __init__(self):
        super().__init__('simulation_manager')

        self.pause_simulation_client = self.create_client(Empty, '/pause_physics')
        self.unpause_simulation_client = self.create_client(Empty, '/unpause_physics')

    def call_service_sync(self, client, request):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def pause_simulation(self):
        self.call_service_sync(self.pause_simulation_client, Empty.Request())

    def unpause_simulation(self):
        self.call_service_sync(self.unpause_simulation_client, Empty.Request())
