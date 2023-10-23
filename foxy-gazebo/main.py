import rclpy
from robot_controller import RobotController
from simulation_manager import SimulationManager
from entity_spawner import EntitySpawner
from time import sleep

def rl_control_loop(robot_controller, simulation_manager):
    while rclpy.ok():
        print('salve')
        # Pause the simulation
        simulation_manager.pause_simulation()

        if robot_controller.scan_data is None or robot_controller.odom_data is None:
            # If there's no data, there's nothing to do yet, just continue
            simulation_manager.unpause_simulation()
            continue

        sleep(0.001)  # This delay might need to be adjusted based on your RL algorithm's need
        print('hi!')

        # Here, you should implement your reinforcement learning algorithm logic
        # which decides the linear and angular velocity based on the current state.
        # For demonstration, we are using constant values.

        linear_vel = 0.2  # This should be decided by your RL algorithm
        angular_vel = 0.1  # This should be decided by your RL algorithm

        robot_controller.control_and_publish(linear_vel, angular_vel)

        # Unpause the simulation for the next step to take effect
        simulation_manager.unpause_simulation()

        # Here you may want to introduce a delay based on your control frequency
        rclpy.spin_once(robot_controller, timeout_sec=0.1)  # Adjust the timing as necessary
    print('not ok')

def main(args=None):
    try:
        rclpy.init(args=args)

        robot_controller = RobotController()
        simulation_manager = SimulationManager()
        entity_spawner = EntitySpawner()

        entity_spawner.spawn_target_in_environment()

        print("Starting RL control loop...")  # Debugging print statement
        rl_control_loop(robot_controller, simulation_manager)

    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        robot_controller.destroy_node()
        simulation_manager.destroy_node()
        entity_spawner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
