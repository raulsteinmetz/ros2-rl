import rclpy
from geometry_msgs.msg import Twist

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("cmd_vel_publisher")

    # Create a publisher for the /cmd_vel topic
    publisher = node.create_publisher(Twist, "/cmd_vel", 10)

    # Create a Twist message with linear and angular velocities
    twist_msg = Twist()
    twist_msg.linear.x = 0.2  # Linear velocity in the x-axis (forward)
    twist_msg.angular.z = 0.1  # Angular velocity in the z-axis (rotation)

    print("Publishing to /cmd_vel topic. Press Ctrl+C to exit.")

    try:
        while rclpy.ok():
            publisher.publish(twist_msg)
            node.get_logger().info("Published Twist message")
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        pass

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
