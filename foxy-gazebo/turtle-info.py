import rclpy
from nav_msgs.msg import Odometry

def odom_callback(msg):
    # Callback function to handle received Odometry messages
    position = msg.pose.pose.position
    x = position.x
    y = position.y
    z = position.z

    orientation = msg.pose.pose.orientation
    # The orientation is represented as a quaternion, you can convert it to Euler angles if needed

    print(f"Robot Position (x, y, z): ({x}, {y}, {z})")

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("odom_listener")

    # Subscribe to the /odom topic
    subscription = node.create_subscription(
        Odometry, "/odom", odom_callback, 10
    )

    print("Listening to /odom topic. Press Ctrl+C to exit.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
