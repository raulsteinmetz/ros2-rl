import rclpy
from sensor_msgs.msg import LaserScan

def scan_callback(msg):
    # Callback function to handle received LaserScan messages
    print("Received LaserScan data:")
    print("Header:")
    print("  Frame ID:", msg.header.frame_id)
    print("  Timestamp:", msg.header.stamp)
    print("Angle Min:", msg.angle_min)
    print("Angle Max:", msg.angle_max)
    print("Angle Increment:", msg.angle_increment)
    print("Time Increment:", msg.time_increment)
    print("Scan Time:", msg.scan_time)
    print("Range Min:", msg.range_min)
    print("Range Max:", msg.range_max)
    print("Ranges:", msg.ranges)
    print("Intensities:", msg.intensities)

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node("scan_listener")

    # Subscribe to the /scan topic
    subscription = node.create_subscription(
        LaserScan, "/scan", scan_callback, 10
    )

    print("Listening to /scan topic. Press Ctrl+C to exit.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Clean up
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
