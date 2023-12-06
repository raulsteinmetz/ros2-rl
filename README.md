# ros2, gazebo, turtlebot3 navigation with deep rl


# How to setup moving objects in Stage 4

Edit your .world file, in your turtlebot3_simulations path, in my case it will be in ~/turtlebot_ws
```bash
~/turtlebot_ws/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage4.world 
```

Before the closing tag of the world ```**</world>**``` , copy and paste the plugin that will alow to control the objects in the sdf with Python script.
```bash
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <ros>
        <namespace>/demo</namespace>
        <argument>model_states:=model_states_demo</argument>
      </ros>
      <update_rate>1.0</update_rate>
    </plugin>

```

After that, import the required libraries and just control the objects in the script. In our code, we did something like this:
```python
        new_state = EntityState()
        new_state.name = obstacle_name
        new_state.pose.position.x = x
        new_state.pose.position.y = y
        new_state.pose.position.z = 0.0 
        new_state.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Sem rotação

        request = SetEntityState.Request()
        request.state = new_state
        future = self.set_entity_state_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None and future.result().success:
            self.get_logger().info(f"{obstacle_name} moved successfully to x: {x}, y: {y}.")
        else:
            self.get_logger().error(f"Failed to move {obstacle_name}.")
```