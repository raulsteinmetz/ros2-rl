# ROS2 Gazebo Simulation for Turtlebot3 Navigation with Deep Reinforcement Learning
In this project, we delve into the fascinating intersection of robotics, artificial intelligence, and simulation, focusing on the navigation of a Turtlebot3 robot in a complex Gazebo-simulated environment. Leveraging the power of ROS2 (Robot Operating System 2), we create a dynamic and responsive simulation that forms the foundation of our experiments with deep reinforcement learning (DRL).

Key Components:
- ROS2: Acts as the backbone of our project, facilitating seamless communication between the different components of the robotic system and the simulation environment.
- Gazebo Simulation: Provides a realistic and intricate 3D environment where our Turtlebot3 robot navigates. The simulation includes obstacles and walls, creating scenarios akin to real-world navigation challenges.
- Turtlebot3 Robot: A versatile and widely-used robotic platform that serves as the agent in our DRL experiments. It navigates through the Gazebo environment, making decisions based on sensor inputs and learning algorithms.
- Deep Reinforcement Learning: At the core of our project, DRL algorithms enable the Turtlebot3 to learn efficient navigation strategies through trial and error, optimizing its pathfinding abilities over time.

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