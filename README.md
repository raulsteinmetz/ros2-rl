# ROS2 Gazebo Simulation for Turtlebot3 Navigation with Deep Reinforcement Learning
In this project, we delve into the fascinating intersection of robotics, artificial intelligence, and simulation, focusing on the navigation of a Turtlebot3 robot in a complex Gazebo-simulated environment. Leveraging the power of ROS2 (Robot Operating System 2), we create a dynamic and responsive simulation that forms the foundation of our experiments with deep reinforcement learning (DRL).

Key Components:
- ROS2: Acts as the backbone of our project, facilitating seamless communication between the different components of the robotic system and the simulation environment.
- Gazebo Simulation: Provides a realistic and intricate 3D environment where our Turtlebot3 robot navigates. The simulation includes obstacles and walls, creating scenarios akin to real-world navigation challenges.
- Turtlebot3 Robot: A versatile and widely-used robotic platform that serves as the agent in our DRL experiments. It navigates through the Gazebo environment, making decisions based on sensor inputs and learning algorithms.
- Deep Reinforcement Learning: At the core of our project, DRL algorithms enable the Turtlebot3 to learn efficient navigation strategies through trial and error, optimizing its pathfinding abilities over time.


To modify DRL stages, you can find the world launch files at: `~/ros_humble/src/turtlebot3_simulations/turtlebot3_gazebo/world`

They will normally call an obstacle sdf, located at `~/ros_humble/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_dqn_world/`, just create one folder with a new sdf and call it!

## Modifying DRL Stages

To modify the Deep Reinforcement Learning (DRL) stages, according to the ones in this repository, follow the steps below:

1. **Copy the contents of the `launch`, `models`, and `worlds` directories to your desired locations using the following commands:**

   - For `launch`:
     ```bash
     cp -rf turtlebot3_gazebo/launch ~/ros_humble/src/turtlebot3_simulations/turtlebot3_gazebo/launch
     ```
   
   - For `models`:
     ```bash
     cp -rf turtlebot3_gazebo/models ~/ros_humble/src/turtlebot3_simulations/turtlebot3_gazebo/models
     ```
   
   - For `worlds`:
     ```bash
     cp -rf turtlebot3_gazebo/worlds ~/ros_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds
     ```

2. **Build your project with `colcon build --symlink-install` to apply the changes:**
   ```bash
   cd ~/ros_humble/
   colcon build --symlink-install
    ```


## Starting TensorBoard to monitor the loss, rewards and everything logged about each algorithm.
To start TensorBoard and point it to the `logdir` log directory, follow these steps:

1. Open a terminal window.
2. Activate your Python virtual environment or Conda environment, if you're using one.
3. Run the following command:

```bash
tensorboard --logdir=runs
```

This command will start TensorBoard and make it use the logs stored in the log directory. After executing the command, TensorBoard will provide a URL, typically http://localhost:6006, which you can open in your web browser to view your TensorBoard dashboard.
