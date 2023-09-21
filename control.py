import time
from coppeliasim_zmqremoteapi_client import *
import math

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

# Replace 'YourDR20Name' with the actual name of your DR20 robot in the scene
robot_name = './dr20'
robot_handle = sim.getObjectHandle(robot_name)

# Get the handles of the left and right wheel joints
left_wheel_joint_name = './dr20/leftWheelJoint_'  # Replace with the correct joint name
right_wheel_joint_name = './dr20/rightWheelJoint_'  # Replace with the correct joint name

left_wheel_joint_handle = sim.getObjectHandle(left_wheel_joint_name)
right_wheel_joint_handle = sim.getObjectHandle(right_wheel_joint_name)

sim.startSimulation()
while (t := sim.getSimulationTime()) < 3:
    s = f'Simulation time: {t:.2f} [s]'
    print(s)

    # Control the DR20 robot using the same kinematic calculations as in the child script
    v0 = 0.4
    wheel_diameter = 0.085
    inter_wheel_distance = 0.254

    # Calculate desired wheel velocities (similar to the child script)
    # You can adjust the calculation based on your desired behavior
    velocity_left = 2 * v0
    velocity_right = 2 * v0

    # Update the robot's position using kinematic calculations
    dt = sim.getSimulationTimeStep()
    p = sim.getJointPosition(left_wheel_joint_handle)
    sim.setJointPosition(left_wheel_joint_handle, p + velocity_left * dt * 2 / wheel_diameter)
    p = sim.getJointPosition(right_wheel_joint_handle)
    sim.setJointPosition(right_wheel_joint_handle, p + velocity_right * dt * 2 / wheel_diameter)
    lin_mov = dt * (velocity_left + velocity_right) / 2.0
    rot_mov = dt * math.atan((velocity_right - velocity_left) / inter_wheel_distance)
    position = sim.getObjectPosition(robot_handle, -1)  # -1 means relative to the world frame
    orientation = sim.getObjectOrientation(robot_handle, -1)
    x_dir = [math.cos(orientation[2]), math.sin(orientation[2]), 0.0]
    position[1] = position[1] + x_dir[0] * lin_mov
    position[2] = position[2] + x_dir[1] * lin_mov
    orientation[2] = orientation[2] + rot_mov
    sim.setObjectPosition(robot_handle, -1, position)
    sim.setObjectOrientation(robot_handle, -1, orientation)

    client.step()

sim.stopSimulation()
