import time
from coppeliasim_zmqremoteapi_client import *

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
while (t := sim.getSimulationTime()) < 10:
    s = f'Simulation time: {t:.2f} [s]'
    print(s)

    # Control the DR20 robot by setting the target velocities of the wheel joints
    left_wheel_velocity = 2  # Replace with your desired left wheel velocity
    right_wheel_velocity = 2  # Replace with your desired right wheel velocity

    # Set the target velocities for the left and right wheel joints
    sim.setJointTargetVelocity(left_wheel_joint_handle, left_wheel_velocity)
    sim.setJointTargetVelocity(right_wheel_joint_handle, right_wheel_velocity)

    # Get the robot's position (x, y, z) and print it
    robot_position = sim.getObjectPosition(robot_handle, -1)  # -1 means relative to the world frame
    x, y, z = robot_position
    print(f'Robot position: X={x:.2f}, Y={y:.2f}, Z={z:.2f}')

    client.step()

sim.stopSimulation()
