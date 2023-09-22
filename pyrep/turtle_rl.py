'''

turtle rl - turtle bot deep rl env attempt 

'''

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)),
                  'turtle_rl.ttt')
POS_MIN, POS_MAX = [0.8, -0.2, 1.0], [1.0, 0.2, 1.4]
EPISODES = 5
EPISODE_LENGTH = 200

class NavigationEnv(object):
    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        self.agent = TurtleBot()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                      size=[0.05, 0.05, 0.05],
                      color=[1.0, 0.1, 0.1],
                      static=True, respondable=False)
        self.starting_pose = self.agent.get_2d_pose()

    def _get_state(self):
        # falta ler os sensores pelo menos
        return np.concatenate([self.agent.get_2d_pose(),
                               self.target.get_position()])
    
    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_2d_pose(self.starting_pose)
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on the wheels?
        self.pr.step()  # Step the physics simulation
        robot_x, robot_y, yaw = self.agent.get_2d_pose()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        reward = -np.sqrt((robot_x - tx) ** 2 + (robot_y - ty) ** 2)
        return reward, self._get_state()

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state):
        del state
        return [0.5, 1.0]

    def learn(self, replay_buffer):
        pass



env = NavigationEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        reward, next_state = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
