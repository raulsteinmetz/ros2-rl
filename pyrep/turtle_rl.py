'''

turtle rl - turtle bot deep rl env attempt 

'''

from agent import Agent
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.mobiles.turtlebot import TurtleBot
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.objects.proximity_sensor import ProximitySensor
import numpy as np
from os import system

SCENE_FILE = join(dirname(abspath(__file__)),
                  'turtle_rl.ttt')
POS_MIN, POS_MAX = [-2.0, -2.0, 0.05], [2.0, 2.0, 0.05]
EPISODES = 1000
EPISODE_LENGTH = 2500 #make it more

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


        self.infra1 = ProximitySensor(111)
        self.infra2 = ProximitySensor(112)
        self.infra3 = ProximitySensor(113)
        self.infra4 = ProximitySensor(114)
        self.infra5 = ProximitySensor(115)

    def _get_state(self, reset=False):
        distance = np.sqrt((self.agent.get_2d_pose()[0] - self.target.get_position()[0])**2
                            + (self.agent.get_2d_pose()[1] - self.target.get_position()[1])**2)
        
        robot_angle = self.agent.get_2d_pose()[2]
        target_position = np.array(self.target.get_position())[:2]
        delta_vector = target_position - self.agent.get_2d_pose()[:2]
        angle_between = np.arctan2(delta_vector[1], delta_vector[0]) - robot_angle
        angle_between = (angle_between + np.pi) % (2 * np.pi) - np.pi

        state = [distance, angle_between]
        
        if reset:
            return np.concatenate([state, [-1 ,-1, -1, -1, -1]])

        return np.concatenate([state, 
                               [self.infra1.read(),
                                self.infra2.read(),
                                self.infra3.read(), 
                                self.infra4.read(), 
                                self.infra5.read()]])
    
    def reset(self):
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_2d_pose(self.starting_pose)
        return self._get_state(reset=True)

    def step(self, action):
        self.agent.set_joint_target_velocities(action)
        self.pr.step()

        # negative reward for distance
        reward = -self._get_state()[0] * 0.01

        # env finishes when the robot is close enough to the target
        done = False
        if self._get_state()[0] < 0.5:
            reward = 200
            done = True

        

        return reward, self._get_state(), done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()




# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

def main():
    env = NavigationEnv()
    critic_lr = 0.002
    actor_lr = 0.001

    # Discount factor for future rewards
    gamma = 0.99
    # Used to update target networks
    tau = 0.005

    state_space = 7
    action_space = 2
    upper_bound = 1.0
    lower_bound = 0.0

    agent = Agent(state_space, action_space, upper_bound, lower_bound, gamma, tau, critic_lr, actor_lr, 0.2)
    
    avg_reward = 0
    rewards = []

    for e in range(EPISODES):
        episodic_reward = 0
        prev_state = env.reset()
        i = 0
        done = False
        while not done:
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
            action = agent.policy(tf_prev_state)

            action = [action[0][0], action[0][1]]

            faster = [i * 5 for i in action]

            reward, state, done = env.step(faster)

            if i > EPISODE_LENGTH:
                done = True

            agent.mem.record((prev_state, action, reward, state))

            agent.learn()
            agent.update_target()

            # print(f'Action: {action}, Reward: {reward}')

            i += 1
            episodic_reward += reward

            system('clear')
            print("Episode * {} * Avg Reward is ==> {}".format(e, avg_reward))
            print(f'State: {state}')
            print(f'Action: {action}, Reward: {reward}')


        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-5:])
        print("Episode * {} * Avg Reward is ==> {}".format(e, avg_reward))
        avg_reward_list.append(avg_reward)
        rewards.append(episodic_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.plot(np.arange(len(rewards)), rewards)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.savefig('result.png')  

if __name__ == '__main__':
    main()