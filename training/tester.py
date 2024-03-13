import numpy as np
from envs.turtle_env.turtle_env import Env
from datetime import datetime
import torch as T
from matplotlib import pyplot as plt

class Tester:
    def __init__(self, algorithm_name='undefined', stage=3):
        """
        Initialize the Tester.

        :param algorithm_name: Name of the algorithm being used.
        :param stage: The stage of testing, used to differentiate different phases or levels of testing.
        """
        self.env = Env()
        self.algorithm_name = algorithm_name
        self.stage = stage

    def test(self, agent, episodes, max_steps, load_models=True, discrete=False, warmup=30):
        """
        Test the agent for a given number of episodes.

        :param agent: The agent to be trained.
        :param episodes: Total number of episodes for training.
        :param max_steps: Maximum steps in each episode.
        :param load_models: Flag to determine if existing models should be loaded.
        :param discrete: Flag to determine if the action space is discrete.
        :warmup_episodes: Number of episodes used for warmup
        """

        agent.load_models()

        acum_rwds = []  # Accumulated rewards for each episode

        for episode in range(episodes + warmup):
            step = 0
            done = False
            state = self.env.reset_simulation(self.stage)
            acum_reward = 0

            print('Episode: ', episode)

            while not done:
                action = agent.choose_action(state)
                reward, done, state_ = self.env.step(action, step, max_steps, discrete, self.stage)
                state = state_
                acum_reward += reward
                step += 1



            if episode > warmup:
                print(f"Episode * {episode-warmup} * Accumulated Reward is ==> {acum_reward}")
                acum_rwds.append(acum_reward)



        return acum_rwds

    def kill_env(self):
        """
        Terminate the environment and close the writer.
        """
        self.env.destroy_node()