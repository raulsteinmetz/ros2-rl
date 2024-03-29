import numpy as np
from envs.turtle_env.turtle_env import Env
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch as T
from matplotlib import pyplot as plt
import pandas as pd

class Trainer:
    def __init__(self, algorithm_name='undefined', stage=3):
        """
        Initialize the Trainer.

        :param algorithm_name: Name of the algorithm being used.
        :param stage: The stage of training, used to differentiate different phases or levels of training.
        """
        self.env = Env()
        self.algorithm_name = algorithm_name
        self.writer = SummaryWriter(f"runs/{algorithm_name}/{str(stage)}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}")
        self.stage = stage
        self.n_steps = 0

    def train(self, agent, episodes, max_steps, load_models=True, discrete=False):
        """
        Train the agent for a given number of episodes.

        :param agent: The agent to be trained.
        :param episodes: Total number of episodes for training.
        :param max_steps: Maximum steps in each episode.
        :param load_models: Flag to determine if existing models should be loaded.
        :param discrete: Flag to determine if the action space is discrete.
        """
        if load_models:
            agent.load_models()

        acum_rwds = []  # Accumulated rewards for each episode
        steps_rwds = []
        mov_avg_rwds = []  # Moving average rewards

        N = 100  # Window size for moving average
        best_moving_average = -np.inf

        for episode in range(episodes):
            step = 0
            done = False
            state = self.env.reset_simulation(self.stage)
            acum_reward = 0

            print('Episode: ', episode)

            while not done:
            # and step < max_steps: # TEST: DDPG doesn't necessarily needs a stop
                action = agent.choose_action(state)
                reward, done, state_ = self.env.step(action, max_steps, self.stage)
                agent.remember(state, action, reward, state_, done)
                state = state_
                acum_reward += reward
                loss = agent.learn()

                if loss is not None:
                    loss_scalar = loss.item() if isinstance(loss, T.Tensor) else loss
                    self.writer.add_scalar('Loss', loss_scalar, episode * max_steps + step)

                step += 1
                self.n_steps += 1
            
            print(f"Episode * {episode} * Accumulated Reward is ==> {acum_reward}")

            self.writer.add_scalar('Accumulated Reward in episode', acum_reward, episode)
            self.writer.add_scalar('Steps per Episode', step, episode)
            self.writer.add_scalar('self.n_steps per episode', self.n_steps, episode)
            acum_rwds.append(acum_reward)
            steps_rwds.append(self.n_steps)

            # Log metrics to tensorboard
            if loss is not None:
                self.writer.add_scalar('Loss', loss, episode)

            # Compute and record moving average
            if episode >= N - 1:
                moving_avg = np.mean(acum_rwds[-N:])
                mov_avg_rwds.append(moving_avg)

                if moving_avg > best_moving_average:
                    best_moving_average = moving_avg
                    agent.save_models()
                    print(f"Saving best models with moving average reward {best_moving_average}...")

            else:
                mov_avg_rwds.append(np.mean(acum_rwds[:episode + 1]))

            if episode == 1 or episode % 50 == 0:
                self.writer.add_scalar('Accumulated Reward each 50 episodes', acum_reward, episode)
                self.writer.add_scalar('Moving Average Reward each 50 episodes', mov_avg_rwds[-1], episode)
                plt.plot(acum_rwds, alpha=0.5, label="Raw Reward" if episode == 0 else "")
                plt.plot(mov_avg_rwds, color='red', label="Moving Avg Reward" if episode == 0 else "")
                plt.xlabel("Episode")
                plt.ylabel("Accumulated Reward")
                plt.legend()
                plt.title(f'{self.algorithm_name} - Stage {self.stage}')
                plt.savefig(f'model_free/{self.algorithm_name}/acum_rwds_{self.stage}.png')

            if episode == 1 or episode % 100 == 0:
                df = pd.DataFrame({'scores': acum_rwds, 'steps':steps_rwds})
                df.index.name = 'episode'
                df.to_csv(f'model_free/{self.algorithm_name}/scores.csv')

        return acum_rwds, steps_rwds

    def kill_env(self):
        """
        Terminate the environment and close the writer.
        """
        self.env.destroy_node()
        self.writer.close()
