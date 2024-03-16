import argparse
import time
import torch
import numpy as np
from itertools import count

import logging

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler

from envs.turtle_env.turtle_env import Env
import rclpy
from matplotlib import pyplot as plt



class Hyparams:
    def __init__(self): # this should be further explored and optimized for turtle task
        self.seed = 42
        self.use_decay = True
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.policy = "Gaussian"
        self.target_update_interval = 1
        self.automatic_entropy_tuning = False
        self.hidden_size = 256
        self.lr = 0.0003
        self.num_networks = 7
        self.num_elites = 5
        self.pred_hidden_size = 200
        self.reward_size = 1
        self.replay_size = 2000000 # was 1 million
        self.model_retain_epochs = 1
        self.model_train_freq = 75 # was 250
        self.rollout_batch_size = 100000
        self.epoch_length = 300 # was 1000
        self.rollout_min_epoch = 20
        self.rollout_max_epoch = 150
        self.rollout_min_length = 1
        self.rollout_max_length = 15
        self.num_epoch = 1000
        self.min_pool_size = 1000
        self.real_ratio = 0.05
        self.train_every_n_steps = 1
        self.num_train_repeat = 20
        self.max_train_repeat_per_step = 5
        self.policy_train_batch_size = 256
        self.init_exploration_steps = 2000 # was 5000
        self.max_path_length = 250 # was 1000
        self.cuda = True

def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0
    reward_sum = 0
    rollout_length = 1
    scores = []
    score_steps = []
    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            # maybe pause during this
            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                print('... training predict model ...')
                train_predict_model(args, env_pool, predict_env)

                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)

            cur_state, action, next_state, reward, done = env_sampler.sample(agent) # real step
            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)

            total_step += 1


            # testing
            # if total_step % args.epoch_length == 0:
            #     print('... testing ...')
            #     '''
            #     avg_reward_len = min(len(env_sampler.path_rewards), 5)
            #     avg_reward = sum(env_sampler.path_rewards[-avg_reward_len:]) / avg_reward_len
            #     logging.info("Step Reward: " + str(total_step) + " " + str(env_sampler.path_rewards[-1]) + " " + str(avg_reward))
            #     print(total_step, env_sampler.path_rewards[-1], avg_reward)
            #     '''
            #     env_sampler.current_state = None
            #     sum_reward = 0
            #     done = False
            #     test_step = 0

            #     while (not done) and (test_step != args.max_path_length):
            #         cur_state, action, next_state, reward, done = env_sampler.sample(agent, eval_t=True)
            #         sum_reward += reward
            #         test_step += 1
            #     print("Score: " + str(sum_reward))


            #     # this is very raw yet, need to do it better
            #     scores.append(sum_reward)
            #     score_steps.append(total_step)
            #     plt.plot(score_steps, scores)
            #     plt.savefig('./plot.png')


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args.init_exploration_steps):
        cur_state, action, next_state, reward, done = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args.rollout_min_length + (epoch_step - args.rollout_min_epoch)
                              / (args.rollout_max_epoch - args.rollout_min_epoch) * (args.rollout_max_length - args.rollout_min_length),
                              args.rollout_min_length), args.rollout_max_length))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        state = next_states[nonterm_mask]


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    if total_step % args.train_every_n_steps > 0:
        return 0

    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)

    return args.num_train_repeat


def main():

    hyp = Hyparams()
    rclpy.init()
    # turtlebot env
    env = Env()

    # Set random seed
    torch.manual_seed(hyp.seed)
    np.random.seed(hyp.seed)

    # Intial agent
    agent = SAC(env.num_states, env.num_actions, hyp)

    # Initial ensemble model
    state_size = np.prod((env.num_states, )) # was env.observation_space.shape
    action_size = np.prod((env.num_actions, )) # was env.action_space.shape
    env_model = EnsembleDynamicsModel(hyp.num_networks, hyp.num_elites, state_size, action_size, hyp.reward_size, hyp.pred_hidden_size,
                                          use_decay=hyp.use_decay)

    # Predict environments
    predict_env = PredictEnv(env_model)

    # Initial pool for env
    env_pool = ReplayMemory(hyp.replay_size)
    # Initial pool for model
    rollouts_per_epoch = hyp.rollout_batch_size * hyp.epoch_length / hyp.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = hyp.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=hyp.max_path_length, stage=2)

    train(hyp, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == '__main__':
    main()
