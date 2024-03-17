import torch
import numpy as np
from itertools import count

from model_based.mbpo.sac.replay_memory import ReplayMemory
from model_based.mbpo.sac.sac_torch import Agent
from model_based.mbpo.model import EnsembleDynamicsModel
from model_based.mbpo.predict_env import PredictEnv
from model_based.mbpo.sample_env import EnvSampler

from envs.turtle_env.turtle_env import Env
import rclpy
from matplotlib import pyplot as plt
import pandas as pd


class Hyparams:
    def __init__(self):
        self.seed = 42 # reproducibility
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
        self.replay_size = 2000000
        self.model_retain_epochs = 1
        self.model_train_freq = 149
        self.rollout_batch_size = 100000
        self.epoch_length = 500
        self.rollout_min_epoch = 20
        self.rollout_max_epoch = 150
        self.rollout_min_length = 1
        self.rollout_max_length = 15
        self.num_epoch = 100
        self.min_pool_size = 1000
        self.real_ratio = 0.05
        self.train_every_n_steps = 1
        self.num_train_repeat = 20
        self.max_train_repeat_per_step = 5
        self.policy_train_batch_size = 256
        self.init_exploration_steps = 2000
        self.max_path_length = 250
        self.cuda = True
        self.stage = 1

def train(args, env_sampler, predict_env, agent, env_pool, model_pool):
    total_step = 0
    rollout_length = 1
    n_steps = args.init_exploration_steps

    # saves training data
    scores = []
    score_steps = []

    exploration_before_start(args, env_sampler, env_pool, agent)

    for epoch_step in range(args.num_epoch):
        print(f'Epoch {epoch_step}')
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
                break

            # maybe pause during this
            if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
                env_sampler.env.pause_simulation()
                print('... training predict model ...')
                train_predict_model(args, env_pool, predict_env)

                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length)
                env_sampler.env.unpause_simulation()

            cur_state, action, next_state, reward, done = env_sampler.sample(agent) # real step
            env_pool.push(cur_state, action, reward, next_state, done)

            if done == True:
                scores.append(reward)
                score_steps.append(n_steps)

            if len(env_pool) > args.min_pool_size:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)

            total_step += 1
            n_steps += 1


    return scores, score_steps


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
    state, action, reward, next_state, done = env_pool.sample(len(env_pool)) # samples from real env
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
        action = agent.choose_action(state)
        next_states, rewards, terminals = predict_env.step(state, action)
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
        agent.learn((batch_state, batch_action, batch_reward, batch_next_state, batch_done))

    return args.num_train_repeat


def main():
    rclpy.init() # for env ros2 node creation
    hyp = Hyparams()
    env = Env()

    # seed for reproducibility
    torch.manual_seed(hyp.seed)
    np.random.seed(hyp.seed)

    # main agent
    # agent = Agent(env.num_states, env.num_actions, hyp) # old sac
    agent = Agent(input_dims=env.num_states, max_action=env.action_upper_bound, n_actions=env.num_actions)

    # env model ensemble
    state_size = np.prod((env.num_states, ))
    action_size = np.prod((env.num_actions, ))
    env_model = EnsembleDynamicsModel(hyp.num_networks, hyp.num_elites, state_size, action_size, hyp.reward_size, hyp.pred_hidden_size,
                                          use_decay=hyp.use_decay)
    predict_env = PredictEnv(env_model)

    
    # env memory
    env_exp_replay = ReplayMemory(hyp.replay_size)

    # model memory
    rollouts_per_epoch = hyp.rollout_batch_size * hyp.epoch_length / hyp.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = hyp.model_retain_epochs * model_steps_per_epoch
    model_exp_replay = ReplayMemory(new_pool_size)

    # for real env samples
    env_sampler = EnvSampler(env, stage=hyp.stage, max_path_length=hyp.max_path_length)

    # train function
    scores, score_steps = train(hyp, env_sampler, predict_env, agent, env_exp_replay, model_exp_replay)

    # training data
    df = pd.DataFrame({'scores': scores, 'steps':score_steps})
    df.to_csv('./model_based/mbpo-sac-scores.csv')


if __name__ == '__main__':
    main()
