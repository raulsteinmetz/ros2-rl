class EnvSampler():
    def __init__(self, env, stage):
        self.env = env

        self.path_length = 0
        self.current_state = None
        self.path_rewards = []
        self.sum_reward = 0
        self.stage = stage

    def sample(self, agent, max_steps, eval_t=False):
        if self.current_state is None:
            self.current_state = self.env.reset_simulation(self.stage)

        cur_state = self.current_state
        action = agent.select_action(self.current_state, eval=eval_t) # choose_action for this repo's sac
        reward, terminal, next_state = self.env.step(action, 250)
        self.path_length += 1
        self.sum_reward += reward

        if terminal:
            self.current_state = None
            self.path_length = 0
            self.path_rewards.append(self.sum_reward)
            self.sum_reward = 0
        else:
            self.current_state = next_state

        return cur_state, action, next_state, reward, terminal
