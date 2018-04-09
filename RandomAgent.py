from AgentPrototype import AgentPrototype as AP
import numpy as np


class RandomAgent(AP):
    def __init__(self, env, D):
        AP.__init__(self, env, D)

    def pick_next_action(self, prev_x, observation):
        action =  self.env.action_space.sample()
        curr_x = np.zeros(self.D)
        x = curr_x - prev_x if prev_x is not None else np.zeros(self.D)
        aprob, h = np.zeros(self.D), np.zeros(self.D)
        action = 0 if np.random.uniform() < aprob else 1
        return action, aprob, h, x, curr_x