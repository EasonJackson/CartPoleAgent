from AgentPrototype import AgentPrototype as AP


class RandomAgent(AP):
    def __init__(self, env, D):
        AP.__init__(env, D)

    def pick_next_action(self, prev_x, observation):
        return self.env.action_space.sample()