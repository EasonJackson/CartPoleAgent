import gym


class myAgent(object):
    def __init__(self, env):
        self.env = env
        pass

    def pick_next_action(self, observation, reward):

        """
        observation is the current screen image. reward is the current reward in the time step
        """

        # best_action = 0
        best_action = env.action_space.sample()  # RANDOMLY pick an action for the next move.
        return best_action


num_episode = 1
render = True
env = gym.make('MsPacman-v0')  # Create a gym instance
env.reset()
agent = myAgent(env)

for _ in range(num_episode):
    observation = env.render()
    episode_reward = 0
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    while not done:
        if render:
            env.render()  # render the game

        action = agent.pick_next_action(observation, reward)
        observation, reward, done, info = env.step(action)
        episode_reward += reward  # adding up the reward in the episode

        if done:  # the episode is done
            print("Episode reward:{}".format(episode_reward))
            episode_reward = 0