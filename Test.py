import gym

num_episodes = 100  # how many episodes to play
render = True  # whether to render the game. You should turn this off to speed up the program.


class myAgent(object):
    def __init__(self, env):
        self.env = env
        pass

    def pick_next_action(self, observation, reward):
        """ observation is the current screen image. reward is the current reward in the time step """
        best_action = 0  # RANDOMLY pick an action for the next move.
        return best_action


env = gym.make('CartPole-v0')  # create the game envirement
agent = myAgent(env)  # create an agent


if __name__ == "__main__":
    # let's play some episodes of the game
    for _ in range(num_episodes):
        observation = env.reset()  # initialize the game
        episode_reward = 0  # the sum of rewards in an episode
        action = env.action_space.sample()  # RANDOMLY pick an action for the next move
        observation, reward, done, info = env.step(action)  # execute the action and get the reward and next observation
        while not done:
            if render:
                env.render()  # render the game
            action = agent.pick_next_action(observation, reward)
            observation, reward, done, info = env.step(action)
            episode_reward += reward  # adding up the reward in the episode
            if done:  # the episode is done
                print("Episode reward:{}".format(episode_reward))
                episode_reward = 0