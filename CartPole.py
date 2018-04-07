import gym
import numpy as np
from Agent import MyAgent

RENDER = True
GAME = 'CartPole-v0'
# GAME = "MountainCar-v0"
NUM_EPISODE = 50


def play_episode(env, agent):
    prev_x, xs, hs, dlogps, drs, running_reward, reward_sum, episode_reward, observation = initilizeState()

    for curr_eps in range(NUM_EPISODE):
        action, aprob, h, x, prev_x = agent.pick_next_action(prev_x, observation)
        record_intermediates(xs, x, hs, h, dlogps, aprob, action)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        drs.append(reward)

        while not done:
            if RENDER:
                env.render()  # render the game

            action, aprob, h, x, curr_x = agent.pick_next_action(prev_x, observation)
            record_intermediates(xs, x, hs, h, dlogps, aprob, action)
            observation, reward, done, info = env.step(action)
            reward_sum += reward  # adding up the reward in the episode
            drs.append(reward)

            if done:  # the episode is done
                print("Episode number:{}".format(curr_eps))
                wrap_up_episode(xs, hs, dlogps, drs, curr_eps)
                save_state(reward_sum, running_reward)
                reward_sum = 0
                prev_x = 0
                observation = env.reset()

def initilizeState():
    prev_x = None
    xs, hs, dlogps, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_reward = 0
    observation = env.reset()
    return prev_x, xs, hs, dlogps, drs, running_reward, reward_sum, episode_reward, observation


def record_intermediates(xs, x, hs, h, dlogps, aprob, action):
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    dlogps.append(y - aprob)


def wrap_up_episode(xs, hs, dlogps, drs, curr_eps):
    epx = np.vstack(xs)
    eph = np.vstack(hs)
    epdlogp = np.vstack(dlogps)
    epr = np.vstack(drs)

    discounted_epr = agent.discount_rewards(epr)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    epdlogp *= discounted_epr
    grad = agent.policy_backward(epx, eph, epdlogp)
    for k in agent.model: agent.grad_buffer[k] += grad[k]

    if curr_eps % agent.batch_size == 0:
        for k, v in agent.model.items():
            g = agent.grad_buffer[k]
            agent.rmsprop_cache[k] = agent.decay_rate * agent.rmsprop_cache[k] + (1 - agent.decay_rate) * g**2
            agent.model[k] += agent.learning_rate * g / (np.sqrt(agent.rmsprop_cache[k]) + 1e-5)
            agent.grad_buffer[k] = np.zeros_like(v)


def save_state(reward_sum, running_reward):
    running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
    # print
    # 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
    #if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))


if __name__ == "__main__":
    env = gym.make(GAME)
    env.reset()
    observation, _, _, _ = env.step(env.action_space.sample())
    agent = MyAgent(env, len(observation))
    play_episode(env, agent)

