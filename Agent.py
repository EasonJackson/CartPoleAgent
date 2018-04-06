import numpy as np


# The agent defined to perform actions after learning
class MyAgent(object):
    RESUME = False
    RENDER = False

    def __init__(self, env, D):
        self.env = env
        self.num_neuron = 200
        self.batch_size = 10
        self.learning_rate = 1e-4
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99
        self.D = D
        self.model = {}
        self.grad_buffer = {}
        self.rmsprop_cache = {}
        self.init_model()

    def pick_next_action(self, prev_x, observation):
        """
        observation is the current screen image. reward is the current reward in the time step
        """
        # cur_x = self.prepro(observation)
        curr_x = np.mat(observation)
        x = curr_x - prev_x if prev_x is not None else np.zeros(self.D)
        aprob, h = self.policy_forward(observation)
        action = 0 if np.random.uniform() < aprob else 1
        return action, aprob, h, x, curr_x


    def init_model(self):
        self.model['W1'] = np.random.randn(self.num_neuron, self.D) / np.sqrt(self.D)
        self.model['W2'] = np.random.randn(self.num_neuron) / np.sqrt(self.num_neuron)
        self.grad_buffer = {k : np.zeros_like(v) for k, v in self.model.items()}
        self.rmsprop_cache = {k : np.zeros_like(v) for k, v in self.model.items()}

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def prepro(self, I):
        return I.astype(np.float).ravel()

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            if r[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def policy_forward(self, x):
        h = np.dot(self.model['W1'], x)
        h[h < 0] = 0
        logp = np.dot(self.model['W2'], h)
        p = self.sigmoid(logp)
        return p, h

    def policy_backward(self, epx, eph, epdlogp):
        dW2 = np.dot(eph.T, epdlogp).ravel()
        dh = np.outer(epdlogp, self.model['W2'])
        dh[eph <= 0] = 0
        dW1 = np.dot(dh.T, epx)
        res = dict()
        res['W1'] = dW1
        res['W2'] = dW2
        return res
