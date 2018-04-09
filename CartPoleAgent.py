import numpy as np
import pickle
from AgentPrototype import AgentPrototype as AP


# Agents that inherit from AgentPrototype
class CartPoleAgent(AP):
    def __init__(self, env, D):
        AP.__init__(self, env, D)

    def init_model(self, resume_flag):
        '''
        Initialize the model of this agent.
        Model by default have two layers of neuron networks.
        :param resume_flag: A boolean flag,
            true indicating loading a model from exist;
            false indicating create a new model.
        '''
        if resume_flag:
            self.model = pickle.load(open("cartpole.p", "rb"))
        else:
            self.model['W1'] = np.random.randn(self.num_neuron, self.D) / np.sqrt(self.D)
            self.model['W2'] = np.random.randn(self.num_neuron) / np.sqrt(self.num_neuron)

        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}

    def pick_next_action(self, prev_x, observation):
        """
        observation is the current screen image.
        """
        # cur_x = self.prepro(observation)
        curr_x = np.mat(observation)
        x = curr_x - prev_x if prev_x is not None else np.zeros(self.D)
        aprob, h = self.policy_forward(observation)
        action = 0 if np.random.uniform() < aprob else 1
        return action, aprob, h, x, curr_x

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def prepro(self, I):
        return I.astype(np.float).ravel()

    def discount_rewards(self, r):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            # if r[t] != 0:
            #     running_add = 0
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
