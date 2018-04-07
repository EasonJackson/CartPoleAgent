import numpy as np


class AgentPrototype(object):
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

    def init_model(self):
        self.model['W1'] = np.random.randn(self.num_neuron, self.D) / np.sqrt(self.D)
        self.model['W2'] = np.random.randn(self.num_neuron) / np.sqrt(self.num_neuron)
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}

    def set_num_neuron(self, num_neuron):
        self.num_neuron = num_neuron

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_learning_rate(self, learn_rate):
        self.learning_rate = learn_rate

    def set_gamma(self, gamma):
        self.gamma = gamma

    def set_decay_rate(self, decay_rate):
        self.decay_rate = decay_rate

    def reset(self):
        self.__init__(self.env, self.D)
