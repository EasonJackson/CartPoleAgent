import numpy as np
import pickle


class AgentPrototype(object):

    # Setting up hyper parameters of the agent
    # num_neuron: Number of neurons per layer
    # batch_size: Number of backward propagation per runs
    # learning_rate: Learning rate
    # gamma: Discount factor for reward
    # decay_rate: decay factor for RMSProp leaky sum of grad**2
    # D: Dimension of observation
    # Initialize an empty model
    # grad_buffer: Buffers that add up gradients over a batch
    # rmsprop_cache: RMSprop cache
    def __init__(self, env, D):
        self.env = env
        self.num_neuron = 10
        self.batch_size = 2
        self.learning_rate = 1e-4
        self.gamma = 0.99  # discount factor for reward
        self.decay_rate = 0.99
        self.D = D
        self.model = {}
        self.grad_buffer = {}
        self.rmsprop_cache = {}

    # Setters of optional parameters
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
