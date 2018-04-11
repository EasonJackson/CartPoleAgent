# CartPoleAgent
A simple AI agent controls a cartpole from falling

## Prerequisite

[OpenAI gym](https://github.com/openai/gym)

```pip install gym```

If running a atari game, a dependency on gym[atari] needs to be installed.

[Atari](https://github.com/openai/gym#atari)

```pip install gym[atari]```

## Example

### Running a CartPole with an agent

- Game parameters

```
# A render flag: true for displaying updated game after every action; turn false to accelerate training procedure
RENDER = False

# Set a game env
# For more game env can be found on [OpenAI](https://github.com/openai/gym)
GAME = 'CartPole-v0'

# Total run of episode
NUM_EPISODE = 10

# Resume an agent with previous model params
# On first training process turn it as false
# Resume loads a model within agent from cartpole.p file with pickle
resume = False
```

- Agent parameters

A prototype agent

```
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
```

All setters are provided to tune the params for customized purposes.

A cartpole agent with inheritance from the prototype

```
# Initialize model neurons with random coefs
self.model['W1'] = np.random.randn(self.num_neuron, self.D) / np.sqrt(self.D)
self.model['W2'] = np.random.randn(self.num_neuron) / np.sqrt(self.num_neuron)
```

```
# Initialize intermediates data buffers
self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
self.rmsprop_cache = {k: np.zeros_like(v) for k, v in self.model.items()}
```

```
# Flatten observations from ndarray(D * D) into ndarray(1 * D^2)
def prepro(self, I):
    return I.astype(np.float).ravel()
```

### Neuron network

A two layer neuron network is used for training the agent.

### Policy gradient

- Policy forward

- Policy backward


## Links

This agent is originally from [Andrej Karpathy blog](http://karpathy.github.io/2016/05/31/rl/)

A gist on the agent: https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5

## Author
**Eason - @2018**