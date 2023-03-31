from spaceship_agent import Agent
import gym
import tensorflow as tf
from spaceship_env import Env
from replay_buffer import ReplayBuffer

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

env = Env()
train = 0
test = 1
num_episodes = 1000
graph = True
input_dims = 7

file = 'saved_networks/dqn_model0'

# EPSILON LOWERED FOR NOW

buffer = ReplayBuffer(5000, input_dims)
dqn_agent = Agent(lr=0.00075, 
                  discount_factor=0.99, 
                  num_actions=3, 
                  epsilon=0.03, 
                  replay_buffer=buffer, 
                  batch_size=50,
                  input_dims=input_dims)

if train and not test:
    dqn_agent.train_model(env, num_episodes, graph, None)
else:
    dqn_agent.test(env, num_episodes, file, graph)
