from spaceship_agent import Agent
import gym
import tensorflow as tf
from spaceship_env import Env

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)

env = Env()
# env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
train = 0
test = 1
num_episodes = 100000
graph = True

file_type = 'tf'
file = 'saved_networks/space_model_best'

# EPSILON LOWERED FOR NOW

dqn_agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=3, epsilon=0.03, batch_size=5000, input_dims=7)

if train and not test:
    dqn_agent.train_model(env, num_episodes, graph, file, file_type)
else:
    dqn_agent.test(env, num_episodes, file_type, file, graph)
