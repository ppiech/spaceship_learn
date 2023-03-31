import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame as pg

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer


def ActionPredictorNetwork(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return q_net


class InverseDynamicsModel:
    def __init__(self, lr, discount_factor, num_actions, epsilon, replay_buffer, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.update_rate = 120
        self.step_counter = 0
        self.buffer = replay_buffer
        self.net = ActionPredictorNetwork(lr, num_actions, input_dims * 2, 256, 256)


    def action_prediction(self, prev_observation, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            states = np.array([prev_observation, observation]).flatten()
            actions = self.net(states)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        return action
    

    def train(self):
        if self.buffer.counter < self.batch_size or self.step_counter % 10 != 0:
            self.step_counter += 1
            return

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        state_transition_batch = np.concatenate((state_batch, new_state_batch), axis=1)

        self.net.train_on_batch(state_transition_batch, action_batch)
        self.step_counter += 1

    def save(self, f):
        print("f:", f)
        self.net.save(("saved_networks/space_model{0}".format(f)))
        self.net.save_weights(("saved_networks/space_model{0}/net_weights{0}.h5".format(0)))

        print("Inverse Network saved")

    def load(self, file, env):
        self.train_model(env, 5, False)
        self.net.load_weights(file)
