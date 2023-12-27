import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame as pg
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer


def DeepQNetwork(lr, num_actions, input_dims, fc1, fc2):
    q_net = Sequential()
    q_net.add(Dense(fc1, input_dim=input_dims, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(fc2, activation='relu'))
    q_net.add(Dense(num_actions, activation=None))
    q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return q_net


class Agent:
    def __init__(self, lr, discount_factor, num_actions, epsilon, batch_size, input_dims):
        self.action_space = [i for i in range(num_actions)]
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.0001
        self.epsilon_final = 0.03
        self.update_rate = 120
        self.step_counter = 0
        self.tau = 0.001
        self.buffer = ReplayBuffer(batch_size * 10, input_dims)
        self.q_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)
        self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, 256, 256)

    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def policy(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_net(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action
    
    def soft_update(self, q_net, target_net):
        for target_weights, q_net_weights in zip(target_net.weights, q_net.weights):
            target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

    def train(self):
        if self.buffer.counter < self.batch_size or self.step_counter % 10 != 0:
            self.step_counter += 1
            return
        # if self.step_counter % self.update_rate == 0:
            # self.q_target_net.set_weights(self.q_net.get_weights())
        self.soft_update(self.q_net, self.q_target_net)
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted = self.q_net(state_batch)
        q_next = self.q_target_net(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.discount_factor*q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val
            
        self.q_net.train_on_batch(state_batch, q_target)
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.step_counter += 1

    def save_file(self, saved_model_dir, step):
        return os.path.join(saved_model_dir, "{0}.keras".format(step))

    def save(self, saved_model_dir, step):
        self.q_net.save(
            self.save_file(saved_model_dir, step))

    def load(self, saved_model_dir, step):
        self.q_net = tf.keras.models.load_model(
            self.save_file(saved_model_dir, step))

    def test(self, env, num_episodes, file, graph):
        clock = pg.time.Clock()
        self.q_net = tf.keras.models.load_model(file)
        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0
        for i in range(num_episodes):
            state, _ = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                env.render()
                clock.tick(300)
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.quit()
                action = self.policy(state)
                new_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated
                episode_score += reward
                state = new_state
            score += episode_score
            scores.append(episode_score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('LunarLander_Test.png')

        env.close()
