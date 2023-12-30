import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame as pg
import os

import gin
import gin.tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from replay_buffer import ReplayBuffer

def DeepQNetwork(lr, num_actions, input_dims, fc_layer_params):
  q_net = Sequential()
  q_net.add(Dense(fc_layer_params[0], input_dim=input_dims, activation='relu'))
  for i in range(1, len(fc_layer_params)):
    q_net.add(Dense(fc_layer_params[i], activation='relu'))
  q_net.add(Dense(num_actions, activation=None))
  q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

  return q_net

@gin.configurable
class Agent:
  def __init__(self, 
               step_var, 
               replay_buffer,
               lr, 
               discount_factor, 
               num_actions, 
               epsilon, 
               batch_size, 
               input_dims, 
               fc_layer_params):
    self.lr = lr
    self.action_space = [i for i in range(num_actions)]
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.epsilon_decay = 0.0001
    self.epsilon_final = 0.03
    self.step_var = step_var
    self.tau = 0.001
    self.buffer = replay_buffer
    self.q_net = DeepQNetwork(lr, num_actions, input_dims, fc_layer_params)
    self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, fc_layer_params)
    self.policy_checkpoint = tf.train.Checkpoint(global_step=self.step_var, model=self.q_net)

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
    step_counter = self.step_var.numpy()

    # Train only every 10 steps, and after at least a batch woth of experience is 
    # accumulated
    if self.buffer.counter < self.batch_size or step_counter % 10 != 0:
      return
    
    # Update the target network weights with a weighted average with q_net 
    self.soft_update(self.q_net, self.q_target_net)

    # Random select an experience sample
    state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
      self.buffer.sample_buffer(self.batch_size)

    # 
    q_predicted = self.q_net(state_batch)
    q_next = self.q_target_net(new_state_batch)
    q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()

    # Set the ground Q value to teh truth reward value plus discounted Q-value
    # determined by the target network.
    q_target = np.copy(q_predicted)
    for idx in range(done_batch.shape[0]):
      target_q_val = reward_batch[idx]
      if not done_batch[idx]:
        target_q_val += self.discount_factor*q_max_next[idx]
      q_target[idx, action_batch[idx]] = target_q_val
      
    # Performd gradient descent and parameter update on the q_net
    self.q_net.train_on_batch(state_batch, q_target)

    # Decay the random action selection.
    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
  
  def restore_from_checkpoint(self, ckpt):
    self.policy_checkpoint.restore(ckpt).expect_partial()
    self.q_target_net.set_weights(self.q_net.get_weights())
    # Workaround: optimizer parameters are not restored correctly by the checkpoint restore.  
    # Reset the optimizer to a new optimizer so it won't attempt to load the mismatched 
    # optimizer variables. 
    self.q_net.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')

  def save(self, saved_model_dir, step):
    filename = os.path.join(saved_model_dir, "{0}.keras".format(step))
    self.q_net.save(filename)

  def load(self, filename):
    self.q_net = tf.keras.models.load_model(filename)
    self.q_target_net.set_weights(self.q_net.get_weights())
