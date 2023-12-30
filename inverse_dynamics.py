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

def InverseDynamicsNetwork(lr, num_actions, input_dims, fc_layer_params):
  q_net = Sequential()
  q_net.add(Dense(fc_layer_params[0], input_dim=input_dims, activation='relu'))
  for i in range(1, len(fc_layer_params)):
    q_net.add(Dense(fc_layer_params[i], activation='relu'))
  q_net.add(Dense(num_actions, activation=None))
  q_net.compile(optimizer=Adam(learning_rate=lr), loss='mse')

  return q_net

@gin.configurable
class InverseDynamics:
  def __init__(self, 
               replay_buffer,
               input_dims, 
               num_actions,
               lr, 
               batch_size,
               fc_layer_params):
    self.lr = lr
    self.action_space = [i for i in range(num_actions)]
    self.batch_size = batch_size
    self.buffer = replay_buffer
    self.net = InverseDynamicsNetwork(lr, num_actions, input_dims * 2, fc_layer_params)
    self.checkpoint = tf.train.Checkpoint(model=self.net)

  def infer_action(self, state, next_state):
    state_and_next_state = np.concatenate((
      np.array([state]), 
      np.array([next_state])), 
      axis=-1)
    actions = self.net(state_and_next_state)
    action = tf.math.argmax(actions, axis=1).numpy()[0]

    return action
  
  def soft_update(self, q_net, target_net):
    for target_weights, q_net_weights in zip(target_net.weights, q_net.weights):
      target_weights.assign(self.tau * q_net_weights + (1.0 - self.tau) * target_weights)

  def train(self):
    step_counter = self.step_var.numpy()
    if self.buffer.counter < self.batch_size or step_counter % 10 != 0:
      return
    # if step_counter % self.update_rate == 0:
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
