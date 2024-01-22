import tensorflow as tf
import numpy as np
import os

import gin
import gin.tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from replay_buffer import ReplayBuffer
from spaceship_agent import Model

def InverseDynamicsNetwork(lr, num_actions, input_dims, fc_layer_params):
  q_net = Sequential()
  q_net.add(Dense(fc_layer_params[0], input_dim=input_dims, activation='relu'))
  for i in range(1, len(fc_layer_params)):
    q_net.add(Dense(fc_layer_params[i], activation='relu'))
  q_net.add(Dense(num_actions, activation='softmax'))
  q_net.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')

  return q_net

@gin.configurable
class InverseDynamics(Model):
  def __init__(self, 
               step_var, 
               replay_buffer,
               train_interval,
               checkpoints_dir,
               max_checkpoints_to_keep,
               input_dims, 
               num_actions,
               lr, 
               batch_size,
               fc_layer_params):
    name = "inverse_dynamics"
    self.lr = lr
    self.num_action = num_actions
    self.step_var = step_var
    self.batch_size = batch_size
    self.buffer = replay_buffer
    self.train_interval = train_interval

    network = InverseDynamicsNetwork(lr, num_actions, input_dims * 2, fc_layer_params)

    self.train_loss = tf.keras.metrics.Mean('{}_train_loss'.format(name), dtype=tf.float32)
    self.action_guess_error = tf.keras.metrics.Mean(name='action_guess_error', dtype=None)
    metrics = [self.train_loss, self.action_guess_error]

    super().__init__(name, network, step_var, checkpoints_dir, max_checkpoints_to_keep, metrics)

  def infer_action(self, state, next_state):
    state_and_next_state = np.concatenate((
      np.array([state]), 
      np.array([next_state])), 
      axis=-1)
    action_probabilities = self.network(state_and_next_state)

    return action_probabilities[0]
  
  def predicted_action_error(self, state, next_state, action):
    action_probabilities = self.infer_action(state, next_state)
    error = 1 - action_probabilities[action]
    self.action_guess_error.update_state(error)
    return error.numpy()

  def action_from_probabilities(self, action_probabilities):
    return tf.math.argmax(action_probabilities).numpy()
  
  def train(self):
    step_counter = self.step_var.numpy()

    # Train only every 10 steps, and after at least a batch woth of experience is 
    # accumulated
    if self.buffer.counter < self.batch_size or step_counter % self.train_interval != 0:
      return

    # Random select an experience sample
    state_batch, new_state_batch, _, action_batch, _, _, _ = \
      self.buffer.sample_buffer(self.batch_size)

    state_and_next_state_batch = np.concatenate(
      (state_batch, new_state_batch), axis=-1)

    # Convert the ground truth actions into logits
    a_target = to_categorical(action_batch, num_classes=3)

    # Train on batch
    loss = self.network.train_on_batch(state_and_next_state_batch, a_target)
    self.train_loss(loss)