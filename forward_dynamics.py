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

def ForwardDynamicsNetwork(lr, input_dims, num_goals, fc_layer_params):

  q_net = Sequential()
  q_net.add(Dense(fc_layer_params[0], input_dim=input_dims, activation='relu'))
  for i in range(1, len(fc_layer_params)):
    q_net.add(Dense(fc_layer_params[i], activation='relu'))
  q_net.add(Dense(num_goals, activation='sigmoid'))
  q_net.compile(
    optimizer=Adam(learning_rate=lr), 
    loss="mean_squared_logarithmic_error")

  return q_net

@gin.configurable
class ForwardDynamics(Model):
  def __init__(self, 
               step_var, 
               replay_buffer,
               train_interval,
               checkpoints_dir,
               max_checkpoints_to_keep,
               input_dims, 
               num_goals,
               num_actions,
               lr, 
               batch_size,
               fc_layer_params):
    name = "forward_dynamics"

    self.lr = lr
    self.step_var = step_var
    self.batch_size = batch_size
    self.buffer = replay_buffer
    self.train_interval = train_interval
    self.num_goals = num_goals

    model_input_size = input_dims + 1
    network = ForwardDynamicsNetwork(lr, model_input_size, num_goals, fc_layer_params)

    metrics = []

    self.train_loss = tf.keras.metrics.Mean('{}_train_loss'.format(name), dtype=tf.float32)
    metrics.append(self.train_loss)

    self.goal_guess_error = tf.keras.metrics.Mean(name='goal_guess_error', dtype=None)
    metrics.append(self.goal_guess_error)

    self.goal_guess_errors = list(map(
        lambda n: tf.keras.metrics.Mean(
          'goal_{}_guess_error'.format(n), dtype=tf.float32), 
        range(num_goals)
      ))
    metrics.extend(self.goal_guess_errors)

    super().__init__(name, network, step_var, checkpoints_dir, max_checkpoints_to_keep, metrics)

  def infer_goals(self, state, action):
    state_and_action = np.concatenate((
      np.array([state]), 
      np.array([[action]])), 
      axis=-1)
    goals = self.network(state_and_action)
    return goals[0]
  
  def predicted_goals_error(self, state, action, goals):
    goal_probabilities = self.infer_goals(state, action)

    errors = tf.math.abs(goals - goal_probabilities)
    for i in range(self.num_goals):
      self.goal_guess_errors[i].update_state(errors[i])

    return errors

  def train(self):
    step_counter = self.step_var.numpy()

    # Train only every 10 steps, and after at least a batch woth of experience is 
    # accumulated
    if self.buffer.counter < self.batch_size or step_counter % self.train_interval != 0:
      return

    # Random select an experience sample
    state_batch, goals_batch, action_batch, reward_batch, new_state_batch, done_batch = \
      self.buffer.sample_buffer(self.batch_size)

    state_and_action_batch = np.concatenate(
      (state_batch, np.reshape(action_batch, (-1, 1))), axis=-1)

    # Train on batch
    loss = self.network.train_on_batch(state_and_action_batch, goals_batch)
    self.train_loss(loss)
  
