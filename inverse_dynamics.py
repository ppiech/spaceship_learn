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
from tensorflow.keras.utils import to_categorical

from replay_buffer import ReplayBuffer

def InverseDynamicsNetwork(lr, num_actions, input_dims, fc_layer_params):
  q_net = Sequential()
  q_net.add(Dense(fc_layer_params[0], input_dim=input_dims, activation='relu'))
  for i in range(1, len(fc_layer_params)):
    q_net.add(Dense(fc_layer_params[i], activation='relu'))
  q_net.add(Dense(num_actions, activation='softmax'))
  q_net.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy')

  return q_net

@gin.configurable
class InverseDynamics:
  def __init__(self, 
               step_var, 
               replay_buffer,
               input_dims, 
               num_actions,
               lr, 
               batch_size,
               fc_layer_params):
    self.lr = lr
    self.action_space = [i for i in range(num_actions)]
    self.step_var = step_var
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
  
  def train(self):
    step_counter = self.step_var.numpy()

    # Train only every 10 steps, and after at least a batch woth of experience is 
    # accumulated
    if self.buffer.counter < self.batch_size or step_counter % 10 != 0:
      return

    # Random select an experience sample
    state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
      self.buffer.sample_buffer(self.batch_size)

    state_and_next_state_batch = np.concatenate(
      (state_batch, new_state_batch), axis=-1)

    # Convert the ground truth actions into logits
    a_target = to_categorical(action_batch, num_classes=3)

    # Train on batch
    self.net.train_on_batch(state_and_next_state_batch, a_target)
  
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
