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
               checkpoints_dir,
               max_checkpoints_to_keep,
               train_interval,
               lr, 
               discount_factor,
               target_network_soft_update_factor,
               num_actions, 
               epsilon,
               epsilon_decay,
               epsilon_final,
               batch_size, 
               input_dims, 
               fc_layer_params):
    self.lr = lr
    self.train_interval = train_interval
    self.action_space = [i for i in range(num_actions)]
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.epsilon_decay = epsilon_decay
    self.epsilon_final = epsilon_final
    self.step_var = step_var
    self.target_network_soft_update_factor = target_network_soft_update_factor
    self.buffer = replay_buffer
    self.q_net = DeepQNetwork(lr, num_actions, input_dims, fc_layer_params)
    self.q_target_net = DeepQNetwork(lr, num_actions, input_dims, fc_layer_params)

    self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

    self.name = "policy"
    self.checkpoint = tf.train.Checkpoint(global_step=self.step_var, model=self.q_net)
    self.checkopint_manager = tf.train.CheckpointManager(
      checkpoint=self.checkpoint,
      directory=os.path.join(checkpoints_dir, self.name),
      max_to_keep=max_checkpoints_to_keep)

  def policy(self, observation):
    if np.random.random() < self.epsilon:
      action = np.random.choice(self.action_space)
    else:
      state = np.array([observation])
      actions = self.q_net(state)
      action = tf.math.argmax(actions, axis=1).numpy()[0]

    return action
  
  def soft_update(self, q_net, target_net):
    tau = self.target_network_soft_update_factor
    for target_weights, q_net_weights in zip(target_net.weights, q_net.weights):
      target_weights.assign(tau * q_net_weights + (1.0 - tau) * target_weights)

  def train(self):
    step = self.step_var.numpy()

    # Train only every 10 steps, and after at least a batch woth of experience is 
    # accumulated
    if self.buffer.counter < self.batch_size or step % self.train_interval != 0:
      return
    
    # Update the target network weights with a weighted average with q_net 
    self.soft_update(self.q_net, self.q_target_net)

    # Random select an experience sample
    state_batch, goal_batch, action_batch, reward_batch, new_state_batch, done_batch = \
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
    loss = self.q_net.train_on_batch(
      state_batch, q_target, reset_metrics=True)

    self.train_loss(loss)

    # Decay the random action selection.
    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final  

  def restore(self, load_from_dir):
    if load_from_dir:
      self.load(load_from_dir)
    elif self.checkopint_manager.latest_checkpoint:
      self.restore_from_checkpoint(self.checkopint_manager.latest_checkpoint)

  def restore_from_checkpoint(self, ckpt):
    self.checkpoint.restore(ckpt).expect_partial()
    self.q_target_net.set_weights(self.q_net.get_weights())
    # Workaround: optimizer parameters are not restored correctly by the checkpoint restore.  
    # Reset the optimizer to a new optimizer so it won't attempt to load the mismatched 
    # optimizer variables. 
    self.q_net.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')

  def save_filename(self, save_dir):
    return os.path.join(save_dir, "policy.keras")

  def save(self, save_dir):
    self.q_net.save(self.save_filename(save_dir))

  def load(self, save_dir):
    self.q_net = tf.keras.models.load_model(self.save_filename(save_dir))
    self.q_target_net.set_weights(self.q_net.get_weights())

  def write_summaries(self, summary_writer, step):
    with summary_writer.as_default():
      tf.summary.scalar('policy_train_loss', self.train_loss.result(), step=step)
  
    self.train_loss.reset_states()
