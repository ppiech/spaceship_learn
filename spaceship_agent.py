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

class Model:
  def __init__(self, 
               name,
               network,
               step_var, 
               checkpoints_dir,
               max_checkpoints_to_keep, 
               metrics):
    self.name = name
    self.network = network
    self.step_var = step_var
    self.metrics = metrics

    self.checkpoint = tf.train.Checkpoint(global_step=self.step_var, model=self.network)
    self.checkopint_manager = tf.train.CheckpointManager(
      checkpoint=self.checkpoint,
      directory=os.path.join(checkpoints_dir, self.name),
      max_to_keep=max_checkpoints_to_keep)

  def restore(self, load_from_dir):
    if load_from_dir:
      self.load(load_from_dir)
    elif self.checkopint_manager.latest_checkpoint:
      self.restore_from_checkpoint(self.checkopint_manager.latest_checkpoint)

  def restore_from_checkpoint(self, ckpt):
    self.checkpoint.restore(ckpt).expect_partial()

  def save_filename(self, save_dir):
    return os.path.join(save_dir, "%s.keras".format(self.name))

  def save(self, save_dir):
    self.network.save(self.save_filename(save_dir))

  def load(self, save_dir):
    self.network = tf.keras.models.load_model(self.save_filename(save_dir))

  def summaries(self):
    summarries = {}
    for metric in self.metrics:
      summarries[metric.name] = metric.result()
      metric.reset_states()
    return summarries

@gin.configurable
class Agent(Model):
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
               num_goals,
               fc_layer_params):
  
    name = "policy"

    self.buffer = replay_buffer
    self.lr = lr
    self.train_interval = train_interval
    self.action_space = [i for i in range(num_actions)]
    self.discount_factor = discount_factor
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.epsilon_decay = epsilon_decay
    self.epsilon_final = epsilon_final
    self.target_network_soft_update_factor = target_network_soft_update_factor

    model_input_size = input_dims + num_goals
    q_net = DeepQNetwork(lr, num_actions, model_input_size, fc_layer_params)
    self.q_target_net = DeepQNetwork(lr, num_actions, model_input_size, fc_layer_params)

    self.train_loss = tf.keras.metrics.Mean('{}_train_loss'.format(name), dtype=tf.float32)
    metrics = [ self.train_loss ]

    super().__init__(name, q_net, step_var, checkpoints_dir, max_checkpoints_to_keep, metrics)

  def policy(self, goal, observation):
    if np.random.random() < self.epsilon:
      action = np.random.choice(self.action_space)
    else:
      input = np.reshape(np.concatenate((goal, observation)), (1, -1))
      actions = self.network(input)
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
    self.soft_update(self.network, self.q_target_net)

    # Random select an experience sample
    state_batch, new_state_batch, goal_batch, action_batch, reward_batch, bonus_batch, done_batch = \
      self.buffer.sample_buffer(self.batch_size)

    input_batch = np.concatenate((goal_batch, state_batch), axis=-1)
    new_input_batch = np.concatenate((goal_batch, new_state_batch), axis=-1)
    # 
    q_predicted = self.network(input_batch)
    q_next = self.q_target_net(new_input_batch)
    q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()

    # Set the ground Q value to teh truth reward value plus discounted Q-value
    # determined by the target network.
    q_target = np.copy(q_predicted)
    for idx in range(done_batch.shape[0]):
      target_q_val = reward_batch[idx]
      if not done_batch[idx]:
        target_q_val += self.discount_factor*q_max_next[idx]
      target_q_val += bonus_batch[idx]
      q_target[idx, action_batch[idx]] = target_q_val
      
    # Performd gradient descent and parameter update on the q_net
    loss = self.network.train_on_batch(
      input_batch, q_target, reset_metrics=True)

    self.train_loss(loss)

    # Decay the random action selection.
    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final  

  def restore_from_checkpoint(self, ckpt):
    Model.restore_from_checkpoint(self, ckpt)
    self.q_target_net.set_weights(self.network.get_weights())
    # Workaround: optimizer parameters are not restored correctly by the checkpoint restore.  
    # Reset the optimizer to a new optimizer so it won't attempt to load the mismatched 
    # optimizer variables. 
    self.network.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')

  def load(self, save_dir):
    Model.load(self, save_dir)
    self.q_target_net.set_weights(self.network.get_weights())

