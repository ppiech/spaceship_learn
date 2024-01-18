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
class ForwardDynamics:
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
    self.name = "forward_dynamics"
    self.lr = lr
    self.step_var = step_var
    self.batch_size = batch_size
    self.buffer = replay_buffer
    self.train_interval = train_interval

    model_input_size = input_dims + 1
    self.net = ForwardDynamicsNetwork(lr, model_input_size, num_goals, fc_layer_params)

    self.train_loss = tf.keras.metrics.Mean('{}_train_loss'.format(self.name), dtype=tf.float32)

    self.checkpoint = tf.train.Checkpoint(model=self.net)
    self.checkopint_manager = tf.train.CheckpointManager(
      checkpoint=self.checkpoint,
      directory=os.path.join(checkpoints_dir, self.name),
      max_to_keep=max_checkpoints_to_keep)

  def infer_goals(self, state, action):
    state_and_action = np.concatenate((
      np.array([state]), 
      np.array([[action]])), 
      axis=-1)
    goals = self.net(state_and_action)
    return goals[0]
  
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
    loss = self.net.train_on_batch(state_and_action_batch, goals_batch)
    self.train_loss(loss)
  
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
    self.net.save(self.save_filename(save_dir))

  def load(self, save_dir):
    self.net = tf.keras.models.load_model(self.save_filename(save_dir))

  def write_summaries(self, summary_writer, step):
    with summary_writer.as_default():
      tf.summary.scalar(self.train_loss.name, self.train_loss.result(), step=step)
