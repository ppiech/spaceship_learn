from __future__ import absolute_import, division, print_function

from absl import logging
import os
import time
import gin
import gin.tf
import numpy as np

import tensorflow as tf

from spaceship_env import SpaceshipEnv

from spaceship_agent import Agent
from replay_buffer import ReplayBuffer
from inverse_dynamics import InverseDynamics

import spaceship_util

from eval import eval

@gin.configurable
def train(
    load_dir,
    num_steps,
    initial_collect_steps,
    replay_buffer_max_length,
    eval_interval,
    log_interval,
    checkpoint_interval,
    save_interval,
    summary_interval,
):

  _, train_dir, _, saved_model_dir, summary_dir = spaceship_util.get_dirs()

  train_env = SpaceshipEnv()

  #
  # Agent
  #
  step_var = tf.Variable(0, dtype=tf.int64)
  step_var.assign(0)
  step = step_var.numpy()

  input_dims = train_env.observation_space.shape[0]

  replay_buffer = ReplayBuffer(replay_buffer_max_length, input_dims)

  agent = Agent(
    step_var=step_var, 
    replay_buffer=replay_buffer,
    num_actions=train_env.action_space.n, 
    input_dims=input_dims)

  policy_checkpointer = tf.train.CheckpointManager(
      checkpoint=agent.checkpoint,
      directory=os.path.join(train_dir, 'policy'),
      max_to_keep=2)

  if load_dir:
    agent.load(load_dir)
  elif policy_checkpointer.latest_checkpoint:
    agent.restore_from_checkpoint(policy_checkpointer.latest_checkpoint)

  inverse_dynamics = InverseDynamics(
    step_var=step_var, 
    replay_buffer=replay_buffer,
    num_actions=train_env.action_space.n, 
    input_dims=input_dims)

  inverse_dynamics_checkpointer = tf.train.CheckpointManager(
    checkpoint=inverse_dynamics.checkpoint,
    directory=os.path.join(train_dir, 'inverse_dynamics'),
    max_to_keep=2)

  if load_dir:
    inverse_dynamics.load(load_dir)
  elif inverse_dynamics_checkpointer.latest_checkpoint:
    inverse_dynamics.restore_from_checkpoint(inverse_dynamics_checkpointer.latest_checkpoint)

  summary_writer = tf.summary.create_file_writer(summary_dir)

  step = step_var.numpy()

  # Summary data
  start_step = step
  state, _ = train_env.reset()
  timed_at_step, time_acc = step, 0

  while step < num_steps:
    logging.set_verbosity(logging.INFO)
    start_time = time.time()

    action = agent.policy(state)
    new_state, reward, terminated, truncated, _ = train_env.step(action)
    done = terminated
    replay_buffer.store_tuples(state, action, reward, new_state, done)
    predicted_action = inverse_dynamics.infer_action(state, new_state)
    state = new_state

    if step > (initial_collect_steps + start_step):
      agent.train()
      inverse_dynamics.train()

    time_acc += time.time() - start_time
    step_var.assign_add(1)
    step = step_var.numpy()

    if done:
      # Reset Summary data
      state, _ = train_env.reset()

    if step % log_interval == 0:
      steps_per_sec = (step - timed_at_step) / time_acc
      timed_at_step, time_acc = step, 0
      
      print("Step: {}, Steps/Sec: {:3f}".format(step, steps_per_sec))

    if step % save_interval == 0:
      save_dir = os.path.join(saved_model_dir, "{0}".format(step))
      spaceship_util.ensure_dir(save_dir)
      agent.save(save_dir)
      inverse_dynamics.save(save_dir)

    if step % checkpoint_interval == 0:
      policy_checkpointer.save()
      inverse_dynamics_checkpointer.save()

    if step % eval_interval == 0:
      # Eval, set load dir to None to force load from checkpoint
      eval(load_dir=None) 

    if step % summary_interval == 0:
      agent.write_summaries(summary_writer, step)

if __name__ == "__main__":
  gin.parse_config_file('config/train.gin')
  train()