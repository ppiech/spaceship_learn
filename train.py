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
from tensorflow.keras.optimizers import Adam

import spaceship_util

import video_util

learning_rate = 1e-3  # @param {type:"number"}

train_sequence_length=1 # @param {type:"integer"}

@gin.configurable
def train(
    root_dir='log',
    policy_file=None,
    num_steps=1000000,
    initial_collect_steps=10000,
    collect_steps_per_iteration=10,
    replay_buffer_max_length=100000,
    batch_size=64,
    num_eval_episodes=10,
    eval_interval=500,
    log_interval=100,
    policy_checkpoint_interval=1000,
    policy_save_interval=2000,
    rb_checkpoint_interval=200,
    summary_interval=10,
    summaries_flush_secs=10,
):

  root_dir, train_dir, eval_dir, saved_model_dir = spaceship_util.get_dirs(root_dir)

  train_env = SpaceshipEnv()
  eval_env = SpaceshipEnv()

  #
  # Agent
  #
  step_var = tf.Variable(0, dtype=tf.int64)
  step_var.assign(0)
  step = step_var.numpy()

  agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=3, epsilon=0.03, batch_size=500, input_dims=7, step_var=step_var)

  # if file:
  #   agent.load(file, env)

  policy_checkpointer = tf.train.CheckpointManager(
      checkpoint=agent.policy_checkpoint,
      directory=os.path.join(train_dir),
      max_to_keep=2)

  if policy_file:
    agent.load(policy_file)
  elif policy_checkpointer.latest_checkpoint:
    agent.restore_from_checkpoint(policy_checkpointer.latest_checkpoint)

  step = step_var.numpy()

  # Summary data
  start_step = step
  episode = 1
  scores = []
  score = 0.0
  state, _ = train_env.reset()
  timed_at_step, time_acc = step, 0

  while step < num_steps:
    logging.set_verbosity(logging.INFO)
    start_time = time.time()

    action = agent.policy(state)
    new_state, reward, terminated, truncated, _ = train_env.step(action)
    done = terminated
    score += reward
    agent.store_tuple(state, action, reward, new_state, done)
    state = new_state

    if step > (initial_collect_steps + start_step):
      agent.train()

    time_acc += time.time() - start_time
    step_var.assign_add(1)
    step = step_var.numpy()

    if done:
      # Print summary
      scores.append(score)
      avg_score = np.mean(scores[-100:])
      steps_per_sec = (step - timed_at_step) / time_acc
      print("Episode {0}, Step: {1}, Score: {2} ({3}), AVG Score: {4}, Steps/sec: {5}"
            .format(episode, step, score, agent.epsilon, avg_score, steps_per_sec))

      # Reset Summary data
      episode += 1
      score = 0.0
      state, _ = train_env.reset()
      timed_at_step, time_acc = step, 0

    if step % policy_save_interval == 0:
      agent.save(saved_model_dir, step)

    if step % policy_checkpoint_interval == 0:
      policy_checkpointer.save()

  agent.save(saved_model_dir, step)


gin.parse_config_file('config/base.gin')

train()