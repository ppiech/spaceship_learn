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

import eval

@gin.configurable
def train(
    policy_file,
    num_steps,
    initial_collect_steps,
    replay_buffer_max_length,
    eval_interval,
    log_interval,
    policy_checkpoint_interval,
    policy_save_interval,
    summary_interval=10,
):

  _, train_dir, _, saved_model_dir = spaceship_util.get_dirs()

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
      checkpoint=agent.policy_checkpoint,
      directory=os.path.join(train_dir),
      max_to_keep=2)

  if policy_file:
    agent.load(policy_file)
  elif policy_checkpointer.latest_checkpoint:
    agent.restore_from_checkpoint(policy_checkpointer.latest_checkpoint)

  step = step_var.numpy()

  inverse_dynamics = InverseDynamics(
    replay_buffer=replay_buffer,
    num_actions=train_env.action_space.n, 
    input_dims=input_dims)

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
    replay_buffer.store_tuples(state, action, reward, new_state, done)
    predicted_action = inverse_dynamics.infer_action(state, new_state)
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
      print("Episode {0}, Step: {1} , AVG Score: {2}, Steps/sec: {3}"
            .format(episode, step, avg_score, steps_per_sec))

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

if __name__ == "__main__":
  gin.parse_config_file('config/train.gin')
  train()