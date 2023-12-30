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

# import train # import for gin config

import spaceship_util
from inverse_dynamics import InverseDynamics

import video_util

@gin.configurable
def eval(
    load_dir,
    num_steps,
    num_eval_episodes,
    log_interval,
    summary_interval=10,
):

  root_dir, train_dir, eval_dir, saved_model_dir = spaceship_util.get_dirs()

  train_env = SpaceshipEnv()
  
  #
  # Agent
  #
  step_var = tf.Variable(0, dtype=tf.int64)
  step_var.assign(0)
  step = step_var.numpy()

  input_dims = train_env.observation_space.shape[0]

  agent = Agent(
    step_var=step_var, 
    replay_buffer=None,
    num_actions=train_env.action_space.n, 
    input_dims=input_dims)

  policy_checkpointer = tf.train.CheckpointManager(
      checkpoint=agent.checkpoint,
      directory=os.path.join(train_dir),
      max_to_keep=2)

  if load_dir:
    agent.load(load_dir)
  elif policy_checkpointer.latest_checkpoint:
    agent.restore_from_checkpoint(policy_checkpointer.latest_checkpoint)

  inverse_dynamics = InverseDynamics(
    step_var=step_var, 
    replay_buffer=None,
    num_actions=train_env.action_space.n, 
    input_dims=input_dims)

  step = step_var.numpy()

  # Summary data
  episode = 1
  scores = []
  score = 0.0
  inverse_dynamic_guess_rate = (0.0, 0.0) # (correct, total)

  state, _ = train_env.reset()
  timed_at_step, time_acc = step, 0

  while step < num_steps:
    logging.set_verbosity(logging.INFO)
    start_time = time.time()

    action = agent.policy(state)
    new_state, reward, terminated, truncated, _ = train_env.step(action)
    done = terminated
    score += reward
    predicted_action = inverse_dynamics.infer_action(state, new_state) 
    inverse_dynamic_guess_rate = (
      inverse_dynamic_guess_rate[0] + (predicted_action == action), 
      inverse_dynamic_guess_rate[1] + 1)
    state = new_state

    time_acc += time.time() - start_time
    step_var.assign_add(1)
    step = step_var.numpy()

    if done:
      # Print summary
      scores.append(score)
      avg_score = np.mean(scores[-100:])
      inverse_action_accuracy = inverse_dynamic_guess_rate[0] / inverse_dynamic_guess_rate[1]
      steps_per_sec = (step - timed_at_step) / time_acc
      print("Episode {}, Step: {} , Score: {:2.2f}, AVG Score: {:2.2f}, Inverse Dynamic Accuracy: {:0.2f}, Steps/sec: {:4.0f}"
            .format(episode, step, score, avg_score, inverse_action_accuracy, steps_per_sec))

      # Reset Summary data
      episode += 1
      score = 0.0
      state, _ = train_env.reset()
      timed_at_step, time_acc = step, 0

  agent.save(saved_model_dir, step)

if __name__ == "__main__":
  gin.parse_config_file('config/base.gin')
  eval()