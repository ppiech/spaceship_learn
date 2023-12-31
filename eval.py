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
    max_steps_per_episode,
    summary_interval
):

  root_dir, train_dir, eval_dir, saved_model_dir, tensorboard_dir = spaceship_util.get_dirs()

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
  episode_start_step = step
  scores = []
  score = 0.0
  inverse_dynamic_guess_rate = (0.0, 0.0) # (correct, total)

  state, _ = train_env.reset()

  while step < num_steps:
    logging.set_verbosity(logging.INFO)

    action = agent.policy(state)
    new_state, reward, terminated, truncated, _ = train_env.step(action)
    done = terminated
    score += reward
    predicted_action = inverse_dynamics.infer_action(state, new_state) 

    inverse_dynamic_guess_rate = (
      inverse_dynamic_guess_rate[0] + (predicted_action == action), 
      inverse_dynamic_guess_rate[1] + 1)
    state = new_state

    step_var.assign_add(1)
    step = step_var.numpy()

    if done or (step - episode_start_step)  > max_steps_per_episode:
      scores.append(score)

      # Reset Summary data
      episode += 1
      score = 0.0
      episode_start_step = step
      state, _ = train_env.reset()

  # Print summary
  avg_score = np.mean(scores)
  inverse_action_accuracy = inverse_dynamic_guess_rate[0] / inverse_dynamic_guess_rate[1]
  print("Episodes {}, Step: {} , AVG Score: {:2.2f}, Inverse Accuracy: {:0.2f}"
        .format(episode, step, avg_score, inverse_action_accuracy))

if __name__ == "__main__":
  gin.parse_config_file('config/base.gin')
  eval()