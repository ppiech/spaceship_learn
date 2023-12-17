from __future__ import absolute_import, division, print_function

from absl import logging
import gin
import gin.tf
import sys
import time
import os
import subprocess
import base64
import imageio
import ffmpeg
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import random

import tensorflow as tf

from spaceship_env import SpaceshipEnv
from spaceship_agent import Agent
import spaceship_util

import video_util

learning_rate = 1e-3  # @param {type:"number"}

train_sequence_length=1 # @param {type:"integer"}

@gin.configurable
def train(
    root_dir='log',
    num_iterations=40000,
    initial_collect_steps=1000,
    collect_steps_per_iteration=10,
    replay_buffer_max_length=100000,
    batch_size=64,
    num_eval_episodes=10,
    eval_interval=5000,
    log_interval=1000,
    train_checkpoint_interval=10000,
    policy_checkpoint_interval=5000,
    rb_checkpoint_interval=2000,
    summary_interval=10,
    summaries_flush_secs=10,
):

  root_dir, train_dir, eval_dir, saved_model_dir = spaceship_util.get_dirs(root_dir)

  train_env = SpaceshipEnv()
  eval_env = SpaceshipEnv()

  #
  # Agent
  #
  train_step_counter = tf.Variable(0, dtype=tf.int64)

  agent = Agent(lr=0.00075, discount_factor=0.99, num_actions=3, epsilon=0.03, batch_size=5000, input_dims=7)

  # if file:
  #   agent.load(file, env)
        
  scores, episodes, avg_scores, obj = [], [], [], []
  goal = 200
  f = 0
  steps = 0
  train = False

  for i in range(num_iterations):
    if i % 300 == 0 and i != 0:
      agent.save(f)
      f += 1
        
    done = False
    score = 0.0
    state, _ = train_env.reset()
    while not done:
      action = agent.policy(state)
      new_state, reward, terminated, truncated, _ = train_env.step(action)
      done = terminated
      score += reward
      agent.store_tuple(state, action, reward, new_state, done)
      state = new_state
      if steps % 10 and train == 0:
          agent.train()
      steps += 1
    # if self.buffer.counter > self.buffer.size - 100:
    #         train = True
    scores.append(score)
    obj.append(goal)
    episodes.append(i)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    
    print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_iterations, score, agent.epsilon,
                                                                       avg_score))
  # if avg_score >= 200.0 and score >= 250:
  agent.save(f)
  f += 1

  num_episodes = 3


# EPSILON LOWERED FOR NOW


train()