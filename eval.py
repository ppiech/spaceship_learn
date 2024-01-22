from __future__ import absolute_import, division, print_function

from absl import logging
from absl import app
import os
import gin
import gin.tf
import numpy as np
from absl import flags

import tensorflow as tf

from spaceship_env import SpaceshipEnv
from spaceship_agent import Agent

from replay_buffer import ReplayBuffer

# import train # import for gin config

import spaceship_util
from inverse_dynamics import InverseDynamics
from forward_dynamics import ForwardDynamics
from spaceship_util import VideoRecorder
from goaly import Goaly

@gin.configurable
def eval(
    load_dir,
    num_steps,
    max_steps_per_episode):

  _, train_dir, eval_dir, _, videos_dir = spaceship_util.get_dirs()

  env = SpaceshipEnv()
  
  #
  # Agent
  #
  step_var = tf.Variable(0, dtype=tf.int64)
  step_var.assign(0)
  step = step_var.numpy()

  input_dims = env.observation_space.shape[0]

  goaly = Goaly(step_var)
  num_actions = env.action_space.n
  num_goals = goaly.num_goals

  inverse_dynamics = InverseDynamics(
    step_var=step_var, 
    replay_buffer=None,
    checkpoints_dir=train_dir,
    num_actions=env.action_space.n, 
    input_dims=input_dims)
  inverse_dynamics.restore(load_dir)

  forward_dynamics = ForwardDynamics(
    step_var=step_var,
    replay_buffer=None,
    checkpoints_dir=train_dir,
    num_goals=num_goals,
    num_actions=env.action_space.n, 
    input_dims=input_dims)
  forward_dynamics.restore(load_dir)

  agent = Agent(
    step_var=step_var, 
    replay_buffer=None,
    checkpoints_dir=train_dir,
    num_actions=env.action_space.n, 
    input_dims=input_dims, 
    num_goals=num_goals)
  agent.restore(load_dir)

  step = step_var.numpy()

  # Summary data
  episode = 1
  start_step = episode_start_step = step
  score = 0.0
  score_ave = tf.keras.metrics.Mean('score', dtype=tf.float32)

  state, _ = env.reset()

  video_filename = os.path.join(videos_dir, "eval-{}.gif".format(step))
  video_recorder = VideoRecorder(env, video_filename)
  video_recorder.capture_frame()


  for _ in range(num_steps):
    logging.set_verbosity(logging.INFO)

    goal = goaly.goal(state)

    action = agent.policy(goal, state)
    new_state, reward, terminated, truncated, _ = env.step(action)

    video_recorder.capture_frame()

    done = terminated
    score += reward    
    predicted_action_error = inverse_dynamics.predicted_action_error(state, new_state, action)
    predicted_goal_error = forward_dynamics.predicted_goal_error(state, action, goal)
    state = new_state

    step_var.assign_add(1)
    step = step_var.numpy()

    if done or (step - episode_start_step)  > max_steps_per_episode:
      score_ave(score)

      # Reset Summary data
      episode += 1
      score = 0.0
      episode_start_step = step
      state, _ = env.reset()
      video_recorder.capture_frame()

  # Print summary
  summaries = {}
  summaries.update(forward_dynamics.summaries())
  summaries.update(inverse_dynamics.summaries())
  summaries.update(goaly.summaries())
  summaries[score_ave.name] = score_ave.result()

  print("")
  for key in summaries: 
    print("{} = {:0.2f}".format(key, summaries[key]))
  print("")

  summary_writer = tf.summary.create_file_writer(eval_dir)
  with summary_writer.as_default():
    for key in summaries: 
      tf.summary.scalar(key, summaries[key], step=start_step)
    
if __name__ == "__main__":

  def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    eval()

  FLAGS = flags.FLAGS

  flags.DEFINE_multi_string(
    'gin_file', 'config/base.gin', 'List of paths to the config files.')
  flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')
  app.run(main)
