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



# flags.DEFINE_multi_string('gin_param', None, 'Gin parameter bindings.')
# FLAGS = flags.FLAGS

# gin.parse_config_files_and_bindings('config/base.gin', FLAGS.gin_param)



@gin.configurable
def eval(
    load_dir,
    num_steps,
    max_steps_per_episode):

  _, train_dir, eval_dir, _, tensorboard_dir, videos_dir = spaceship_util.get_dirs()

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
  episode_start_step = step
  score = 0.0
  score_ave = tf.keras.metrics.Mean('score', dtype=tf.float32)
  metrics = []
  action_guess_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='action_guess_accuracy', dtype=None
    )
  metrics.append(action_guess_accuracy)

  goal_guess_overall_accuracies = tf.keras.metrics.BinaryAccuracy(
        'goals_overall_accuracy', threshold=0.5, dtype=None)
  metrics.append(goal_guess_overall_accuracies)

  goal_guess_accuracies = list(map(
      lambda n: tf.keras.metrics.Mean(
        'goal_{}_guess_accuracy'.format(n), dtype=tf.float32), 
      range(num_goals)
    ))
  metrics.extend(goal_guess_accuracies)

  state, _ = env.reset()

  video_filename = os.path.join(videos_dir, "eval-{}.gif".format(step))
  video_recorder = VideoRecorder(env, video_filename)
  video_recorder.capture_frame()

  for _ in range(num_steps):
    logging.set_verbosity(logging.INFO)

    goals = goaly.goal(state)

    action = agent.policy(goals, state)
    new_state, reward, terminated, truncated, _ = env.step(action)

    video_recorder.capture_frame()

    done = terminated
    score += reward
    
    predicted_actions_probabilibies = inverse_dynamics.infer_action(state, new_state) 
    predicted_action = inverse_dynamics.action_from_probabilities(predicted_actions_probabilibies)
    action_guess_accuracy.update_state(action, predicted_actions_probabilibies)

    predicted_goal_probabilities = forward_dynamics.infer_goals(state, action)

    # print(predicted_goal_probabilities)
    goal_guess_overall_accuracies.update_state(goals, predicted_goal_probabilities)
    for i in range(num_goals):
      goal_guess_accuracies[i](goals[i] == (predicted_goal_probabilities[i] > 0.5))

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
  score_ave.result()
  print("")
  print("Episodes {}, Step: {}".format(episode, step))
  print("AVG Score: {:2.2f}".format(score_ave.result()))
  print("Inverse Accuracy: {:0.2f}".format(action_guess_accuracy.result()))
  print("Goals Overall Accuracy: {:0.2f}".format(goal_guess_overall_accuracies.result()))
  for i in range(num_goals):
    print("Goal {} Accuracy: {:0.2f}".format(i, goal_guess_accuracies[i].result()))
  print("")

  summary_writer = tf.summary.create_file_writer(tensorboard_dir)
  with summary_writer.as_default():
    for metric in metrics:
      tf.summary.scalar(metric.name, metric.result(), step=step)

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
