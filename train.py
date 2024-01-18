from __future__ import absolute_import, division, print_function

from absl import logging
from absl import flags
from absl import app

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
from forward_dynamics import ForwardDynamics
from goaly import Goaly

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
  _, train_dir, _, saved_model_dir, _ = spaceship_util.get_dirs()

  env = SpaceshipEnv()

  #
  # Agent
  #
  step_var = tf.Variable(0, dtype=tf.int64)
  step_var.assign(0)
  step = step_var.numpy()

  input_dims = env.observation_space.shape[0]

  goaly = Goaly(step_var)
  num_goals = goaly.num_goals

  replay_buffer = ReplayBuffer(replay_buffer_max_length, input_dims, num_goals)

  inverse_dynamics = InverseDynamics(
    step_var=step_var,
    replay_buffer=replay_buffer,
    checkpoints_dir=train_dir,
    input_dims=input_dims,
    num_actions=env.action_space.n)
  inverse_dynamics.restore(load_dir)

  forward_dynamics = ForwardDynamics(
    step_var=step_var,
    replay_buffer=replay_buffer,
    checkpoints_dir=train_dir,
    num_goals=num_goals,
    num_actions=env.action_space.n, 
    input_dims=input_dims)
  forward_dynamics.restore(load_dir)

  agent = Agent(
    step_var=step_var, 
    replay_buffer=replay_buffer,
    checkpoints_dir=train_dir,
    num_actions=env.action_space.n, 
    input_dims=input_dims, 
    num_goals=num_goals)
  agent.restore(load_dir)

  summary_writer = tf.summary.create_file_writer(train_dir)

  step = step_var.numpy()

  # Summary data
  start_step = step
  state, _ = env.reset()
  timed_at_step, time_acc = step, 0

  while step < num_steps:
    logging.set_verbosity(logging.INFO)
    start_time = time.time()

    goals = goaly.goal(state)

    action = agent.policy(goals, state)
    new_state, reward, terminated, truncated, _ = env.step(action)
    
    done = terminated
    predicted_action_error = inverse_dynamics.predicted_action_error(state, new_state, action)
    predicted_goal_error = forward_dynamics.predicted_goals_error(state, action, goals)
    replay_buffer.store_tuples(state, goals, action, reward, new_state, done)
    state = new_state

    if step > (initial_collect_steps + start_step):
      agent.train()
      forward_dynamics.train()
      inverse_dynamics.train()
      
    time_acc += time.time() - start_time
    step_var.assign_add(1)
    step = step_var.numpy()

    if done:
      # Reset Summary data
      state, _ = env.reset()

    if step % log_interval == 0:
      steps_per_sec = (step - timed_at_step) / time_acc
      timed_at_step, time_acc = step, 0
      
      print("Step: {}, Steps/Sec: {:3f}".format(step, steps_per_sec))

    if step % save_interval == 0:
      save_dir = os.path.join(saved_model_dir, "{0}".format(step))
      spaceship_util.ensure_dir(save_dir)
      agent.save(save_dir)
      forward_dynamics.save(save_dir)
      inverse_dynamics.save(save_dir)

    if step % checkpoint_interval == 0:
      agent.checkopint_manager.save()
      forward_dynamics.checkopint_manager.save()
      inverse_dynamics.checkopint_manager.save()

    if step % eval_interval == 0:
      # Eval, set load dir to None to force load from checkpoint
      eval(load_dir=None) 

    if step % summary_interval == 0:
      agent.write_summaries(step)
      forward_dynamics.write_summaries(step)
      inverse_dynamics.write_summaries(step)


if __name__ == "__main__":

  def main(argv):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    train()
    
  FLAGS = flags.FLAGS
  
  flags.DEFINE_multi_string(
    'gin_file', 'config/train.gin', 'List of paths to the config files.')
  flags.DEFINE_multi_string(
    'gin_param', None, 'Newline separated list of Gin parameter bindings.')
  app.run(main)
