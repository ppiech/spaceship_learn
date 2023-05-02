from collections.abc import Sequence
import os
import time

from absl import app
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from google3.experimental.users.ppiech.spaceship_learn.spaceship_env import SpaceshipEnv

# from spaceship_env import SpaceshipEnv

num_iterations = 40000  # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 10  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 5000  # @param {type:"integer"}

log_interval = 1000  # @param {type:"integer"}
train_checkpoint_interval = 10000  # @param {type:"integer"}
policy_checkpoint_interval = 5000  # @param {type:"integer"}
rb_checkpoint_interval = 2000  # @param {type:"integer"}
keep_rb_checkpoint = False  # @param {type:"boolean"}
train_sequence_length = 1  # @param {type:"integer"}
summary_interval = 1000  # @param {type:"integer"}
summaries_flush_secs = 10  # @param {type:"integer"}
root_dir = os.path.expanduser('/usr/local/google/home/ppiech/rl/log')
train_dir = os.path.join(root_dir, 'train')
eval_dir = os.path.join(root_dir, 'eval')
saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not tf.io.gfile.exists(saved_model_dir):
    tf.io.gfile.makedirs(saved_model_dir)

  #
  # Environment
  #

  train_py_env = SpaceshipEnv()
  eval_py_env = SpaceshipEnv()

  train_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  #
  # Policy
  #
  fc_layer_params = (256, 256)
  action_tensor_spec = tensor_spec.from_spec(train_py_env.action_spec())
  num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

  def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'
        ),
    )
  dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
  q_values_layer = tf.keras.layers.Dense(
      num_actions,
      activation=None,
      kernel_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.03, maxval=0.03
      ),
      bias_initializer=tf.keras.initializers.Constant(-0.2),
  )
  q_net = sequential.Sequential(dense_layers + [q_values_layer])

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  train_step_counter = tf.Variable(0, dtype=tf.int64)

  #
  # Agent
  #
  agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter,
  )
  agent.initialize()

  policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(train_dir, 'policy'),
      max_to_keep=None,
      policy=agent.policy,
      global_step=agent.train_step_counter,
  )

  saved_model = policy_saver.PolicySaver(
      greedy_policy.GreedyPolicy(agent.policy), train_step=train_step_counter
  )

  def save_policy(global_step_value):
    """Saves policy using both checkpoint saver and saved model."""
    policy_checkpointer.save(global_step=global_step_value)
    saved_model_path = os.path.join(
        saved_model_dir, 'policy_' + ('%d' % global_step_value).zfill(8)
    )
    saved_model.save(saved_model_path)

  eval_policy = agent.policy

  #
  # Replay buffer
  #

  table_name = 'uniform_table'
  replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
  replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)

  replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
      agent.collect_data_spec,
      batch_size=train_env.batch_size,
      max_length=replay_buffer_max_length,
  )

  rb_observer = replay_buffer.add_batch

  rb_ckpt_dir = os.path.join(train_dir, 'replay_buffer')
  rb_checkpointer = common.Checkpointer(
      ckpt_dir=rb_ckpt_dir, max_to_keep=1, replay_buffer=replay_buffer
  )

  #
  # Train Metrics
  #
  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
      tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
      tf_metrics.ChosenActionHistogram(),
  ]

  train_checkpointer = common.Checkpointer(
      ckpt_dir=train_dir,
      max_to_keep=1,
      agent=agent,
      global_step=agent.train_step_counter,
      metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'),
  )

  #
  # Driver
  #
  random_policy = random_tf_policy.RandomTFPolicy(
      train_env.time_step_spec(), train_env.action_spec()
  )

  DynamicStepDriver(
      train_env,
      agent.collect_policy,
      [rb_observer],
      num_steps=initial_collect_steps,
  ).run()

  # Dataset generates trajectories with shape [Bx2x...]
  dataset = replay_buffer.as_dataset(
      num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
  ).prefetch(3)

  iterator = iter(dataset)

  #
  # Eval Metrics
  #

  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000
  )

  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  eval_policy = greedy_policy.GreedyPolicy(agent.policy)

  metric_utils.eager_compute(
      eval_metrics,
      eval_env,
      eval_policy,
      num_episodes=num_eval_episodes,
      summary_writer=eval_summary_writer,
      train_step=0,
  )

  metric_utils.log_metrics(eval_metrics)

  #
  # Train Loop
  #

  # (Optional) Optimize by wrapping some of the code in a graph using TF 
  # function.
  agent.train = common.function(agent.train)

  # Reset the train step.
  agent.train_step_counter.assign(0)

  # Reset the environment.
  time_step = train_env.reset()

  # Create a driver to collect experience.
  collect_driver = DynamicStepDriver(
      train_env,
      agent.collect_policy,
      [rb_observer] + train_metrics,
      num_steps=collect_steps_per_iteration,
  )

  timed_at_step = agent.train_step_counter.numpy()
  time_acc = 0

  for _ in range(num_iterations):
    start_time = time.time()

    # Collect a few steps and save to the replay buffer.
    time_step, _ = collect_driver.run(time_step)

    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(iterator)
    train_loss = agent.train(experience).loss

    step_tf = agent.train_step_counter
    step = step_tf.numpy()

    time_acc += time.time() - start_time

    for train_metric in train_metrics:
      train_metric.tf_summaries(
          train_step=agent.train_step_counter, step_metrics=train_metrics[:2]
      )

    if step % train_checkpoint_interval == 0:
      train_checkpointer.save(global_step=step)

    if step % policy_checkpoint_interval == 0:
      save_policy(step)

    if step % rb_checkpoint_interval == 0:
      rb_checkpointer.save(global_step=step)

    if step % log_interval == 0:
      print('step = %d, loss = %f' % (step, train_loss.numpy()))
      steps_per_sec = (
          (step - timed_at_step) * collect_steps_per_iteration / time_acc
      )
      print('%.3f steps/sec' % steps_per_sec)
      tf.compat.v2.summary.scalar(
          name='env_steps_per_sec', data=steps_per_sec, step=step_tf
      )
      timed_at_step = step
      time_acc = 0

    if step % eval_interval == 0:
      metric_utils.eager_compute(
          eval_metrics,
          eval_env,
          eval_policy,
          num_episodes=num_eval_episodes,
          summary_writer=eval_summary_writer,
          train_step=step,
      )
      metric_utils.log_metrics(eval_metrics)

if __name__ == '__main__':
  # This isn't used when launching with XManager.
  app.run(main)
