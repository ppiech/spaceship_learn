import os
from absl import logging
import gin
import gin.tf
from spaceship_agent import train_agent
from spaceship_env import SpaceshipEnv
from spaceship_util import create_policy_eval_video
from spaceship_util import get_dirs
from tensorboard.backend.event_processing import event_file_loader
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts


@gin.configurable
def eval(
    root_dir='log',
    eval_step=None,
    num_eval_episodes=10,
    summaries_flush_secs=10,
    save_video=False,
):
  # Directories
  root_dir, train_dir, eval_dir, _ = get_dirs(root_dir)

  if eval_step is None:
    eval_step = find_latest_eval_step(eval_dir)

  checkpoint_path = None
  if eval_step is not None:
    checkpoint_path = os.path.join(train_dir, 'ckpt-{}'.format(eval_step))

  # Environment
  eval_py_env = SpaceshipEnv()
  eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  # Agent
  global_step_tf = tf.Variable(0, dtype=tf.int64)

  agent = train_agent(eval_env, eval_py_env.action_spec(), global_step_tf)
  agent.initialize()

  eval_policy = greedy_policy.GreedyPolicy(agent.policy)

  # Run the agent on dummy data to force instantiation of the network.
  dummy_obs = tensor_spec.sample_spec_nest(
      eval_env.observation_spec(), outer_dims=(eval_env.batch_size,)
  )
  eval_policy.action(
      ts.restart(dummy_obs, batch_size=eval_env.batch_size),
      eval_policy.get_initial_state(eval_env.batch_size),
  )
  policy_checkpoint = tf.train.Checkpoint(
      policy=agent.policy, global_step=global_step_tf
  )
  logging.info('Loading checkpoint: %s', checkpoint_path)
  load_status = policy_checkpoint.restore(checkpoint_path)

  # Initialize the agent.
  load_status.initialize_or_restore()
  logging.info('Loaded checkpoint at global_step %d', global_step_tf.numpy())
  global_step = global_step_tf.numpy()

  # Eval Metrics
  eval_summary_writer = tf.compat.v2.summary.create_file_writer(
      eval_dir, flush_millis=summaries_flush_secs * 1000
  )
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]
  metric_utils.eager_compute(
      eval_metrics,
      eval_env,
      eval_policy,
      num_episodes=num_eval_episodes,
      summary_writer=eval_summary_writer,
      train_step=global_step,
  )

  metric_utils.log_metrics(eval_metrics)
  eval_summary_writer.flush()

  video_file = os.path.join(eval_dir, 'video-{}.mp4'.format(eval_step))
  create_policy_eval_video(
      py_env=eval_py_env, policy=eval_policy, filename=video_file
  )


def find_latest_eval_step(eval_dir: str):
  if not tf.io.gfile.exists(eval_dir):
    return None

  steps = []
  for events_file in tf.io.gfile.listdir(eval_dir):
    loader = event_file_loader.EventFileLoader(
        os.path.join(eval_dir, events_file)
    )
    for event in loader.Load():
      steps.append(event.step)

  return max(steps)
