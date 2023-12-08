import os
import time

import ffmpeg
import imageio

def create_policy_eval_video(py_env, policy, filename, num_episodes=5, fps=30):
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = py_env.reset()
      video.append_data(py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = py_env.step(action_step.action)
        video.append_data(py_env.render())


def get_dirs(root_dir):
  root_dir = os.path.expanduser(root_dir)
  train_dir = os.path.join(root_dir, 'train')
  eval_dir = os.path.join(root_dir, 'eval')
  saved_model_dir = os.path.join(root_dir, 'policy_saved_model')

  return root_dir, train_dir, eval_dir, saved_model_dir
