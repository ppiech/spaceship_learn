import os
import time
import tensorflow as tf
import imageio
import PIL
import gin
import datetime
import numpy as np

def ensure_dir(dir):
  if not tf.io.gfile.exists(dir):
    tf.io.gfile.makedirs(dir)
  return dir

@gin.configurable
def get_dirs(root_dir):
  root_dir = ensure_dir(os.path.expanduser(root_dir))
  train_dir = ensure_dir(os.path.join(root_dir, 'train'))
  eval_dir = ensure_dir(os.path.join(root_dir, 'eval'))
  saved_model_dir = ensure_dir(os.path.join(root_dir, 'saved_models'))
  videos_dir = ensure_dir(os.path.join(root_dir, 'videos'))

  return root_dir, train_dir, eval_dir, saved_model_dir, videos_dir

@gin.configurable
class VideoRecorder:
  def __init__(self, env, filename, num_steps_to_capture, capture_ever_n_frames, scale):
    self.env = env
    self.video_step = 0
    self.video_steps = num_steps_to_capture
    self.scale = scale
    self.capture_ever_n_frames = capture_ever_n_frames
    if self.video_steps > 0:
      frame_rate = 25 / self.capture_ever_n_frames
      self.video = imageio.get_writer(filename, fps=frame_rate)
    
  def capture_frame(self):
    if self.video_step < self.video_steps:
      if self.video_step % self.capture_ever_n_frames == 0:
        
        image_array = self.env.render()

        if self.scale != 1:
          im = PIL.Image.fromarray(image_array)
          width = int(im.size[0] * self.scale)
          height = int(im.size[1] * self.scale)
          im = im.resize((width, height), PIL.Image.NEAREST)
          image_array = np.array(im)

        self.video.append_data(image_array)

      self.video_step += 1
  