from absl import app
from absl import flags
from absl import logging
import gin
import gin.tf
from spaceship_train import train
from spaceship_eval import eval

# from spaceship_env import SpaceshipEnv

ROOT_DIR = flags.DEFINE_string(
    'root_dir', 'log', 'Root directory for writing log and checkpoints'
)

NUM_ITERATIONS = flags.DEFINE_integer(
    'num_iterations', 40000, 'Training iterations to run'
)

GIN_FILE = flags.DEFINE_multi_string(
    'gin_file', None, 'Paths to the study config files.'
)

GIN_PARAM = flags.DEFINE_multi_string(
    'gin_param', None, 'Gin binding to pass through.'
)

EVAL = flags.DEFINE_boolean('eval', 'false', 'Whether to run evaluation only')

EVAL_STEP = flags.DEFINE_integer('eval_step', None, 'Step to evaluate')

FLAGS = flags.FLAGS

def main(_) -> None:
  logging.set_verbosity(logging.INFO)

  gin.parse_config_files_and_bindings(
      GIN_FILE.value, GIN_PARAM.value, skip_unknown=True
  )

  if EVAL.value:
    eval(ROOT_DIR.value, EVAL_STEP.value)
  else:
    train(FLAGS.root_dir, num_iterations=FLAGS.num_iterations)
    
if __name__ == '__main__':
  # This isn't used when launching with XManager.
  app.run(main)
