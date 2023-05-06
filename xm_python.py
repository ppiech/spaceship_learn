# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""XManager script for launching spaceship train on local workstation

EXAMPLE USAGE:
POOL=xx; ALLOC=yy  # Set to your team's resource pool and allocation.
google_xmanager launch xm_launch.py -- \
"""

from absl import app
from absl import flags
from xmanager import xm
from xmanager import xm_local


_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'spaceship train', 'Name of the experiment.', short_name='n'
)

_INTERACTIVE = flags.DEFINE_bool(
    'interactive',
    False,
    'Launch the container and allow interactive access to it.',
)

_TENSORBOARD_LOG_DIR = flags.DEFINE_string(
    'tensorboard_log_dir',
    None,
    'Log directory to be used by workers and Tensorboard.',
)

_TENSORBOARD_TIMEOUT_SECS = flags.DEFINE_integer(
    'tensorboard_timeout_secs',
    60 * 60,
    'The amount of time the Tensorboard job should run for.',
)

def main(_) -> None:

  with xm_local.create_experiment(
      'Launch XManager spaceship_learn.'
  ) as experiment:
    docker_options = xm_local.DockerOptions(interactive=_INTERACTIVE.value)
    executor = xm_local.Local(
        experimental_stream_output=True,
        docker_options=docker_options,
    )

    # The build target is expected to be a py_binary rule from which the .par
    # file will be built.
    
    spec = xm.PythonContainer(
        path='//experimental/users/ppiech/spaceship_learn',
        entrypoint=xm.ModuleName('spaceship_train'),
    )
    [executable] = experiment.package(
        [
            xm.Packageable(
                executable_spec=spec,
                executor_spec=xm_local.Local.Spec(),
                label='//experimental/users/ppiech/spaceship_learn - python',
            )
        ]
    )

    # Hyper-parameter definition.
    # Note that the hyperparameter arguments, if used, must correspond to flags
    # defined in your training binary.
    parameters = [{}]

    for hparams in parameters:
      experiment.add(
          xm.Job(
              executable=executable,
              executor=executor,
              args=hparams,
          )
      )

if __name__ == '__main__':
  app.run(main)  # This block is not executed when run under XManager.
