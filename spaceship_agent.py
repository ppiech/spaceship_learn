import gin
import gin.tf
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import sequential
from tf_agents.specs import tensor_spec
from tf_agents.utils import common


@gin.configurable
def policy_network(action_spec, fc_layer_params=(256, 256)):
  action_tensor_spec = tensor_spec.from_spec(action_spec)
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

  return q_net

@gin.configurable
def train_agent(
    train_env,
    action_spec,
    train_step_counter,
    learning_rate=1e-3,
):
  q_net = policy_network(action_spec)
  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  agent = dqn_agent.DqnAgent(
      train_env.time_step_spec(),
      train_env.action_spec(),
      q_network=q_net,
      optimizer=optimizer,
      td_errors_loss_fn=common.element_wise_squared_loss,
      train_step_counter=train_step_counter,
  )
  agent.initialize()

  return agent
