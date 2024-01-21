import numpy as np


class ReplayBuffer:
  def __init__(self, size, input_shape, num_goals):
    self.size = size
    self.counter = 0
    self.state_buffer = np.zeros((self.size, input_shape), dtype=np.float32)
    self.new_state_buffer = np.zeros((self.size, input_shape), dtype=np.float32)
    self.goal_buffer = np.zeros((self.size, num_goals), dtype=np.float32)
    self.action_buffer = np.zeros(self.size, dtype=np.int32)
    self.reward_buffer = np.zeros(self.size, dtype=np.float32)
    self.bonus_buffer = np.zeros(self.size, dtype=np.float32)
    self.terminal_buffer = np.zeros(self.size, dtype=np.bool_)

  def store_tuples(self, state, new_state, goal, action, reward, bonus, done):
    idx = self.counter % self.size
    self.state_buffer[idx] = state
    self.new_state_buffer[idx] = new_state
    self.goal_buffer[idx] = goal
    self.action_buffer[idx] = action
    self.reward_buffer[idx] = reward
    self.bonus_buffer[idx] = bonus
    self.terminal_buffer[idx] = done
    self.counter += 1

  def sample_buffer(self, batch_size):
    max_buffer = min(self.counter, self.size)
    batch = np.random.choice(max_buffer, batch_size, replace=False)

    return \
      self.state_buffer[batch], \
      self.new_state_buffer[batch], \
      self.goal_buffer[batch], \
      self.action_buffer[batch], \
      self.reward_buffer[batch], \
      self.bonus_buffer[batch], \
      self.terminal_buffer[batch]
