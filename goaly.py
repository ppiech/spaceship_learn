import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import gin
import gin.tf

from inverse_dynamics import InverseDynamics
from replay_buffer import ReplayBuffer

@gin.configurable
class Goaly:
  def __init__(self, 
               step_var, 
               num_goals):
    
    self.num_goals = num_goals
    self.step_var = step_var
    self.bonus_history_length = 100
    self.goal_switch_bonus_level = 0.5

    self.bonus_metric = tf.keras.metrics.Mean('bonus', dtype=tf.float32)

    self.current_goal = np.zeros(self.num_goals)
    self.goal_bonus_history = np.ones((self.bonus_history_length,))

  def goal(self, observation):
    if np.average(self.goal_bonus_history) < self.goal_switch_bonus_level:
      self.current_goal[0] = 1 - self.current_goal[1]
    
    return self.current_goal
  
  def bonus(self, action_error, goal_error):
    bonus = - (1 - action_error) * goal_error - (1 - goal_error) * action_error 
    self.bonus_metric.update_state(bonus)

    self.goal_bonus_history[self.step_var.numpy() % self.bonus_history_length] = bonus
    
    return bonus
  
  def summaries(self):
    summarries = { self.bonus_metric.name: self.bonus_metric.result() }
    self.bonus_metric.reset_states()
    return summarries
