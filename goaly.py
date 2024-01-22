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

    self.bonus_metric = tf.keras.metrics.Mean('bonus', dtype=tf.float32)


  # def goal(self, observation):
  #   return self.step_var.np() / 100 

  def goal(self, observation):
    return np.zeros(self.num_goals)
  
  def bonus(self, action_error, goal_error):
    bonus = (1 - action_error) * goal_error + (1 - goal_error) * action_error 
    self.bonus_metric.update_state(bonus)
    return (1 - action_error) * goal_error + (1 - goal_error) * action_error
  

  def summaries(self):
    summarries = { self.bonus_metric.name: self.bonus_metric.result() }
    self.bonus_metric.reset_states()
    return summarries
