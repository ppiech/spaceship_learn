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

  def goal(self, observation):
    return self.step_var.np() / 100 

  def goal(self, observation):
    return np.zeros(self.num_goals)