import pygame as pg
import pygame.image
import pygame.transform
import pygame.display
import pygame.draw
import pygame.surfarray

import gin
import gin.tf
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

import numpy as np
import PIL.Image
import random

BACKGROUND_COLOR = (0, 30, 120)
ACCELERATION = 0.03
DECELERATION = 0.03
MAX_VELOCITY = 500
MAX_STEPS = 2000

@gin.configurable
class SpaceshipEnv(py_environment.PyEnvironment):
    
  def __init__(
      self,
      screen_dimension=(1400, 1000),
      shaped_rewards=True
  ):
    self.screen_dimension = screen_dimension
    self.shaped_rewards = shaped_rewards

    self.action_space = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=2, name='action'
    )
    
    self.step_num = 0

    self.observation_space = array_spec.BoundedArraySpec(
        shape=(7,),
        dtype=np.float32,
        minimum=-100,
        name='observation',
    )

    self.spaceship = Spaceship(
        np.array([100, 100], dtype=np.int32),
        np.array([0, 0], dtype=np.int32),
        0,
        [25, 50],
    )
    self.target = Target(
        np.array([500, 500]), np.array([0, 0]), (30, 30), (200, 30, 30)
    )
    self.ui = Ui(self.screen_dimension)
    super(SpaceshipEnv, self).__init__(handle_auto_reset=True)

  def action_spec(self):
    return self.action_space

  def observation_spec(self):
    return self.observation_space

  def return_state(self):
    return np.array([
        self.spaceship.pos[0],
        self.spaceship.pos[1],
        self.spaceship.velocity[0],
        self.spaceship.velocity[1],
        self.spaceship.rotation,
        self.target.pos[0],
        self.target.pos[1],
    ], dtype=np.float32)

  def _step(self, action):
    first_distance = abs(self.spaceship.pos[0] - self.target.pos[0]) + abs(
        self.spaceship.pos[1] - self.target.pos[1]
    )
    rotate = -1 if action == 2 else action
    self.spaceship.rotation += 90 * rotate
    self.spaceship.rotation %= 360

    self.spaceship.pos += self.spaceship.velocity

    done = False
    reward = -0.1

    if action == 0:
      reward += 0.1

    if self.shaped_rewards:
      if (
          abs(self.spaceship.velocity[0]) < 1
          and abs(self.spaceship.velocity[1]) < 1
      ):
        reward -= 50

      distance = abs(self.spaceship.pos[0] - self.target.pos[0]) + abs(
          self.spaceship.pos[1] - self.target.pos[1]
      )

      if distance < first_distance:
        max_velocity = max(self.spaceship.velocity)
        if max_velocity > 0.5:
          reward += 15 * max(self.spaceship.velocity)
        else:
          reward -= 100

    if self.check_for_collision(self.spaceship, self.target):
      reward += 1000
      done = True

    if self.check_for_out_of_bounds(self.spaceship):
      reward -= 500
      done = True

    self.step_num = self.step_num + 1
    if self.step_num >= MAX_STEPS:
      done = True

    self.update(self.spaceship)

    state = self.return_state()
    # print('state:', state)
    
    if done:
      return ts.termination(state, reward=reward)
    else:
      return ts.transition(state, reward=reward, discount=1.0)

  def update(self, object1):
    for i in range(object1.velocity.shape[0]):
      if object1.velocity[i] > 0:
        object1.velocity[i] = min(
            max(0, object1.velocity[i] - DECELERATION), 100
        )
      else:
        object1.velocity[i] = max(
            min(0, object1.velocity[i] + DECELERATION), -100
        )
    if object1.rotation == 0:
      object1.velocity[0] += 2 * ACCELERATION
    elif object1.rotation / 90 == 1:
      object1.velocity[1] -= 2 * ACCELERATION
    elif object1.rotation / 90 == 2:
      object1.velocity[0] -= 2 * ACCELERATION
    elif object1.rotation / 90 == 3:
      object1.velocity[1] += 2 * ACCELERATION
      
    object1.velocity[0] = min(object1.velocity[0], MAX_VELOCITY)
    object1.velocity[1] = min(object1.velocity[1], MAX_VELOCITY)

  def check_for_collision(self, object1, object2):
    if object1.rotation == 0 or object1.rotation == 180:
      sizes = [object1.size[1], object1.size[0]]
    else:
      sizes = [object1.size[0], object1.size[1]]
    # print(object1.pos[0] - object2.pos[0], width + object2.size[0])
    # print(object1.pos[1] - object2.pos[1], height + object2.size[1])
    # print()

    dists = [object1.pos[0] - object2.pos[0], object1.pos[1] - object2.pos[1]]
    for i, dist in enumerate(dists):
      if dist < 0:
        if dist < -(sizes[i] + object2.size[i]):
          return False
      else:
        if dist > object2.size[i]:
          return False
    return True

  def check_for_out_of_bounds(self, object1):
    if object1.pos[0] < 0:
      return True
    if object1.pos[1] < 0:
      return True
    if object1.pos[0] + object1.size[0] > self.screen_dimension[0]:
      return True
    if object1.pos[1] + object1.size[1] > self.screen_dimension[1]:
      return True
    return False

  def render(self, mode='rgb_array'):
    pixels = self.ui.draw(self.spaceship, self.target)
    pixels = pixels.transpose(1, 0, 2)  # convert from pygame coordinates
    # image = PIL.Image.fromarray(pixels)
    # return image.convert('RGB')
    return pixels

  def _reset(self):
    self.step_num = 0
    if self.check_for_out_of_bounds(self.spaceship):
      self.spaceship = Spaceship(
          np.array([100, 100], dtype=np.float32),
          np.array([0, 0], dtype=np.float32),
          0,
          [25, 50],
      )
    random_pos = [
        random.randrange(0, self.screen_dimension[0]),
        random.randrange(0, self.screen_dimension[1]),
    ]
    self.target = Target(
        np.array(random_pos), np.array([0, 0]), (30, 30), (200, 30, 30)
    )

    return ts.restart(self.return_state())


class Body:

  def __init__(self, pos, velocity, size):
    self.pos = pos
    self.velocity = velocity
    self.size = size

  def draw(self, screen):
    return


class Spaceship(Body):

  def __init__(self, pos, velocity, rotation, size):
    super().__init__(pos, velocity, size)
    self.rotation = rotation
    self.size = size
    self.image = None
    

  def draw(self, screen):
    if self.image == None:
      self.image = pg.image.load(
          'images/spaceship_image.bmp'
      )
      self.image = pg.transform.scale(self.image, self.size)
      
    image = pg.transform.rotate(self.image, self.rotation - 90)
    screen.blit(image, self.pos)

class Target(Body):

  def __init__(self, pos, velocity, size, color):
    super().__init__(pos, velocity, size)
    self.color = color

  def draw(self, screen):
    pg.draw.circle(screen, self.color, self.pos, self.size[0])


class Ui:

  def __init__(self, screen_dimension):
    self.screen_dimension = screen_dimension
    self.screen = pg.surface.Surface(self.screen_dimension)

  def draw(self, spaceship, target):
    self.screen.fill(BACKGROUND_COLOR)
    spaceship.draw(self.screen)
    target.draw(self.screen)
    return pg.surfarray.pixels3d(self.screen)


if __name__ == '__main__':
  env = SpaceshipEnv()

  env.render(mode='human')
  # pause = input('pause')
  clock = pg.time.Clock()

  while True:
    clock.tick(100)
    action = 0
    for event in pg.event.get():
      if event.type == pg.QUIT:
        pg.quit()
      if event.type == pg.KEYDOWN:
        if event.key == pg.K_a:
          action = 1
        elif event.key == pg.K_d:
          action = 2
        elif event.key == pg.K_p:
          unpause = False
          while True:
            if unpause:
              break
            for event in pg.event.get():
              if event.type == pg.QUIT:
                pg.quit()
              if event.type == pg.KEYDOWN:
                print('keydown')
                if event.key == pg.K_p:
                  unpause = True
                  break

    # if pg.key.get_pressed()[pg.K_a]:
    #     env.step(2)
    # elif pg.key.get_pressed()[pg.K_d]:
    #     env.step(1)
    # else:
    #     env.step(0)
    time_step = env.step(action)
    if time_step.is_last():
      env.reset()
      done = False
    env.render()
