import pygame as pg
import pygame.image
import pygame.transform
import pygame.display
import pygame.draw
import pygame.surfarray

import gin
import gym

# from tf_agents.environments import py_environment
# from tf_agents.environments import utils
# from tf_agents.specs import array_spec
# from tf_agents.environments import wrappers
# from tf_agents.trajectories import time_step as ts

import numpy as np
import PIL.Image
import random

BACKGROUND_COLOR = (0, 30, 120)
ACCELERATION = 0.03
DECELERATION = 0.03
MAX_VELOCITY = 500
MAX_STEPS = 2000

@gin.configurable
class SpaceshipEnv(gym.Env):
    
  def __init__(
      self,
      screen_dimension=(1400, 1000),
      shaped_rewards=True
  ):
    self.screen_dimension = screen_dimension
    self.shaped_rewards = shaped_rewards

    self.action_space = gym.spaces.Discrete(3, start=-1)

    self.step_num = 0

    self.observation_space = gym.spaces.Box(
        low = -100,
        high = 360,
        shape = (7,),
        dtype = 'uint8'
    )
    
    self.spaceship = Spaceship(
        np.array([100, 100], dtype=np.float32),
        np.array([0, 0], dtype=np.float32),
        0,
        [25, 50],
    )
    self.target = Target(
        np.array([500, 500]), np.array([0, 0]), (30, 30), (200, 30, 30)
    )

    # Late init UI once the display mode is known
    self.ui = None

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

  def step(self, action):
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
    
    return state, reward, done, None, None

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
    if self.ui == None:
      self.ui = Ui(self.screen_dimension, mode)

    if mode == 'human':
      pg.display.update()

    pixels = self.ui.draw(self.spaceship, self.target)
    pixels = pixels.transpose(1, 0, 2)  # convert from pygame coordinates

    if mode == 'human':
      pg.display.update()

    # image = PIL.Image.fromarray(pixels)
    # return image.convert('RGB')
    return pixels

  def reset(self):
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

    return self.return_state(), None


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

  def __init__(self, screen_dimension, mode):
    self.screen_dimension = screen_dimension
    if mode == 'human':
        self.screen = pg.display.set_mode((self.screen_dimension), pg.RESIZABLE)
        self.screen.fill(BACKGROUND_COLOR)
    else:
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

    new_state, reward, terminated, truncated, _ = env.step(action)
    print (new_state)

    if terminated:
      env.reset()
      done = False
    env.render(mode='human')
