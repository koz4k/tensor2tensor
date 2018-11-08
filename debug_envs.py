from copy import deepcopy

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from gym.spaces.box import Box

from copy import deepcopy

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from gym.spaces.box import Box


class VideoNumbersEnv:
  def __init__(self):
    self.state = dict()
    self.state['timestep'] = -9

    self.action_space = gym.spaces.discrete.Discrete(3)
    self.reset()
    self.observation_space = Box(0, 255, self._give_ob().shape, dtype=np.uint8)
    self.reward_range = (-np.inf, np.inf)
    self.metadata = dict()
    self.unwrapped = self

  def close(self):
    return

  def step(self, action):
    # actions have no effect on environment
    self.state['timestep'] += 1
    ob = self._give_ob()
    info = dict(state=deepcopy(self.state))
    self.done = self.state['timestep'] >= 99
    reward = 0
    return ob, reward, self.done, info

  def _give_ob(self):
    # Draws timestep on image, background pixel value also depends on timestep
    number_to_draw = np.min([99, self.state['timestep']])
    im = Image.fromarray(np.full([16, 16, 3], number_to_draw, dtype=np.uint8),
                         mode='RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSans.ttf", 12)
    number_str = str(number_to_draw).rjust(2, '0')
    draw.text((0, 0), number_str, font=font)
    return np.array(im)

  def reset(self):
    self.state = dict(
      timestep=0,
    )
    ob = self._give_ob()
    self.done = False
    return ob


class VideoNumbersWithActions:
  def __init__(self):
    self.state = dict()

    self.action_space = gym.spaces.discrete.Discrete(3)
    self.reset()
    self.observation_space = Box(0, 255, self._give_ob().shape, dtype=np.uint8)
    self.reward_range = (-np.inf, np.inf)
    self.metadata = dict()
    self.unwrapped = self

  def close(self):
    return

  def step(self, action):
    # actions have no effect on environment
    self.state['number'] += action + 1
    self.state['last_action'] = action
    ob = self._give_ob()
    info = dict(state=deepcopy(self.state))
    self.done = self.state['number'] >= 99
    reward = 0
    return ob, reward, self.done, info

  def _give_ob(self):
    # Draws internal number and last action on image, number determines also
    # background color for world model
    # convenience
    text_to_draw = 'S{:02}\nA{}'.format(self.state['number'],
                                        self.state['last_action'])
    im = Image.fromarray(
      np.full([28, 28, 3], self.state['number'], dtype=np.uint8), mode='RGB')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype("DejaVuSans.ttf", 9)
    draw.multiline_text((0, 0), text_to_draw, font=font, spacing=1)
    return np.array(im)

  def reset(self):
    self.state = dict(
      last_action=' ',
      number=0
    )
    ob = self._give_ob()
    self.done = False
    return ob


class ActionBackground:
  def __init__(self, arena_size=50):
    self.state = dict()
    self.action_to_RGB = [np.array([0, 0, 255]), np.array([0, 255, 0]),
                          np.array([255, 0, 0]), np.array([100, 100, 100])]
    self.arena_size = arena_size
    self.action_space = gym.spaces.discrete.Discrete(4)
    self.reset()
    self.observation_space = Box(0, 255, self._give_ob().shape, dtype=np.uint8)
    self.reward_range = (-np.inf, np.inf)
    self.metadata = dict()
    self.unwrapped = self

  def close(self):
    return

  def step(self, action):
    self.state['timestep'] += 1
    self.state['last_action'] = action
    ob = self._give_ob()
    info = dict(state=deepcopy(self.state))
    self.done = self.state['timestep'] >= 99
    reward = 0
    return ob, reward, self.done, info

  def _give_ob(self):
    # Draws internal number and last action on image, number determines also
    # background color for world model
    # convenience
    background = self.action_to_RGB[self.state['last_action']]
    ob = np.full([self.arena_size, self.arena_size, 3], background,
                 dtype=np.uint8)
    return ob

  def reset(self):
    self.state = dict(
      last_action=0,
      timestep=0,
    )
    ob = self._give_ob()
    self.done = False
    return ob


if __name__ == '__main__':
  env = ActionBackground()
  ob = env.reset()
  for i in range(4):
    ob, _, _, _ = env.step(i)
    print('ob corner', ob[0, 0, :])
