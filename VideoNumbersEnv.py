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
		self.observation_space = Box(0, 255, self.give_ob().shape, dtype=np.uint8)
		self.reward_range = (-np.inf, np.inf)
		self.metadata = None

	def step(self, action):
		# actions have no effect on environment
		self.state['timestep'] += 1
		ob = self.give_ob()
		info = dict(state=deepcopy(self.state))
		done = self.state['timestep'] >= 99
		reward = 0
		return ob, reward, self.done, info

	def give_ob(self):
		# Draws timestep on image, background pixel value also depends on timestep
		number_to_draw = np.min([99, self.state['timestep']])
		im = Image.fromarray(np.full([16, 16, 3], number_to_draw, dtype=np.uint8), mode='RGB')
		draw = ImageDraw.Draw(im)
		font = ImageFont.truetype("arial.ttf", 15)
		number_str = str(number_to_draw).rjust(2, '0')
		draw.text((0, 0), number_str, font=font)
		return np.array(im)

	def reset(self):
		self.state = dict(
			timestep=0,
		)
		ob = self.give_ob()
		self.done = False
		return ob


if __name__ == '__main__':
	env = VideoNumbersEnv()
	ob = env.reset()
	for i in range(10):
		ob, _,_,_ = env.step(0)
		print('ob corner', ob[0, 0, 0])