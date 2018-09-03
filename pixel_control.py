from copy import deepcopy
from itertools import product

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from gym.spaces.box import Box
from scipy import ndimage


def random_discretization(x, rng=None):
	"""
	For x = n + r where n is integer and r in [0, 1] return n with probabilty r and n+1 with probabilty 1-r
	If x is an array, sample independently for each element.
	"""
	if not rng:
		rng = np.random.RandomState()
	x = np.array(x)
	floor = np.floor(x).astype(np.int32)
	return floor + rng.binomial(1, p=x - floor)


def rand_noise_on_img(img, rng=None, p=0.):
	if not rng:
		rng = np.random.RandomState()
	assert issubclass(img.dtype.type, np.uint8)
	mask = np.bool8(rng.binomial(1, p, size=img.shape))
	rand_img = rng.randint(256, size=img.shape, dtype=np.uint8)
	img[mask] = rand_img[mask]
	return img


class PixelControl:

	def __init__(self, obstacles_ratio=0.3, seed=None, area_size=6,
				 rescale_img=2, max_steps=18, only_reachable_init=True):
		self._obstacle_RGB = np.array([255, 0, 0])
		self._agent_RGB = np.array([0, 0, 255])
		self._goal_RGB = np.array([0, 255, 0])
		self.img_size = area_size
		self.n_goals = 1
		self.approx_obstacles_ratio = obstacles_ratio
		self.n_obstacles = int(area_size ** 2 * self.approx_obstacles_ratio)
		self.max_steps = max_steps
		assert self.max_steps <= 255
		self.backgroud_gradiation_per_step = 255 // max_steps
		self.state = dict()
		self.area_size = area_size
		assert self.n_obstacles + 2 <= self.area_size ** 2, 'to many obstacles'
		self.rescale_img = rescale_img
		self.only_reachable_init = only_reachable_init

		self.state = dict(
			agent_possition=None,
			goal_possition=None,
			obstacle_possitions=None,
			timestep=None,
		)

		self.seed(seed)
		self.metadata = None
		self.unwrapped = self

		self.reward_range = (-np.inf, np.inf)
		self.action_space = gym.spaces.discrete.Discrete(4)
		ob = self.reset()
		self.observation_space = Box(0, 255, ob.shape, dtype=np.uint8)

	def seed(self, seed):
		self.rng = np.random.RandomState(seed)

	def step(self, action):
		"""
		Move in one of 4 directions
		"""
		assert self.action_space.contains(action), 'action {} is not in action space {}'.format(action,
																								self.action_space)
		if self.done:
			reward = 0
			obs = self.give_ob()
			info = dict(state=deepcopy(self.state))
			return obs, reward, self.done, info

		proposed_poss = self.state['agent_possition'].copy()
		if action == 0:
			proposed_poss[0] += 1  # move right
		elif action == 1:
			proposed_poss[1] -= 1  # move up
		elif action == 2:
			proposed_poss[0] -= 1  # move left
		elif action == 3:
			proposed_poss[1] += 1  # move down

		# If move is possible, move the agent

		if (proposed_poss < self.area_size).all() and \
				(proposed_poss >= 0).all() and \
				(proposed_poss != self.state['obstacle_possitions']).any(axis=1).all():
			self.state['agent_possition'] = proposed_poss

		self.state['timestep'] += 1

		reward = 0
		if self.state['timestep'] >= self.max_steps:
			self.done = True
		elif (self.state['agent_possition'] == self.state['goal_possition']).all():
			reward = 1
			self.done = True

		info = dict(state=deepcopy(self.state))
		ob = self.give_ob()

		return ob, reward, self.done, info

	def close(self):
		return

	def give_ob(self, no_rescale=False):
		background_color = self.state['timestep'] * self.backgroud_gradiation_per_step
		ob = np.full([self.area_size, self.area_size, 3], background_color, dtype=np.uint8)

		goal = self.state['goal_possition']
		ob[goal[0], goal[1], :] = self._goal_RGB

		agent = self.state['agent_possition']
		ob[agent[0], agent[1], :] = self._agent_RGB

		obstacles = self.state['obstacle_possitions']
		ob[obstacles[:, 0], obstacles[:, 1], :] = self._obstacle_RGB

		if not no_rescale and self.rescale_img != 1:
			assert ob.dtype == np.uint8  # Image interprets other dtypes wrongly
			im = Image.fromarray(ob, mode='RGB')
			rescaled_size = self.area_size * self.rescale_img
			ob = np.array(im.resize((rescaled_size, rescaled_size), Image.BOX))
		return ob

	def reset(self):
		avilable_possitions = np.array([(i, j) for i, j in product(range(self.area_size), range(self.area_size))])
		object_ix = self.rng.choice(len(avilable_possitions),
									size=self.n_obstacles + 2,
									replace=False)
		object_possitions = avilable_possitions[object_ix, :]
		agent = object_possitions[0, :]
		goal = object_possitions[1, :]
		obstacles = object_possitions[2:, :]
		self.state = dict(
			agent_possition=agent,
			goal_possition=goal,
			obstacle_possitions=obstacles,
			timestep=0,
		)

		if self.only_reachable_init:
			ob_ = self.give_ob(no_rescale=True)
			obstacle_map_2d = (ob_ == self._obstacle_RGB.reshape(1, 1, 3)).all(axis=2)
			components_map, _ = ndimage.label(~obstacle_map_2d)
			goal_is_reachable = components_map[goal[0], goal[1]] == components_map[agent[0], agent[1]]
			if not goal_is_reachable:
				self.reset()

		ob = self.give_ob()
		self.done = False
		return ob


if __name__ == '__main__':
	env = PixelControl()
	ob = env.reset()
	print(ob)
