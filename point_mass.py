from copy import deepcopy

import gym
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from gym.spaces.box import Box


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


class PointMassFO:
	def __init__(self, n_directions=8, n_traps=3, rgb_obs=True, seed=None, step_cost=0.02, noise=0.,
				 const_noise_per_traj=False, discreate_reward=False, img_size=64):
		if const_noise_per_traj:
			raise NotImplementedError()
		self.noise = noise
		self.const_noise_per_traj = const_noise_per_traj
		self.step_cost = step_cost
		self.rgb_obs = rgb_obs
		self.n_directions = n_directions
		self.discreate_reward = discreate_reward
		self.img_size = img_size
		self.n_goals = 1
		self.n_traps = n_traps
		self.trap_radius = 0.15
		self.goal_radius = 0.2
		self.agent_img_radius = 0.1
		self.move_length = 0.2

		self.max_steps = 200

		self.state = dict()
		self.state['agent_possition'] = np.full(2, np.nan)
		self.state['goal_possition'] = np.full(2, np.nan)
		self.state['trap_possitions'] = np.full([self.n_traps, 2], np.nan)

		self.seed(seed)
		self.action_space = gym.spaces.discrete.Discrete(self.n_directions)

		self.reset()
		self.observation_space = Box(0, 255, self.give_ob().shape, dtype=np.uint8)
		self.reward_range = (-np.inf, np.inf)
		self.metadata = None
		self.unwrapped = self

	def seed(self, seed):
		self.rng = np.random.RandomState(seed)

	def step(self, action):
		"""
		Action moves agent in one of the self.n_direction directions.
		"""

		assert action in range(self.n_directions), 'action: {}, n_directions'.format(action, self.n_directions)
		move_direction = action / self.n_directions * 2 * np.pi
		move = np.array([np.cos(move_direction), np.sin(move_direction)]) * self.move_length
		agent_poss = self.state['agent_possition']
		agent_poss += move

		done = self.done

		dist_to_goal = np.sqrt(((agent_poss - self.state['goal_possition'])**2).sum())
		if self.n_traps > 0:
			dist_to_nearest_trap = (np.sqrt(((agent_poss - self.state['trap_possitions'])**2).sum(axis=1))).min()
		else:
			dist_to_nearest_trap = np.inf
		self.dist_to_goal = dist_to_goal
		self.dist_to_nearest_trap = dist_to_nearest_trap
		self.steps_done += 1
		crash = False
		goal_reached = False
		if done:
			reward = 0.
		else:
			if (np.abs(agent_poss) >= 1.).any() or \
					(dist_to_nearest_trap < self.trap_radius) or \
					(self.steps_done > self.max_steps):
				crash = True
			elif dist_to_goal < self.goal_radius:
				goal_reached = True

			if crash:
				# print('crash')
				reward = -1 - np.exp(dist_to_goal) / 10.  # for easier training
				done = True
			elif goal_reached:
				# print('goal_reached')
				reward = 1.
				done = True
			else:
				reward = -self.step_cost
		if self.discreate_reward:
			reward = random_discretization(reward, self.rng)

		self.done = done
		ob = self.give_ob()
		info = dict(state=deepcopy(self.state))
		if self.rgb_obs:
			info['rgb_ob'] = ob
		else:
			info['rgb_ob'] = self.give_rgb_ob()

		return ob, reward, self.done, info

	def close(self):
		return

	def state_observation(self):
		return np.concatenate([self.state['agent_possition'].copy(), self.state['goal_possition'].copy()] +
							  list(self.state['trap_possitions'].copy()))

	def give_rgb_ob(self):
		im = Image.fromarray(np.full([self.img_size, self.img_size, 3], 256, dtype=np.uint8), mode='RGB')
		draw = ImageDraw.Draw(im)
		state = self.state
		for trap_poss in state['trap_possitions']:
			self.draw_circle(draw, trap_poss, self.trap_radius, 'blue')
		self.draw_circle(draw, state['goal_possition'], self.goal_radius, 'yellow')
		self.draw_circle(draw, state['agent_possition'], self.agent_img_radius, 'red')
		ret = np.array(im)  # np.array(im.resize((self.img_size, self.img_size), Image.ANTIALIAS))
		if self.noise > 0.:
			ret = rand_noise_on_img(ret, self.rng, self.noise)
		return ret

	def give_ob(self):
		if self.rgb_obs:
			return self.give_rgb_ob()
		else:
			return self.give_num_ob()

	def give_num_ob(self):
		state = deepcopy(self.state)
		return np.concatenate([state['agent_possition'], state['goal_possition']] +
							  list(state['trap_possitions']))

	def reset(self):
		flip = self.rng.choice([-1, 1], 2)
		agent_poss = self.rng.uniform(-0.9, -0.3, 2) * flip
		goal_poss = self.rng.uniform(0.5, 1, 2) * flip
		trap_poss = self.rng.uniform(-0.4, 0.4, (self.n_traps, 2)) * flip
		self.steps_done = 0
		self.state = dict(
			agent_possition=agent_poss,
			goal_possition=goal_poss,
			trap_possitions=trap_poss
		)
		ob = self.give_ob()
		self.done = False
		return ob

	@staticmethod
	def draw_circle(draw, center, radius, color):
		# coordinates between -1, 1
		x, y = draw.im.size
		assert x == y
		center = np.array(center)
		lu = stand_to_pil_coord(center + np.array([-radius, radius]), x)
		rd = stand_to_pil_coord(center + np.array([radius, -radius]), x)
		bbox = [lu[0], lu[1], rd[0], rd[1]]

		draw.ellipse(tuple(bbox), fill=ImageColor.getrgb(color))
		return draw


def stand_to_pil_coord(coord, size):
	# point in [-1,1] square to PIL image pixel coordinates
	assert coord.size == 2
	arr = np.array(coord)
	return (np.array([arr[0], -arr[1]]) + 1) / 2 * size


if __name__ == '__main__':
	env = PointMassFO()
	ob = env.reset()
	print(ob)