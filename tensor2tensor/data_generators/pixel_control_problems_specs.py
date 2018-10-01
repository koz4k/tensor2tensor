# PixelControl
# pixel_control
from tensor2tensor.data_generators.gym_problems import GymDiscreteProblem, \
  GymDiscreteProblemWithAutoencoder, GymDiscreteProblemAutoencoded, \
  GymSimulatedDiscreteProblemForWorldModelEval, \
  GymSimulatedDiscreteProblemAutoencoded, GymSimulatedDiscreteProblem, \
  GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded
from tensor2tensor.utils import registry


@registry.register_problem
class GymPixelControlRandom(GymDiscreteProblem):

  @property
  def env_name(self):
    return "T2TGymPixelControl-v1"

  @property
  def min_reward(self):
    return -1

  @property
  def num_rewards(self):
    return 1

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymDiscreteProblemWithAgentOnPixelControl(GymRealDiscreteProblem,
                                                   GymPixelControlRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnPixelControlWithAutoencoder(
    GymDiscreteProblemWithAutoencoder, GymPixelControlRandom):
  pass


@registry.register_problem
class GymDiscreteProblemWithAgentOnPixelControlAutoencoded(
    GymDiscreteProblemAutoencoded, GymPixelControlRandom):
  pass


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnPixelControl(
    GymSimulatedDiscreteProblem, GymPixelControlRandom):
  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pixel_control"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOnPixelControl(
    GymSimulatedDiscreteProblemForWorldModelEval, GymPixelControlRandom):
  """Simulated pong for world model evaluation."""

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pixel_control"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemWithAgentOnPixelControlAutoencoded(
    GymSimulatedDiscreteProblemAutoencoded, GymPixelControlRandom):

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pixel_control_autoencoded"

  @property
  def num_testing_steps(self):
    return 100


@registry.register_problem
class GymSimulatedDiscreteProblemForWorldModelEvalWithAgentOnPixelControlAutoencoded(  # pylint: disable=line-too-long
    GymSimulatedDiscreteProblemForWorldModelEvalAutoencoded,
    GymPixelControlRandom):

  @property
  def initial_frames_problem(self):
    return "gym_discrete_problem_with_agent_on_pixel_control_autoencoded"

  @property
  def num_testing_steps(self):
    return 100

