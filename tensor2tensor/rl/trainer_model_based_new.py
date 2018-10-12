# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Training of model-based RL agents.

Example invocation:

python -m tensor2tensor.rl.trainer_model_based \
    --output_dir=$HOME/t2t/rl_v1 \
    --loop_hparams_set=rlmb_base \
    --loop_hparams='num_real_env_frames=10000,epochs=3'
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import copy
import datetime
import math
import os
import time

import gym

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.data_generators.gym_env import T2TGymEnv
from tensor2tensor.models.research import rl
from tensor2tensor.rl import rl_trainer_lib
from tensor2tensor.rl import trainer_model_based_params
from tensor2tensor.rl.envs.utils import InitialFrameChooser
from tensor2tensor.utils import trainer_lib

import tensorflow as tf


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("loop_hparams_set", "rlmb_base",
                    "Which RL hparams set to use.")
flags.DEFINE_string("loop_hparams", "", "Overrides for overall loop HParams.")
flags.DEFINE_string("job_dir_to_evaluate", "",
                    "Directory of a job to be evaluated.")
flags.DEFINE_string("eval_results_dir", "/tmp",
                    "Directory to store result of evaluation")


@contextlib.contextmanager
def temporary_flags(flag_settings):
  old_values = {}
  for flag_name, flag_value in flag_settings.items():
    old_values[flag_name] = getattr(FLAGS, flag_name)
    setattr(FLAGS, flag_name, flag_value)
  yield
  for flag_name, flag_value in old_values.items():
    setattr(FLAGS, flag_name, flag_value)


def _ppo_training_epochs(hparams, epoch, is_final_epoch, real_env_training):
  """Helper for PPO restarts."""
  if hparams.gather_ppo_real_env_data:
    assert hparams.real_ppo_epochs_num is 0, (
        "Should be put to 0 to enforce better readability")
    real_training_ppo_epochs_num = int(math.ceil(
        hparams.num_real_env_frames /
        (hparams.epochs*hparams.real_ppo_epoch_length)))
  else:
    real_training_ppo_epochs_num = hparams.real_ppo_epochs_num

  simulated_training_ppo_epochs_num = hparams.ppo_epochs_num

  if epoch == -1:
    assert real_env_training, (
        "Epoch -1 should only be used for PPO collection in real environment.")
    return real_training_ppo_epochs_num
  ppo_training_epochs = (epoch + 1) * (simulated_training_ppo_epochs_num
                                       + real_training_ppo_epochs_num)
  if is_final_epoch:  # Length of training in the final epoch is doubled.
    ppo_training_epochs += simulated_training_ppo_epochs_num
  if real_env_training:
    ppo_training_epochs += real_training_ppo_epochs_num
  return ppo_training_epochs


def setup_directories(base_dir, subdirs):
  base_dir = os.path.expanduser(base_dir)
  tf.gfile.MakeDirs(base_dir)

  all_dirs = {}
  for subdir in subdirs:
    dir_name = os.path.join(base_dir, subdir)
    tf.gfile.MakeDirs(dir_name)
    all_dirs[subdir] = dir_name
  return all_dirs


def make_relative_timing_fn():
  """Make a function that logs the duration since it was made."""
  start_time = time.time()

  def format_relative_time():
    time_delta = time.time() - start_time
    return str(datetime.timedelta(seconds=time_delta))

  def log_relative_time():
    tf.logging.info("Timing: %s", format_relative_time())

  return log_relative_time


def make_log_fn(epoch, log_relative_time_fn):

  def log(msg, *args):
    msg %= args
    tf.logging.info("%s Epoch %d: %s", ">>>>>>>", epoch, msg)
    log_relative_time_fn()

  return log


@contextlib.contextmanager
def temporary_flags(flag_settings):
  old_values = {}
  for flag_name, flag_value in flag_settings.items():
    old_values[flag_name] = getattr(FLAGS, flag_name)
    setattr(FLAGS, flag_name, flag_value)
  yield
  for flag_name, flag_value in old_values.items():
    setattr(FLAGS, flag_name, flag_value)


def _ppo_training_epochs(hparams, epoch, is_final_epoch, real_env_training):
  """Helper for PPO restarts."""
  if hparams.gather_ppo_real_env_data:
    assert hparams.real_ppo_epochs_num is 0, (
        "Should be put to 0 to enforce better readability")
    real_training_ppo_epochs_num = int(math.ceil(
        hparams.num_real_env_frames /
        (hparams.epochs*hparams.real_ppo_epoch_length)))
  else:
    real_training_ppo_epochs_num = hparams.real_ppo_epochs_num

  simulated_training_ppo_epochs_num = hparams.ppo_epochs_num

  if epoch == -1:
    assert real_env_training, (
        "Epoch -1 should only be used for PPO collection in real environment.")
    return real_training_ppo_epochs_num
  ppo_training_epochs = (epoch + 1) * (simulated_training_ppo_epochs_num
                                       + real_training_ppo_epochs_num)
  if is_final_epoch:  # Length of training in the final epoch is doubled.
    ppo_training_epochs += simulated_training_ppo_epochs_num
  if real_env_training:
    ppo_training_epochs += real_training_ppo_epochs_num
  return ppo_training_epochs


def train_agent(environment_spec, agent_model_dir,
                event_dir, world_model_dir, epoch_data_dir, hparams, epoch=0,
                is_final_epoch=False):
  """Train the PPO agent in the simulated environment."""
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents",
                      "optimization_epochs", "eval_every_epochs"]

  for param_name in ppo_params_names:
    ppo_param_name = "ppo_" + param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, False)
  ppo_hparams.save_models_every_epochs = 10
  ppo_hparams.world_model_dir = world_model_dir
  ppo_hparams.add_hparam("force_beginning_resets", True)

  # Adding model hparams for model specific adjustments
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  ppo_hparams.add_hparam("model_hparams", model_hparams)

  environment_spec = copy.copy(environment_spec)
  environment_spec_param_names = [
      "simulation_random_starts", "simulation_flip_first_random_for_beginning",
      "intrinsic_reward_scale"
  ]
  for param_name in environment_spec_param_names:
    environment_spec.set_hparam(param_name, hparams.get(param_name))
  ppo_hparams.add_hparam("environment_spec", environment_spec)

  ppo_hparams.add_hparam("initial_frame_chooser", InitialFrameChooser(
      environment_spec, mode=tf.estimator.ModeKeys.EVAL
  ))

  # TODO(koz4k): Pass by arguments.
  with temporary_flags({
      "problem": environment_spec.initial_frames_problem,
      "model": hparams.generative_model,
      "hparams_set": hparams.generative_model_params,
      "output_dir": world_model_dir,
      "data_dir": epoch_data_dir,
  }):
    rl_trainer_lib.train(ppo_hparams, event_dir + "sim", agent_model_dir,
                         name_scope="ppo_sim%d" % (epoch + 1))


def train_agent_real_env(
    env, agent_model_dir, event_dir, epoch_data_dir,
    hparams, epoch=0, is_final_epoch=False):
  """Train the PPO agent in the real environment."""
  # TODO: Implement
  ppo_hparams = trainer_lib.create_hparams(hparams.ppo_params)
  ppo_params_names = ["epochs_num", "epoch_length",
                      "learning_rate", "num_agents", "eval_every_epochs",
                      "optimization_epochs", "effective_num_agents"]

  # This should be overridden.
  ppo_hparams.add_hparam("effective_num_agents", None)
  for param_name in ppo_params_names:
    ppo_param_name = "real_ppo_"+ param_name
    if ppo_param_name in hparams:
      ppo_hparams.set_hparam(param_name, hparams.get(ppo_param_name))

  ppo_hparams.epochs_num = _ppo_training_epochs(hparams, epoch,
                                                is_final_epoch, True)
  # We do not save model, as that resets frames that we need at restarts.
  # But we need to save at the last step, so we set it very high.
  ppo_hparams.save_models_every_epochs = 1000000

  environment_spec = tf.contrib.training.HParams(batch_env=env,
                                                 wrappers=None,
                                                 simulated_env=False)

  ppo_hparams.add_hparam("environment_spec", environment_spec)

  rl_trainer_lib.train(ppo_hparams, event_dir + "real", agent_model_dir,
                       name_scope="ppo_real%d" % (epoch + 1))

  # Save unfinished rollouts to history.
  env.reset()


def train_world_model(env, data_dir, output_dir, hparams, epoch):
  """Train the world model on problem_name."""
  # TODO: Implement
  train_steps = hparams.model_train_steps * (
      epoch + hparams.inital_epoch_train_steps_multiplier)
  model_hparams = trainer_lib.create_hparams(hparams.generative_model_params)
  learning_rate = model_hparams.learning_rate_constant
  if epoch > 0: learning_rate *= hparams.learning_rate_bump


def setup_env(hparams):
  # TODO(kc): set reward clipping, when this will be possible
  assert hparams.game == 'pong', 'Currently only games with [-1, 1] reward ' \
                                 'range are working'
  game_mode = "Deterministic-v4"
  camel_game_name = "".join(
    [w[0].upper() + w[1:] for w in hparams.game.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name
  env = T2TGymEnv([gym.make(env_name)],
                  grayscale=hparams.grayscale,
                  resize_width_factor=hparams.resize_width_factor,
                  resize_height_factor=hparams.resize_height_factor)
  return env


def eval_unclipped_reward(env):
  # TODO: Implement, this should read data from env and aggregate (without
  # playing)
  pass


def training_loop(hparams, output_dir, report_fn=None, report_metric=None):
  """Run the main training loop."""
  # TODO: does anyone need this report_fn?
  if report_fn:
    assert report_metric is not None

  # Directories
  subdirectories = ["data", "tmp", "world_model", "ppo"]
  directories = setup_directories(output_dir, subdirectories)

  epoch = -1
  env = setup_env(hparams)
  env.start_new_epoch(epoch)

  # Timing log function
  log_relative_time = make_relative_timing_fn()

  # Per-epoch state
  epoch_metrics = []
  epoch_data_dirs = []

  data_dir = os.path.join(directories["data"], "initial")
  epoch_data_dirs.append(data_dir)
  # Collect data from the real environment with PPO or random policy.
  # TODO: do we need option not to gather_ppo_real_env_data?
  # We could set learning_rate=0 if this flag == False.
  assert hparams.gather_ppo_real_env_data
  ppo_model_dir = directories["ppo"]
  tf.logging.info("Initial training of PPO in real environment.")
  ppo_event_dir = os.path.join(directories["world_model"],
                               "ppo_summaries/initial")
  mean_reward = train_agent_real_env(
      env, ppo_model_dir,
      ppo_event_dir, data_dir,
      hparams, epoch=epoch, is_final_epoch=False)
  tf.logging.info("Mean reward (initial): {}".format(mean_reward))

  eval_metrics_event_dir = os.path.join(directories["world_model"],
                                        "eval_metrics_event_dir")
  eval_metrics_writer = tf.summary.FileWriter(eval_metrics_event_dir)

  mean_unclipped_reward_summary = tf.Summary()
  mean_unclipped_reward_summary.value.add(tag="mean_unclipped_reward",
                                          simple_value=None)
  mean_clipped_reward_summary = tf.Summary()
  mean_clipped_reward_summary.value.add(tag="mean_clipped_reward",
                                        simple_value=None)

  sim_env_spec = rl.standard_atari_env_simulated_spec(
      env,
      # Hardcoded for now.
      video_num_input_frames=4, video_num_target_frames=1
  )

  for epoch in range(hparams.epochs):
    env.start_new_epoch(epoch)
    is_final_epoch = (epoch + 1) == hparams.epochs
    log = make_log_fn(epoch, log_relative_time)

    epoch_data_dir = os.path.join(directories["data"], str(epoch))
    tf.gfile.MakeDirs(epoch_data_dir)
    env.generate_data(epoch_data_dir, directories['tmp'])
    epoch_data_dirs.append(epoch_data_dir)

    # Train world model
    log("Training world model")
    train_world_model(env, epoch_data_dir,
                      directories["world_model"], hparams, epoch)

    # Train PPO
    log("Training PPO in simulated environment.")
    ppo_event_dir = os.path.join(directories["world_model"],
                                 "ppo_summaries", str(epoch))
    ppo_model_dir = directories["ppo"]
    if not hparams.ppo_continue_training:
      ppo_model_dir = ppo_event_dir
    # TODO: build environment_spec (for simulated env)
    train_agent(sim_env_spec, ppo_model_dir,
                ppo_event_dir, directories["world_model"], epoch_data_dir,
                hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    # Train PPO on real env (short)
    log("Training PPO in real environment.")
    # TODO: pass env, return summaries?
    # TODO(kc): generation_mean_reward vs mean_reward (clipped?)
    mean_clipped_reward = train_agent_real_env(
        env, ppo_model_dir,
        ppo_event_dir, epoch_data_dir,
        hparams, epoch=epoch, is_final_epoch=is_final_epoch)

    if hparams.stop_loop_early:
      return 0.0

    log("Mean clipped reward during generation: {}".format(
        mean_clipped_reward))  # this was output of generate_real_env_data(...)

    mean_unclipped_reward = eval_unclipped_reward(env)
    log("Mean eval reward (unclipped): {}".format(mean_unclipped_reward))

    # Summarize metrics.
    mean_unclipped_reward_summary.value[0].simple_value = mean_unclipped_reward
    eval_metrics_writer.add_summary(mean_unclipped_reward_summary, epoch)
    mean_clipped_reward_summary.value[0].simple_value = int(mean_clipped_reward)
    eval_metrics_writer.add_summary(mean_clipped_reward_summary, epoch)
    eval_metrics_writer.flush()

    # Report metrics
    eval_metrics = {"mean_unclipped_reward": mean_unclipped_reward}
    epoch_metrics.append(eval_metrics)
    log("Eval metrics: %s", str(eval_metrics))
    if report_fn:
      report_fn(eval_metrics[report_metric], epoch)

  # Return the evaluation metrics from the final epoch
  return epoch_metrics[-1]


def main(_):
  hp = trainer_model_based_params.create_loop_hparams()
  assert not FLAGS.job_dir_to_evaluate
  training_loop(hp, FLAGS.output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
