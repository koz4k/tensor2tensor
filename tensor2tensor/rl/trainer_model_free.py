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

r"""Training of RL agent with PPO algorithm.

Example invocation:

python -m tensor2tensor.rl.trainer_model_free \
    --output_dir=$HOME/t2t/rl_v1 \
    --hparams_set=pong_model_free \
    --loop_hparams='batch_size=15'
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

import numpy as np
import six

from tensor2tensor.data_generators import gym_env
from tensor2tensor.models.research import rl
from tensor2tensor.rl.ppo_learner import PPOLearner
from tensor2tensor.utils import flags as t2t_flags  # pylint: disable=unused-import
from tensor2tensor.utils import trainer_lib

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erring. Apologies for the ugliness.
try:
  flags.DEFINE_string("output_dir", "", "Base output directory for run.")
except:  # pylint: disable=bare-except
  pass


LEARNERS = {
    "ppo": PPOLearner
}


def setup_env(hparams, batch_size, max_num_noops):
  """Setup."""
  game_mode = "Deterministic-v4"
  camel_game_name = "".join(
      [w[0].upper() + w[1:] for w in hparams.game.split("_")])
  camel_game_name += game_mode
  env_name = camel_game_name

  env = gym_env.T2TGymEnv(base_env_name=env_name,
                          batch_size=batch_size,
                          grayscale=hparams.grayscale,
                          resize_width_factor=hparams.resize_width_factor,
                          resize_height_factor=hparams.resize_height_factor,
                          base_env_timesteps_limit=hparams.env_timesteps_limit,
                          max_num_noops=max_num_noops)
  return env


def update_hparams_from_hparams(target_hparams, source_hparams, prefix):
  """Copy a subset of hparams to target_hparams."""
  for (param_name, param_value) in six.iteritems(source_hparams.values()):
    if param_name.startswith(prefix):
      target_hparams.set_hparam(param_name[len(prefix):], param_value)


def initialize_env_specs(hparams):
  """Initializes env_specs using T2TGymEnvs."""
  if getattr(hparams, "game", None):
    game_name = gym_env.camel_case_name(hparams.game)
    if hparams.batch_size > 0:
      env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                              batch_size=hparams.batch_size)
      env.start_new_epoch(0)
      train_env_fn = rl.make_real_env_fn(env)
    else:
      train_env_fn = None
    hparams.add_hparam("env_fn", train_env_fn)

    eval_env = gym_env.T2TGymEnv("{}Deterministic-v4".format(game_name),
                                 batch_size=hparams.eval_batch_size)
    eval_env.start_new_epoch(0)
    hparams.add_hparam("eval_env_fn", rl.make_real_env_fn(eval_env))
  return hparams


def evaluate_single_config(hparams, stochastic, max_num_noops, agent_model_dir):
  """Evaluate the PPO agent in the real environment."""
  eval_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  env = setup_env(
      hparams, batch_size=hparams.eval_batch_size, max_num_noops=max_num_noops
  )
  env.start_new_epoch(0)
  env_fn = rl.make_real_env_fn(env)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, base_event_dir=None,
      agent_model_dir=agent_model_dir
  )
  learner.evaluate(env_fn, eval_hparams, stochastic)
  rollouts = env.current_epoch_rollouts()
  env.close()

  return tuple(
      compute_mean_reward(rollouts, clipped) for clipped in (True, False)
  )


def get_metric_name(x, stochastic, max_num_noops, clipped):
  return "{}_reward/eval/stochastic_{}_max_noops_{}_{}".format(
      x, stochastic, max_num_noops, "clipped" if clipped else "unclipped")


def evaluate_all_configs(hparams, agent_model_dir):
  """Evaluate the agent with multiple eval configurations."""
  metrics = {}
  # Iterate over all combinations of picking actions by sampling/mode and
  # whether to do initial no-ops.
  #for stochastic in (True, False):
  for max_num_noops in (hparams.eval_max_num_noops, 0):
    scores = evaluate_single_config(
        hparams, True, max_num_noops, agent_model_dir
    )
    for (score, clipped) in zip(scores, (True, False)):
      metric_name = get_metric_name("mean", True, max_num_noops, clipped)
      metrics[metric_name] = score[0]
      metric_name = get_metric_name("std", True, max_num_noops, clipped)
      metrics[metric_name] = score[1]
      tf.logging.info("Score for %s: %s", metric_name, str(score))

  return metrics


def compute_mean_reward(rollouts, clipped):
  """Calculate mean rewards from given epoch."""
  reward_name = "reward" if clipped else "unclipped_reward"
  rewards = []
  for rollout in rollouts:
    if rollout[-1].done:
      rollout_reward = sum(getattr(frame, reward_name) for frame in rollout)
      rewards.append(rollout_reward)
  if rewards:
    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)
  else:
    mean_rewards = 0
    std_rewards = 0
  return (mean_rewards, std_rewards)


def summarize_metrics(eval_metrics_writer, metrics, epoch):
  """Write metrics to summary."""
  for (name, value) in six.iteritems(metrics):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    eval_metrics_writer.add_summary(summary, epoch)
  eval_metrics_writer.flush()


def train(hparams, output_dir, report_fn=None):
  hparams = initialize_env_specs(hparams)
  learner = LEARNERS[hparams.base_algo](
      hparams.frame_stack_size, output_dir, output_dir
  )
  policy_hparams = trainer_lib.create_hparams(hparams.base_algo_params)
  update_hparams_from_hparams(
      policy_hparams, hparams, hparams.base_algo + "_"
  )
  eval_metrics_writer = tf.summary.FileWriter("eval_metrics")
  if hparams.env_fn is not None:
    learner.train(
        hparams.env_fn, policy_hparams, simulated=False, save_continuously=True,
        epoch=0, eval_env_fn=hparams.eval_env_fn, report_fn=report_fn
    )
  eval_metrics = evaluate_all_configs(hparams, output_dir)
  summarize_metrics(eval_metrics_writer, eval_metrics, 0)
  tf.logging.info(
      "Agent eval metrics:\n{}".format(pprint.pformat(eval_metrics))
  )
  with open("metrics", "w") as f:
    f.write(str(eval_metrics))


def main(_):
  hparams = trainer_lib.create_hparams(FLAGS.hparams_set, FLAGS.hparams)
  train(hparams, FLAGS.output_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
