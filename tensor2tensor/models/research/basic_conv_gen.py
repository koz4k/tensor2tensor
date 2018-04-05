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

"""Basic models for testing simple tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf


@registry.register_model
class BasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    # filters = self.hparams.hidden_size
    # num_hidden_layers = self.hparams.num_hidden_layers

    kernel_sizes = self.hparams.kernel_sizes
    filter_numbers = self.hparams.filter_numbers

    #TODO: possibly make embeding of inputs_0 and inputs_1
    cur_frame = features["inputs_0"]
    prev_frame = features["inputs_1"]
    action_embedding_size = 32
    action_space_size = 10
    kernel = (3, 3)
    # Gather all inputs.
    action = common_layers.embedding(tf.to_int64(features["action"]),
                                     action_space_size, action_embedding_size)
    action = tf.reshape(action, [-1, 1, 1, action_embedding_size])
    #broadcast to the shape compatibile with pictures
    action += tf.expand_dims(tf.zeros_like(cur_frame[..., 0]), -1)
    x = tf.concat([cur_frame, prev_frame, action], axis=3)
    for kernel_size, filters_number in zip(kernel_sizes, filter_numbers):
      x = tf.layers.conv2d(x, filters_number, activation=tf.nn.relu,
                           kernel_size=kernel_size, padding="SAME")

    # Up-convolve.
    # x = tf.layers.conv2d_transpose(
    #     frames, filters, kernel, activation=tf.nn.relu,
    #     strides=(1, 1), padding="SAME")
    # Output size is 3 * 256 for 3-channel color space.
    res = tf.layers.conv2d(x, 3 * 256, kernel, padding="SAME")
    x = tf.layers.flatten(x)

    # TODO: pm->pm: add done
    res_done = tf.layers.dense(x, 2)

    return {"targets":res, "reward": x}


@registry.register_hparams
def basic_conv_small():
  """Small conv model."""
  hparams = common_hparams.basic_params1()
  hparams.kernel_sizes = [(3,3), (5,5)]
  hparams.filter_numbers = [32, 3*256]
  hparams.batch_size = 2
  return hparams



@registry.register_model
class KanapaBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    filters = self.hparams.hidden_size
    cur_frame = features["inputs_0"] / 255.0  # TODO - mniej na pałę.
    prev_frame = features["inputs_1"] / 255.0
    with tf.variable_scope("kanapa_model", reuse=tf.AUTO_REUSE):
      action = common_layers.embedding(tf.to_int64(features["action"]),
                                       10, filters)
      action = tf.reshape(action, [-1, 1, 1, filters])

      frames = tf.concat([cur_frame, prev_frame], axis=3)
      h1 = tf.layers.conv2d(frames, filters, kernel_size=(3, 3), padding="SAME")
      h2 = tf.layers.conv2d(tf.nn.relu(h1 + action), filters,
                            kernel_size=(5, 5), padding="SAME")
      res = tf.layers.conv2d(tf.nn.relu(h2 + action), 3 * 256,
                             kernel_size=(3, 3), padding="SAME")
      reward_pred_h1 = tf.reduce_mean(res, axis=[1, 2])
      reward_pred = tf.layers.dense(reward_pred_h1, 2, name="reward")
    # reward_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #   labels=tf.to_int32(features["reward"]), logits=reward_pred)
    # reward_loss = tf.reduce_mean(reward_loss)
    x = tf.layers.flatten(h2)
    # l = tf.shape(res)[1]
    # w = tf.shape(res)[2]
    l = 210
    w = 160
    res = tf.reshape(res, [-1, l, w, 768])
    return {"targets":res, "reward": x}

@registry.register_model
class ResidualBasicConvGen(t2t_model.T2TModel):

  def body(self, features):
    filters = 38
    num_hidden_layers = self.hparams.num_hidden_layers
    #TODO: possibly make embeding of inputs_0 and inputs_1
    cur_frame = features["inputs_0"] / 255.0  # TODO - mniej na pałę.
    prev_frame = features["inputs_1"] / 255.0
    action_embedding_size = 32
    action_space_size = 10
    kernel = (3, 3)
    # Gather all inputs.
    with tf.variable_scope("residual_model", reuse=tf.AUTO_REUSE):
      action = common_layers.embedding(tf.to_int64(features["action"]),
                                       action_space_size, action_embedding_size)
      action = tf.reshape(action, [-1, 1, 1, action_embedding_size])
      #broadcast to the shape compatibile with pictures
      action += tf.expand_dims(tf.zeros_like(cur_frame[..., 0]), -1)
      frames = tf.concat([cur_frame, prev_frame, action], axis=3)
      # x = tf.layers.conv2d(frames, filters, kernel, activation=tf.nn.relu,
      #                      strides=(2, 2), padding="SAME")
      # Run a stack of convolutions.
      x = frames
      for _ in range(num_hidden_layers):
        y = tf.layers.conv2d(x, filters, kernel, activation=tf.nn.relu,
                             strides=(1, 1), padding="SAME")
        x = common_layers.layer_norm(x + y)
      # Up-convolve.
      # x = tf.layers.conv2d_transpose(
      #     frames, filters, kernel, activation=tf.nn.relu,
      #     strides=(1, 1), padding="SAME")
      # Output size is 3 * 256 for 3-channel color space.
      res = tf.layers.conv2d(x, 3 * 256, kernel, padding="SAME")
      x = tf.layers.flatten(x)

      # TODO: pm->pm: add done
      res_done = tf.layers.dense(x, 2)

    return {"targets":res, "reward": x}




