# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""unit tests of embedding_lookup APIs
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute.cluster_resolver import cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import server_lib


def _get_sparse_variable(name,
                         key_dtype=dtypes.int64,
                         value_dtype=dtypes.float32,
                         dim=2,
                         init_size=16,
                         ps_devices=None,
                         bp_v2=False,
                         restrict_policy=None,
                         initializer=0.1,
                         distribute_strategy=None):
  devar = de.get_variable(name,
                          key_dtype=key_dtype,
                          value_dtype=value_dtype,
                          dim=dim,
                          init_size=init_size,
                          bp_v2=bp_v2,
                          devices=ps_devices,
                          restrict_policy=restrict_policy,
                          initializer=initializer)
  shadow_name = name + '-shadow'
  shadow = de.shadow_ops.ShadowVariable(devar,
                                        name=shadow_name,
                                        distribute_strategy=distribute_strategy)
  return devar, shadow


def _sort_keys_and_values(keys, values):
  seq = np.argsort(keys)
  keys = np.sort(keys)
  values = values[seq]
  return keys, values


def _create_ps_and_worker_servers(spec):
  ps_list, worker_list = [], []
  for job_name, ip_port_list in spec.as_dict().items():
    for i, v in enumerate(ip_port_list):
      node = server_lib.Server(spec,
                               job_name=job_name,
                               task_index=i,
                               config=default_cluster_config)
      if job_name == 'ps':
        ps_list.append(node)
      elif job_name == 'worker':
        worker_list.append(node)
      else:
        raise TypeError(
            'Expecting ps or worker in cluster_spec, but get {}'.format(
                job_name))
  return ps_list, worker_list


default_config = config_pb2.ConfigProto(
    allow_soft_placement=True,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

default_cluster_config = config_pb2.ConfigProto(allow_soft_placement=False)


@test_util.run_all_in_graph_and_eager_modes
class ShadowVariableTest(test.TestCase):

  def test_create(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    key_dtypes = [dtypes.int32, dtypes.int64]
    value_dtypes = [dtypes.int32, dtypes.float32, dtypes.float64]
    dims = [1, 4]
    trainable_options = [True, False]
    devices = ['/CPU:0']
    var_list = []
    rnd = 0
    for comb in itertools.product(key_dtypes, value_dtypes, dims,
                                  trainable_options):
      devar = de.get_variable('sparse_domain-' + str(rnd),
                              key_dtype=comb[0],
                              value_dtype=comb[1],
                              dim=comb[2],
                              initializer=0.1,
                              devices=devices,
                              init_size=1)
      name = 'shadow-' + str(rnd)
      var = de.shadow_ops.ShadowVariable(devar, name=name, trainable=comb[3])
      self.assertEqual(var.dtype, devar.value_dtype)
      self.assertEqual(var.ids.dtype, devar.key_dtype)
      rnd += 1

  def test_lookup(self):
    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('tk049', dim=2)
      self.evaluate(variables.global_variables_initializer())
      ids = constant_op.constant([2, 5], dtype=dtypes.int64)
      values = array_ops.ones((2, 2), dtype=np.float32)
      self.evaluate(
          var.upsert(ids, ops.convert_to_tensor(values, dtype=dtypes.float32)))

      ext_ids = constant_op.constant([2, 5, 8], dtype=dtypes.int64)
      exp_values = np.array([[1, 1], [1, 1], [0.1, 0.1]], dtype=np.float32)
      emb = self.evaluate(de.shadow_ops.embedding_lookup(shadow_var, ext_ids))
      self.assertAllEqual(exp_values, emb)

  def test_update_with_optimizer_v1(self):
    if not context.executing_eagerly():
      self.skipTest('Only test when eagerly.')

    for bp_v2 in [False, True]:
      var, shadow_var = _get_sparse_variable('bh890-bpv2-%s' % bp_v2,
                                             dim=2,
                                             bp_v2=bp_v2)
      optimizer = adam.AdamOptimizer(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      initialized = False
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        ids = []
        for i in range(10):
          ids.append(i)
          tf_ids = ops.convert_to_tensor(ids, dtype=dtypes.int64)

          def _loss_fn(shadow_var, ids):
            emb = de.shadow_ops.embedding_lookup(
                shadow_var, ops.convert_to_tensor(ids, dtype=dtypes.int64))
            loss = math_ops.reduce_mean(emb, axis=0)
            loss = array_ops.reshape(loss, (-1, 2))
            loss = math_ops.matmul(loss, array_ops.transpose(loss))
            return loss

          train_op = optimizer.minimize(lambda: _loss_fn(shadow_var, ids),
                                        var_list=[shadow_var])
          if not initialized:
            self.evaluate(variables.global_variables_initializer())
            initialized = True
          self.evaluate(train_op)
          keys, values = _sort_keys_and_values(*self.evaluate(var.export()))
          result_keys, result_values = _sort_keys_and_values(*self.evaluate(
              [shadow_var.ids, shadow_var.read_value(False)]))
          self.assertAllEqual(keys, result_keys)
          self.assertAllEqual(values, result_values)

  def test_update_with_optimizer_v2(self):
    if not context.executing_eagerly():
      self.skipTest('Only test when eagerly.')

    for bp_v2 in [False, True]:
      var, shadow_var = _get_sparse_variable('bh890-bpv2-%s' % bp_v2,
                                             dim=2,
                                             bp_v2=bp_v2)
      optimizer = optimizer_v2.adagrad.Adagrad(1E-3)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)
      initialized = False
      with self.session(use_gpu=test_util.is_gpu_available(),
                        config=default_config):
        ids = []
        for i in range(10):
          ids.append(i)
          tf_ids = ops.convert_to_tensor(ids, dtype=dtypes.int64)

          def _loss_fn(shadow_var, ids):
            emb = de.shadow_ops.embedding_lookup(
                shadow_var, ops.convert_to_tensor(ids, dtype=dtypes.int64))
            loss = math_ops.reduce_mean(emb, axis=0)
            loss = array_ops.reshape(loss, (-1, 2))
            loss = math_ops.matmul(loss, array_ops.transpose(loss))
            return loss

          train_op = optimizer.minimize(lambda: _loss_fn(shadow_var, ids),
                                        [shadow_var])
          if not initialized:
            self.evaluate(variables.global_variables_initializer())
            initialized = True
          self.evaluate(train_op)
          keys, values = _sort_keys_and_values(*self.evaluate(var.export()))
          result_keys, result_values = _sort_keys_and_values(*self.evaluate(
              [shadow_var.ids, shadow_var.read_value(False)]))
          self.assertAllEqual(keys, result_keys)
          self.assertAllEqual(values, result_values)

  def test_wrapper_tf_function(self):
    if not context.executing_eagerly():
      self.skipTest('Skip test tf.function in eager mode.')
    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable('pf988', dim=2)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      @def_function.function
      def compute_fn(var, ids):
        emb = de.shadow_ops.embedding_lookup(var, ids)
        return math_ops.reduce_mean(emb)

      start = 0
      size = 0
      for i in range(10):
        ids = math_ops.range(start, i + 1, dtype=dtypes.int64)
        start = math_ops.reduce_max(ids) + 1
        size += array_ops.size(ids)
        optimizer.minimize(lambda: compute_fn(shadow_var, ids), [shadow_var])
        self.assertAllEqual(var.size(), size)

  def test_training_with_restrict_policy(self):
    if not context.executing_eagerly():
      self.skipTest('Skip test tf.function in eager mode.')

    with self.session(use_gpu=test_util.is_gpu_available(),
                      config=default_config):
      var, shadow_var = _get_sparse_variable(
          'pf988', dim=2, restrict_policy=de.TimestampRestrictPolicy)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

      @def_function.function
      def compute_fn(var, ids):
        emb = de.shadow_ops.embedding_lookup(var, ids)
        return math_ops.reduce_mean(emb)

      start = 0
      size = 0
      for i in range(10):
        ids = math_ops.range(start, i + 1, dtype=dtypes.int64)
        start = math_ops.reduce_max(ids) + 1
        size += array_ops.size(ids)
        optimizer.minimize(lambda: compute_fn(shadow_var, ids), [shadow_var])
        self.assertAllEqual(var.size(), size)
        self.assertAllEqual(var.restrict_policy.status.size(), size)

  def test_training_with_distributed_strategy(self):
    # TODO(Lifann) Servers will be alive and thus make other test cases
    # across the cases failed. So this case is kept only for demonstration.
    self.skipTest('Only for demonstration.')

    if not context.executing_eagerly():
      self.skipTest('Only test in eager mode.')

    cluster_spec = tf.train.ClusterSpec({
        'ps': ['localhost:2220', 'localhost:2221'],
        'worker': ['localhost:2222', 'localhost:2223']
    })
    ps_list, worker_list = _create_ps_and_worker_servers(cluster_spec)

    resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
        cluster_spec)
    strategy = tf.distribute.experimental.ParameterServerStrategy(resolver)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)
    with strategy.scope() as scope:
      var = de.get_variable('pf988',
                            dim=2,
                            initializer=0.1,
                            devices=['/job:ps/task:0', '/job:ps/task:1'])
      shadow_var = de.shadow_ops.ShadowVariable(var,
                                                name='pf988-shadow',
                                                distribute_strategy=strategy)
      optimizer = optimizer_v2.adam.Adam(1E-4)
      optimizer = de.DynamicEmbeddingOptimizer(optimizer)

    def dist_dataset_fn():
      dataset_values = np.arange(0, 10, dtype=np.int64)
      fn = lambda x: tf.data.Dataset.from_tensor_slices(dataset_values).batch(
          4).repeat(None)
      return strategy.distribute_datasets_from_function(fn)

    dataset = coordinator.create_per_worker_dataset(dist_dataset_fn)

    @tf.function
    def step_fn(iterator):

      def replica_fn(ids):

        def loss_fn(ids):
          batch_size = tf.shape(ids)[0]
          emb = de.shadow_ops.embedding_lookup(shadow_var, ids)
          loss = tf.reduce_mean(emb)
          return loss

        optimizer.minimize(lambda: loss_fn(ids), [shadow_var])

      return strategy.run(replica_fn, args=(next(iterator),))

    iterator = iter(dataset)
    for i in range(5):
      coordinator.schedule(step_fn, args=(iterator,))
    coordinator.join()
    self.assertAllEqual(var.size(), 10)


if __name__ == '__main__':
  test.main()