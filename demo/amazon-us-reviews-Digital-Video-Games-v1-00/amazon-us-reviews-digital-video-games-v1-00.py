import json
import os
import time
import sys

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_recommenders_addons import dynamic_embedding as de
from tensorflow.python.ops import variables

from absl import flags
from absl import app

ENCODDING_SEGMENT_LENGTH = 1000000
NON_LETTER_OR_NUMBER_PATTERN = r'[^a-zA-Z0-9]'

np.random.seed(0)
batch_size = 32
mode = 'keras'
#mode = 'legacy'
embedding_size = 8


class _RawFeature(object):
  def __init__(self, dtype, category):
    if not isinstance(category, int):
      raise TypeError('category must be an integer.')
    self.category = category

  def encode(self, tensor): 
    raise NotImplementedError

  def match_category(self, tensor):
    min_code = self.category * ENCODDING_SEGMENT_LENGTH
    max_code = (self.category + 1) * ENCODDING_SEGMENT_LENGTH
    mask = tf.math.logical_and(
        tf.greater_equal(tensor, min_code),
        tf.less(tensor, max_code))
    return mask

class _StringFeature(_RawFeature):
  def __init__(self, dtype, category):
    super(_StringFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


class _TextFeature(_RawFeature):
  def __init__(self, dtype, category):
    super(_TextFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.strings.regex_replace(tensor, NON_LETTER_OR_NUMBER_PATTERN, ' ')
    tensor = tf.strings.split(tensor, sep=' ').to_tensor('')
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


class _IntegerFeature(_RawFeature):
  def __init__(self, dtype, category):
    super(_IntegerFeature, self).__init__(dtype, category)

  def encode(self, tensor):
    tensor = tf.as_string(tensor)
    tensor = tf.strings.to_hash_bucket_fast(tensor, ENCODDING_SEGMENT_LENGTH)
    tensor += ENCODDING_SEGMENT_LENGTH * self.category
    return tensor


FEATURE_AND_ENCODER = {
    'customer_id': _StringFeature(tf.string, 0),
    'helpful_votes': _IntegerFeature(tf.int32, 1),
    #'marketplace': _StringFeature(tf.string, 2),  # Not used. All marketplaces are 'US'
    'product_category': _StringFeature(tf.string, 3),
    'product_id': _StringFeature(tf.string, 4),
    'product_parent': _StringFeature(tf.string, 5),
    'product_title': _TextFeature(tf.string, 6),
    'review_body': _TextFeature(tf.string, 7),
    #'review_date': _DateFeature(tf.string, 8),  # Not used.
    'review_headline': _TextFeature(tf.string, 9),
    'review_id': _StringFeature(tf.string, 10),
    'star_rating': _IntegerFeature(tf.int32, 11),
    'total_votes': _IntegerFeature(tf.int32, 12),
    #'verified_purchase': _Label(tf.int64),
    #'vine': _Label(tf.int64),
}

FAETURES = ['customer_id', 'helpful_votes', 'marketplace', 'product_category',
            'product_id', 'product_parent', 'product_title', 'review_body',
            'review_date', 'review_headline', 'review_id', 'star_rating', 'total_votes']
LABEL = 'verified_purchase'


def encode_feature(data):
  collected_features = []
  for ft, encoder in FEATURE_AND_ENCODER.items():
    feature = encoder.encode(data[ft])
    feature = tf.reshape(feature, (-1,))
    collected_features.append(feature)
  collected_features = tf.concat(collected_features, 0)
  return collected_features


def get_labels(data):
  return data['verified_purchase']


def initialize_dataset(batch_size=1, split='train', shuffle_size=0):
  video_games_data = tfds.load('amazon_us_reviews/Digital_Video_Games_v1_00', split=split, as_supervised=False)
  video_games_data = video_games_data.batch(batch_size)
  if shuffle_size > 0:
    video_games_data.shuffle(shuffle_size)
  iterator = video_games_data.__iter__()
  return iterator


def input_fn(iterator):
  nested_input = iterator.get_next()
  data = nested_input['data']
  collected_features = encode_feature(data)
  labels = get_labels(data)
  return collected_features, labels


class VideoGameDNN(tf.keras.Model):
  def __init__(self, lr='0.001', batch_size=1, embedding_size=1, nn=[256, 128, 32, 1], activation='relu'):
    super(VideoGameDNN, self).__init__()
    self.batch_size = batch_size
    self.lr = lr
    self.embedding_size = embedding_size
    self.embedding_store = de.get_variable('video_feature_embedding',
                                           key_dtype=tf.int64,
                                           value_dtype=tf.float32,
                                           dim=embedding_size,
                                           devices=['/CPU:0'],
                                           initializer=tf.keras.initializers.RandomNormal(-1, 1),
                                           trainable=True)
    self.activation = activation
    self.dnn0 = tf.keras.layers.Dense(1024, activation=self.activation, use_bias=True, bias_initializer='glorot_uniform')
    self.dnn1 = tf.keras.layers.Dense(256, activation=self.activation, use_bias=True, bias_initializer='glorot_uniform')
    self.dnn2 = tf.keras.layers.Dense(1, use_bias=False)
    self.embedding_trainables = []

  def embedding(self, x):
    embed_w, trainable = de.embedding_lookup(self.embedding_store, x, return_trainable=True, name='e1')
    self.embedding_trainables.clear()
    self.embedding_trainables.append(trainable)
    embed = []
    for name, encoder in FEATURE_AND_ENCODER.items():
      mask = encoder.match_category(x)
      indices = tf.where(mask)
      categorical_w = tf.gather(embed_w, indices)
      categorical_w = tf.reshape(categorical_w, (self.batch_size, -1, self.embedding_size))
      categorical_w = tf.reduce_sum(categorical_w, axis=1)
      embed.append(tf.reshape(categorical_w, (self.batch_size, self.embedding_size)))
    embed = tf.concat(embed, axis=1)  # with shape (batch_size, N, embedding_size)
    return embed

  def dnn_net(self, x):
    out = x
    out = self.dnn0(x)
    out = self.dnn1(out)
    out = self.dnn2(out)
    return out

  def call(self, x):
    logits = self.dnn_net(x)
    predict = tf.nn.sigmoid(logits)
    return predict


def build(model, optimizer):
  model.build((model.batch_size, len(FEATURE_AND_ENCODER) * embedding_size))
  if isinstance(optimizer, tf.compat.v1.train.Optimizer):
    return
  dummy_x = tf.zeros((4096), dtype=tf.int64)
  dummy_y = tf.zeros((model.batch_size,), dtype=tf.float32)
  def dummy_loss_fn():
    embed = model.embedding(dummy_x)
    z = model(embed)
    loss = tf.keras.losses.MeanSquaredError()(dummy_y, z)
    return loss
  with tf.GradientTape() as tape:
    dummy_loss = dummy_loss_fn()
    grads = tape.gradient(dummy_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train_step(model, optimizer, iterator):
  def _loss_fn():
    features, labels = input_fn(iterator)
    labels = tf.cast(labels, dtype=tf.float32)
    embed = model.embedding(features)
    predictions = model(embed)
    loss = tf.keras.losses.MeanSquaredError()(labels, predictions)
    return loss
  def _var_fn():
    return model.trainable_variables
  if mode == 'keras':
    with tf.GradientTape() as tape:
      loss = _loss_fn()
      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
  else:
    optimizer.minimize(_loss_fn)


def test_step(model, iterator):
  features, labels = input_fn(iterator)
  labels = tf.cast(labels, dtype=tf.float32)
  embed = model.embedding(features)
  predictions = model(embed)
  loss = tf.keras.losses.MeanSquaredError()(labels, predictions)
  return predictions, labels, loss


if __name__ == '__main__':
  iterator = initialize_dataset(batch_size=batch_size)
  if mode == 'keras':
    optmz = tf.keras.optimizers.Adam(0.001)
  else:
    optmz = tf.compat.v1.train.AdamOptimizer(0.001)
  optmz = de.DynamicEmbeddingOptimizer(optmz)

  model = VideoGameDNN(batch_size=batch_size, embedding_size=embedding_size, nn=[256, 32, 1])
  build(model, optmz)

  print('------- start training ------')

  for step in range(100):
    train_step(model, optmz, iterator)
    if step % 10 == 0:
      pred, labels, loss = test_step(model, iterator)
      print('step: {}, loss: {}'.format(step))
      print('emb size: ', model.embedding_store.size())
      print('')

  # TODO(Lifann) Support save TrainableWrapper or instead of saving Variable.lookup.
  # options = tf.saved_model.SaveOptions(namespace_whitelist=['TFRA'])
  # model.save('export', options=options, signatures=infer)
