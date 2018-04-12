from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:14:05 2018

@author: vinhn
"""

#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for DNN-based text classification with DBpedia data."""


import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
import pdb

FLAGS = None

MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 64
n_words = None
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.

_BASE_LR=0.1
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    (1.0, 3), (0.1, 10), (0.01, 20), (0.001, 30),
]

def learning_rate_schedule(current_epoch):
  """Handles linear scaling rule, gradual warmup, and LR decay.

  The learning rate starts at 0, then it increases linearly per gradient.
  After 5 epochs we reach the base learning rate.
  After 30, 60 and 80 epochs the learning rate is divided by 10.
  After 90 epochs training stops and the LR is set to 0.

  Args:
    current_epoch: the epoch that we are processing now.
  Returns:
    The learning rate for the current epoch.
  """
  scaled_lr = _BASE_LR

  decay_rate = scaled_lr * _LR_SCHEDULE[0][0] * current_epoch / _LR_SCHEDULE[0][1]  # pylint: disable=protected-access,line-too-long
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate


def estimator_spec_for_softmax_classification(logits, labels, mode):
  """Returns EstimatorSpec instance for softmax classification."""
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  global_step = tf.train.get_global_step()
  current_epoch = (tf.cast(global_step, tf.float32) / 560000/1024)
  #learning_rate = learning_rate_schedule(current_epoch)
  learning_rate = 0.01
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      'accuracy':
          tf.metrics.accuracy(labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def bag_of_words_model(features, labels, mode):
  """A bag-of-words model. Note it disregards the word order in the text."""
  bow_column = tf.feature_column.categorical_column_with_identity(
      WORDS_FEATURE, num_buckets=n_words)
  bow_embedding_column = tf.feature_column.embedding_column(
      bow_column, dimension=EMBEDDING_SIZE)
  bow = tf.feature_column.input_layer(
      features, feature_columns=[bow_embedding_column])
  logits = tf.layers.dense(bow, MAX_LABEL, activation=None)

  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def rnn_model(features, labels, mode):
  """RNN model to predict from sequence of words to a class."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = tf.contrib.layers.embed_sequence(
      features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)

  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unstack(word_vectors, axis=1)

  # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
  #cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
  #cell = tf.nn.rnn_cell.LSTMCell(EMBEDDING_SIZE)
  cell = tf.nn.rnn_cell.BasicLSTMCell(EMBEDDING_SIZE, state_is_tuple=False)

  # Create an unrolled Recurrent Neural Networks to length of
  # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
  _, encoding = tf.nn.static_rnn(cell, word_list, dtype=tf.float32)

  # Given encoding of RNN, take encoding of last step (e.g hidden size of the
  # neural network of last step) and pass it as features for softmax
  # classification over output classes.
  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)
  return estimator_spec_for_softmax_classification(
      logits=logits, labels=labels, mode=mode)


def main(unused_argv):
  global n_words
  tf.logging.set_verbosity(tf.logging.INFO)

  # Prepare training and testing data
  dbpedia = tf.contrib.learn.datasets.load_dataset(
      'dbpedia', size='large', test_with_fake_data=FLAGS.test_with_fake_data)
  
  print("Shuffling data set...")
  x_train = dbpedia.train.data[:, 1]
  y_train = dbpedia.train.target
  s = np.arange(len(y_train))
  np.random.shuffle(s)
  x_train = x_train[s]
  y_train = y_train[s]
  print("Done!")  
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(dbpedia.test.data[:, 1])
  y_test = pandas.Series(dbpedia.test.target)

  print('Train data size:', x_train.shape)
  print('Test data size:', x_test.shape)
  # Process vocabulary
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)

  # Build model
  # Switch between rnn_model and bag_of_words_model to test different models.
  model_fn = rnn_model
  if FLAGS.bow_model:
    # Subtract 1 because VocabularyProcessor outputs a word-id matrix where word
    # ids start from 1 and 0 means 'no word'. But
    # categorical_column_with_identity assumes 0-based count and uses -1 for
    # missing word.
    x_train -= 1
    x_test -= 1
    model_fn = bag_of_words_model
  classifier = tf.estimator.Estimator(model_fn=model_fn, model_dir='./text_results')

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=1024,
      queue_capacity=10000,
      num_epochs=None,
      shuffle=True)

  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
      
  for e in range(20):    
      classifier.train(input_fn=train_input_fn, steps=1000)
      # Score with tensorflow.
      scores = classifier.evaluate(input_fn=test_input_fn)
      print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

  # Predict.

  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)



  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={WORDS_FEATURE: x_train},
      y=y_train,
      batch_size=1024,
      num_epochs=1,
      shuffle=False)
  scores = classifier.evaluate(input_fn=train_input_fn)
  print('Train Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
  print('Train Loss (tensorflow): {0:f}'.format(scores['loss']))
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--test_with_fake_data',
      default=False,
      help='Test the example code with fake data.',
      action='store_true')
  parser.add_argument(
      '--bow_model',
      default=False,
      help='Run with BOW model instead of RNN.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
  
  
  #vim /home/vinhngx/.local/lib/python2.7/site-packages/tensorflow/contrib/learn/python/learn/datasets/text_datasets.py
