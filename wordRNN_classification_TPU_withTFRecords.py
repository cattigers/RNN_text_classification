from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
import tensorflow as tf
import os

from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.estimator import estimator

FLAGS = flags.FLAGS
flags.DEFINE_bool(
    'use_tpu', True,
    help=('Use TPU to execute the model for training and evaluation. If'
          ' --use_tpu=false, will use whatever devices are available to'
          ' TensorFlow by default (e.g. CPU and GPU)'))

# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

flags.DEFINE_string(
    'tpu_name', default=None,
    help='Name of the Cloud TPU for Cluster Resolvers. You must specify either '
    'this flag or --master.')

flags.DEFINE_string(
    'master', default=None,
    help='gRPC URL of the master (i.e. grpc://ip.address.of.tpu:8470). You '
    'must specify either this flag or --tpu_name.')

flags.DEFINE_bool(
    'test_with_fake_data', default=False,
    help='use fake test data')

flags.DEFINE_string(
    'model_dir', default=None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))

tf.flags.DEFINE_string(
    'data_dir', default='.',
    help='The directory where the ImageNet input data is stored.')


flags.DEFINE_bool(
    'skip_host_call', default=False,
    help=('Skip the host_call which is executed every training step. This is'
          ' generally used for generating training summaries (train loss,'
          ' learning rate, etc...). When --skip_host_call=false, there could'
          ' be a performance drop if host_call function is slow and cannot'
          ' keep up with the TPU-side computation.'))

flags.DEFINE_integer(
    'iterations_per_loop', default=100,
    help=('Number of steps to run on TPU before outfeeding metrics to the CPU.'
          ' If the number of iterations in the loop would exceed the number of'
          ' train steps, the loop will exit before reaching'
          ' --iterations_per_loop. The larger this value is, the higher the'
          ' utilization on the TPU.'))

flags.DEFINE_integer(
    'num_cores', default=8,
    help=('Number of TPU cores. For a single TPU device, this is 8 because each'
          ' TPU has 4 chips each with 2 cores.'))
         
flags.DEFINE_string(
    'data_format',
    default='channels_last',
    help=('A flag to override the data format used in the model. The value '
          'is either channels_first or channels_last. To run the network on '
          'CPU or TPU, channels_last should be used.'))

flags.DEFINE_integer(
    'train_batch_size', default=1024, help='Batch size for training.')

flags.DEFINE_integer(
    'eval_batch_size', default=1024, help='Batch size for evaluation.')

flags.DEFINE_integer(
    'steps_per_eval', default=1000,
    help=('Controls how often evaluation is performed. Since evaluation is'
          ' fairly expensive, it is advised to evaluate as infrequently as'
          ' possible (i.e. up to --train_steps, which evaluates the model only'
          ' after finishing the entire training regime).'))

flags.DEFINE_integer(
    'train_steps', default=54687,
    help=('The number of steps to use for training. Default is 54687 steps'
          ' which is approximately 100 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))

tf.flags.DEFINE_float(
    'learning_rate', default=0.1,
    help=('base learning assuming a batch size of 1024.'
          'For other batch sizes it is scaled linearly with batch size.'))

flags.DEFINE_integer(
    'rnn_size', default=256,
    help=('Number of hidden units in RNN'))
    
# For training data infeed parallelism.
flags.DEFINE_integer(
    'num_files_infeed',
    default=8,
    help='The number of training files to read in parallel.')

flags.DEFINE_integer(
    'num_parallel_calls',
    default=8,
    help='Number of threads to use for transforming images.')

flags.DEFINE_integer(
    'prefetch_buffer_size',
    default=100 * 1000 * 1000,
    help='Prefetch buffer for each file, in bytes.')    

flags.DEFINE_integer('shuffle_buffer_size', 1000,
                        'Size of the shuffle buffer used to randomize ordering')

flags.DEFINE_integer(
    'max_document_length', default=200,
    help=('The number of steps to use for training. Default is 54687 steps'
          ' which is approximately 100 epochs at batch size 1024. This flag'
          ' should be adjusted according to the --train_batch_size flag.'))
                        
_NUM_TRAIN_IMAGES = 560000
_NUM_EVAL_IMAGES = 70000

MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 64
n_words = None
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.

# Learning hyperaparmeters
_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4
_LR_SCHEDULE = [  # (LR multiplier, epoch to start)
    (1.0, 5), (0.1, 15), (0.01, 25), (0.001, 30), 
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
  scaled_lr = FLAGS.learning_rate * (FLAGS.train_batch_size / 1024.0)

  decay_rate = scaled_lr * _LR_SCHEDULE[0][0] * current_epoch / _LR_SCHEDULE[0][1]  # pylint: disable=protected-access,line-too-long
  for mult, start_epoch in _LR_SCHEDULE:
    decay_rate = tf.where(current_epoch < start_epoch,
                          decay_rate, scaled_lr * mult)
  return decay_rate
  
class DBPediaInput(object):
  """Wrapper class that acts as the input_fn to TPUEstimator."""

  def __init__(self, is_training, data_dir=None):
    self.is_training = is_training
    self.data_dir = data_dir if data_dir else FLAGS.data_dir

  def dataset_parser(self, value):
    """Parse an Imagenet record from value."""
    keys_to_features = {
        'X': tf.FixedLenFeature(shape=[MAX_DOCUMENT_LENGTH], dtype=tf.int64),
        'Y': tf.FixedLenFeature(shape=[1], dtype=tf.int64)            
    }
    parsed = tf.parse_single_example(value, keys_to_features)
    X = tf.cast(parsed['X'], tf.int32)
    Y = tf.cast(parsed['Y'], tf.int32)
    return X, Y

  def __call__(self, params):
    """Input function which provides a single batch for train or eval."""
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    batch_size = params['batch_size']

    # Shuffle the filenames to ensure better randomization
    file_pattern = os.path.join(
        self.data_dir, 'train*' if self.is_training else 'test*')
    dataset = tf.data.Dataset.list_files(file_pattern)
    if self.is_training:
      dataset = dataset.shuffle(buffer_size=1024)  # 1024 files in dataset

    if self.is_training:
      dataset = dataset.repeat()

    def prefetch_dataset(filename):
      buffer_size =  FLAGS.prefetch_buffer_size
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    dataset = dataset.apply(
        tf.contrib.data.parallel_interleave(
            prefetch_dataset, cycle_length= FLAGS.num_files_infeed,
            sloppy=True))
    dataset = dataset.shuffle(FLAGS.shuffle_buffer_size)

    dataset = dataset.map(
        self.dataset_parser,
        num_parallel_calls=FLAGS.num_parallel_calls)
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))

    dataset = dataset.prefetch(2)  # Prefetch overlaps in-feed with training
    images, labels = dataset.make_one_shot_iterator().get_next()
    return images, labels

  
def char_rnn_model(features, labels, mode, params):
  """Character level recurrent neural network model to predict classes."""
  batch_size = params['batch_size']
  
  #byte_vectors = tf.squeeze(tf.one_hot(features, 256, 1., 0.))
  #byte_list = tf.unstack(byte_vectors, num=MAX_DOCUMENT_LENGTH, axis=1)
  #for item in byte_list:
  #    item.set_shape((params['batch_size'], 256))

  byte_vectors = tf.one_hot(features, 256, 1., 0.)
  byte_list = tf.unstack(byte_vectors, axis=1)
  
  
  cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
  _, encoding = tf.nn.static_rnn(cell, byte_list, dtype=tf.float32)

  logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        predictions={
            'class': predicted_classes,
            'prob': tf.nn.softmax(logits)
        })

  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) +\
          _WEIGHT_DECAY * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()
          if 'batch_normalization' not in v.name])
  
  #get current training epoch
  batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
  global_step = tf.train.get_global_step()
  current_epoch = (tf.cast(global_step, tf.float32)/batches_per_epoch)
  learning_rate = learning_rate_schedule(current_epoch)
  
  if mode == tf.estimator.ModeKeys.TRAIN:
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=_MOMENTUM, use_nesterov=True)
    if FLAGS.use_tpu:
      # When using TPU, wrap the optimizer with CrossShardOptimizer which
      # handles synchronization details between different TPU cores. To the
      # user, this should look like regular synchronous training.
      optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tpu_estimator.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)

  #trick to report Learning rate as a metric: repeat batch_size time
  lr_repeat = tf.reshape(
      tf.tile(tf.expand_dims(learning_rate, 0), [
          batch_size,
      ]), [batch_size, 1])
      
  ce_repeat = tf.reshape(
      tf.tile(tf.expand_dims(current_epoch, 0), [
          batch_size,
      ]), [batch_size, 1])      
  eval_metrics = None
  if mode == tf.estimator.ModeKeys.EVAL:
    def metric_fn(labels, logits, lr_repeat, ce_repeat):
      """Evaluation metric fn. Performed on CPU, do not reference TPU ops."""      
      
      predicted_classes = tf.argmax(logits, 1)
      return {
          'accuracy': tf.metrics.accuracy(
                                  labels=labels, predictions=predicted_classes),
          'learning_rate': tf.metrics.mean(lr_repeat),
          'current_epoch': tf.metrics.mean(ce_repeat)
          }

    eval_metrics = (metric_fn, [labels, logits, lr_repeat, ce_repeat])
    #eval_metrics= (lambda x,y: {'dummy':tf.metrics.accuracy(x,y)},[labels,predicted_classes])
    #eval_metrics=(lambda x:{'accuracy': tf.metrics.mean(x)},[labels])
    
  return tpu_estimator.TPUEstimatorSpec(
      mode=mode, loss=loss, eval_metrics=eval_metrics)


import pdb  
def main(unused_argv):
  global HIDDEN_SIZE
  global MAX_DOCUMENT_LENGTH
  HIDDEN_SIZE = FLAGS.rnn_size
  MAX_DOCUMENT_LENGTH = FLAGS.max_document_length
  
  tpu_grpc_url = None
  tpu_cluster_resolver = None
  if FLAGS.use_tpu:
    # Determine the gRPC URL of the TPU device to use
    #if not FLAGS.master and not FLAGS.tpu_name:
    #  raise RuntimeError('You must specify either --master or --tpu_name.')

    if FLAGS.master:
      if FLAGS.tpu_name:
        tf.logging.warn('Both --master and --tpu_name are set. Ignoring'
                        ' --tpu_name and using --master.')
      tpu_grpc_url = FLAGS.master
    else:
      tpu_cluster_resolver = (
          tf.contrib.cluster_resolver.TPUClusterResolver(
              FLAGS.tpu_name,
              zone=FLAGS.tpu_zone,
              project=FLAGS.gcp_project))
  else:
    # URL is unused if running locally without TPU
    tpu_grpc_url = None
 
  config = tpu_config.RunConfig(
      master=tpu_grpc_url,
      evaluation_master=tpu_grpc_url,
      model_dir=FLAGS.model_dir,
      #cluster=tpu_cluster_resolver,
      tpu_config=tpu_config.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_cores))
  batches_per_epoch = _NUM_TRAIN_IMAGES / FLAGS.train_batch_size
  #pdb.set_trace()
  # Build model
  #classifier = tf.estimator.Estimator(model_fn=char_rnn_model)
  classifier = tpu_estimator.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=char_rnn_model,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      params={'batches_per_epoch': batches_per_epoch})

  current_step = 0
  # Train.
  current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)  # pylint: disable=protected-access,line-too-long
  tf.logging.info('Training for %d steps (%.2f epochs in total). Current '
                    'step %d' % (FLAGS.train_steps,
                                 FLAGS.train_steps / batches_per_epoch,
                                 current_step))

  if current_step > 0:                  
      scores = classifier.evaluate(input_fn=DBPediaInput(True), steps=10)
      print('Accuracy: {0:f}'.format(scores['accuracy']))


  while current_step < FLAGS.train_steps:
    # Train for up to steps_per_eval number of steps.
    # At the end of training, a checkpoint will be written to --model_dir.
    next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                      FLAGS.train_steps)  

    classifier.train(
        input_fn=DBPediaInput(True), max_steps=next_checkpoint)
    current_step = next_checkpoint
    
    # Eval.
    tf.logging.info('Starting to evaluate.')
    eval_results = classifier.evaluate(
        input_fn=DBPediaInput(False), 
        steps=10)
    tf.logging.info('Test eval results: %s' % eval_results)
    
  scores = classifier.evaluate(input_fn=DBPediaInput(False), steps=70)
  print('Accuracy: {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

  #run CMD: 
  #python text_classification_TPU_withTFRecords.py --use_tpu=False --model_dir=./char_results_TFRecords --train_batch_size=128  #converge to 97.57% accuracy 
  #export CUDA_VISIBLE_DEVICES=2
  #python text_classification_TPU_withTFRecords.py \--use_tpu=False --model_dir=./char_results_TFRecords --train_batch_size=128
