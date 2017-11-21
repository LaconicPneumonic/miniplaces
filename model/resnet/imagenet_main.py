# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import tensorflow as tf

from DataLoader import *
import resnet_model
import vgg_preprocessing

# Dataset Parameters
batch_size = 200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 10000
path_save = 'alexnet'
start_from = ''


parser = argparse.ArgumentParser()

parser.add_argument(
    '--data_dir', type=str, default='',
    help='The directory where the ImageNet input data is stored.')

parser.add_argument(
    '--model_dir', type=str, default='build/resnet_model',
    help='The directory where the model will be stored.')

parser.add_argument(
    '--resnet_size', type=int, default=50, choices=[18, 34, 50, 101, 152, 200],
    help='The size of the ResNet model to use.')

parser.add_argument(
    '--train_epochs', type=int, default=1,
    help='The number of epochs to use for training.')

parser.add_argument(
    '--epochs_per_eval', type=int, default=1,
    help='The number of training epochs to run between evaluations.')

parser.add_argument(
    '--batch_size', type=int, default=100,
    help='Batch size for training and evaluation.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')

_DEFAULT_IMAGE_SIZE = 128
_NUM_CHANNELS = 3
_LABEL_CLASSES = 100

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 100000,
    'validation': 10000,
    'test': 100000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500


def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(1024)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]


def record_parser(value, is_training):
  """Parse an ImageNet record from `value`."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = vgg_preprocessing.preprocess_image(
      image=image,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
#

def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  print("INPUT_FN CALLED")
  if is_training:
    dataset = loader_train.next_batch(batch_size)
  else:
    dataset = loader_val.next_batch(batch_size)

  images, labels = dataset[0].astype('float32'), dataset[1].astype('float32')

  return images, tf.one_hot(labels, 100)


def resnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  tf.summary.image('images', features, max_outputs=6)
  print(features.shape)
  network = resnet_model.imagenet_resnet_v2(
      params['resnet_size'], _LABEL_CLASSES, params['data_format'])
  logits = network(
      inputs=features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
  print(logits.shape)
  print(labels)
  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # Add weight decay to the loss. We exclude the batch norm variables because
  # doing so leads to a small improvement in accuracy.
  loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'batch_normalization' not in v.name])

  if mode == tf.estimator.ModeKeys.TRAIN:
    # Scale the learning rate linearly with the batch size. When the batch size
    # is 256, the learning rate should be 0.1.
    initial_learning_rate = 0.1 * params['batch_size'] / 256
    batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
    global_step = tf.train.get_or_create_global_step()

    # Multiply the learning rate by 0.1 at 30, 60, 80, and 90 epochs.
    boundaries = [
        int(batches_per_epoch * epoch) for epoch in [30, 60, 80, 90]]
    values = [
        initial_learning_rate * decay for decay in [1, 0.1, 0.01, 1e-3, 1e-4]]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32), boundaries, values)

    # Create a tensor named learning_rate for logging purposes.
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate,
        momentum=_MOMENTUM)

    # Batch norm requires update_ops to be added as a train_op dependency.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss, global_step)
  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes.
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=metrics)


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  resnet_classifier = tf.estimator.Estimator(
      model_fn=resnet_model_fn, model_dir=FLAGS.model_dir, config=run_config,
      params={
          'resnet_size': FLAGS.resnet_size,
          'data_format': FLAGS.data_format,
          'batch_size': FLAGS.batch_size,
      })

  for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    print("HELLO")
    """
    tensors_to_log = {
        'learning_rate': 'learning_rate',
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    print('Starting a training cycle.')
    resnet_classifier.train(
        input_fn=lambda: input_fn(
            True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])
    """
    
    print('Starting to evaluate.')
    eval_tensors_to_log = {
        'cross_entropy': 'cross_entropy',
        'train_accuracy': 'train_accuracy'
    }
    
    eval_logging_hook = tf.train.LoggingTensorHook(
        tensors=eval_tensors_to_log, every_n_iter=100)

    eval_results = resnet_classifier.evaluate(
        input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size),
        hooks=[eval_logging_hook])
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
