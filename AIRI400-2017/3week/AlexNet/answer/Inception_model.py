from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import time

import numpy as np
import tensorflow as tf

def print_features(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def inception_block(net, deepth, is_training):
    """35x35 resnet block"""
    with tf.variable_scope("branch_0"):
        br0 = tf.layers.conv2d(net, deepth, [1, 1], [2, 2], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br0 = tf.layers.batch_normalization(br0, training=is_training)
        br0 = tf.nn.relu(br0)
    with tf.variable_scope("branch_1"):
        br1 = tf.layers.conv2d(net, deepth, [1, 1], [2, 2], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br1 = tf.nn.relu(tf.layers.batch_normalization(br1))
        br1 = tf.layers.conv2d(br1, deepth, [3, 3], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br1 = tf.layers.batch_normalization(br1, training=is_training)
        br1 = tf.nn.relu(br1)
    with tf.variable_scope("branch_2"):
        br2 = tf.layers.conv2d(net, deepth, [1, 1], [2, 2], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br2 = tf.nn.relu(tf.layers.batch_normalization(br2))
        br2 = tf.layers.conv2d(br2, deepth, [5, 5], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br2 = tf.layers.batch_normalization(br2, training=is_training)
        br2 = tf.nn.relu(br2)
    with tf.variable_scope("branch_3"):
        # BN?
        br3 = tf.layers.max_pooling2d(net, [3, 3], [2, 2], padding='SAME',
                         name='pool_3x3')
        br3 = tf.layers.conv2d(br3, deepth, [1, 1], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br3 = tf.layers.batch_normalization(br3, training=is_training)
        br3 = tf.nn.relu(br3)

    concatenated = tf.concat([br0, br1, br2, br3], 3)
    return concatenated

def residual_A(x, deepth, filter_shape, is_training):
    """Residual unit with 2 sub layers."""

    orig_x = x
    
    with tf.variable_scope('sub1'):
        # full_pre-activation
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, deepth, filter_shape, padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        
    with tf.variable_scope('sub2'):
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d(x, deepth, filter_shape, padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))

    with tf.variable_scope('sub_add'):
        x += orig_x

    #tf.logging.debug('feature maps after unit %s', x.get_shape())
    return x

class Inception(object):

  def __init__(self, x, num_classes, is_training):
    """생성자. 이미지 데이터와 dropout layer 컨트롤을 위한 is_training을 placeholder로 받음."""
    self.X = x
    self.is_training = is_training
    self.NUM_CLASSES = num_classes

    self.create()


  def create(self):
    """Create the network graph.
    We will use tf.layers.conv2d/max_pooling2d/dense/dropout, tf.nn.lrn, etc.
    """
    # input image size = 224x224x3

    ##################################
    # 1-1. conv1 = tf.layers.conv2d...
    # print_features(conv1)
    # tensor shape: 55(56)x55x96
    #
    # 1-2. Batch Normalization
    #
    # 1-3. Activation
    #
    # 1-4. pool1
    # tensor shape: 27(26)x27x96
    ###################################
    is_training = self.is_training
    # img = tf.reshape(self.X, shape=[None, 224*224*3])
    conv1 = tf.layers.conv2d(
                            self.X, 
                            96, 
                            [11, 11], 
                            strides=[4, 4],
                            padding='SAME',
                            activation=None,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            bias_initializer=tf.zeros_initializer(),
                            name='conv1'
                            )
    conv1 = tf.layers.batch_normalization(conv1, training=is_training)
    conv1 = tf.nn.relu(conv1)
    pool1 = tf.layers.max_pooling2d(
                                conv1,
                                [3,3],
                                strides=[2,2],
                                padding='VALID',
                                name='pool1')


    incep1 = inception_block(pool1, 64, is_training)
    # conv1x1 depth relu x activ x
    res1 = residual_A(incep1, 128, [5, 5], is_training)
    res2 = residual_A(incep1, 128, [3, 3], is_training)
    res3 = residual_A(incep1, 128, [1, 1], is_training)

    with tf.variable_scope("branch_4"):
        br4 = tf.layers.conv2d(res1, 96, [1, 1], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br4 = tf.layers.batch_normalization(br4, training=is_training)
        br4 = tf.nn.relu(br4)
    with tf.variable_scope("branch_5"):
        br5 = tf.layers.conv2d(res2, 48, [1, 1], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br5 = tf.nn.relu(tf.layers.batch_normalization(br5))
        br5 = tf.layers.conv2d(br5, 48, [3, 3], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br5 = tf.layers.batch_normalization(br5, training=is_training)
        br5 = tf.nn.relu(br5)
    with tf.variable_scope("branch_6"):
        br6 = tf.layers.conv2d(res3, 32, [1, 1], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_1x1',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br6 = tf.nn.relu(tf.layers.batch_normalization(br6))
        br6 = tf.layers.conv2d(br6, 32, [5, 5], padding='SAME',
                         activation=None,
                         use_bias=True, name='conv_3x3',
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        br6 = tf.layers.batch_normalization(br6, training=is_training)
        br6 = tf.nn.relu(br6)

    result = tf.concat([br4, br5, br6], 3)

    self.logits = tf.layers.dense(
                        result,
                        10,
                        activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer,
                        bias_initializer=tf.constant_initializer(0.1),
                        name='logits'
                        )