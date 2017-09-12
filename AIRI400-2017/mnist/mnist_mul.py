import os
import sys
import numpy as np
import tensorflow as tf
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt

fd = open('train-images.idx3-ubyte')
train_images = np.fromfile(file=fd,dtype=np.uint8)
fd = open('train-labels.idx1-ubyte')
train_labels = np.fromfile(file=fd,dtype=np.uint8)
fd = open('t10k-images.idx3-ubyte')
test_images = np.fromfile(file=fd,dtype=np.uint8)
fd = open('t10k-labels.idx1-ubyte')
test_labels = np.fromfile(file=fd,dtype=np.uint8)

train_images = (train_images[16:]/127-1).reshape([-1, 784]).astype(np.float)
train_labels = train_labels[8:].reshape([-1,]).astype(np.int)
test_images = (test_images[16:]/127-1).reshape([-1, 784]).astype(np.float)
test_labels = test_labels[8:].reshape([-1]).astype(np.int)

X = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='X')
Y = tf.placeholder(shape=[None], dtype=tf.uint8, name='Y')
W = tf.Variable(tf.zeros([784, 10]), name='W')
B = tf.Variable(tf.zeros([10]), name='B')


def save_image(path, image):
    scipy.misc.imsave(path, image)

def layer(input, weight_shape, bias_shape, name):
	with tf.name_scope(name):
		W = tf.Variable(tf.truncated_normal(shape=weight_shape, mean=0.0, stddev=0.02, name='layer_W'))
		B = tf.Variable(tf.zeros(bias_shape), name='layer_B')
		H = tf.matmul(input, W, name='ful_H') + B
		output = tf.nn.relu(H, name='layer_relu')
		return W, B, H, output

def fully_connect(input, weight_shape, bias_shape, name):
	with tf.name_scope(name):
		W = tf.Variable(tf.truncated_normal(shape=weight_shape, mean=0.0, stddev=0.02, name='layer_W'))
		B = tf.Variable(tf.zeros(bias_shape), name='ful_B')
		H = tf.matmul(input, W, name='ful_H') + B
		softmax = tf.nn.softmax(H, name='softmaxw')
		return H, softmax


## Layer 1 => 0-W, 1-B, 2-W, 3-output
output_L1 = layer(X, [784, 200], [200], 'layer1')[3]

output_L2 = layer(output_L1, [200, 400], [400], 'layer2')[3]

output_L3 = layer(output_L2, [400, 800], [800], 'layer3')[3]

output_L4 = layer(output_L3, [800, 1600], [1600], 'layer4')[3]

output_L5 = layer(output_L4, [1600, 3200], [3200], 'layer5')[3]

output_L6 = layer(output_L5, [3200, 1600], [1600], 'layer6')[3]

output_L7 = layer(output_L6, [1600, 800], [800], 'layer7')[3]

output_L8 = layer(output_L7, [800, 400], [400], 'layer8')[3]

## fully connect => 0-output, 1-sofrmax
ful_output, softmax = fully_connect(output_L8, [400, 10], [10], 'ful1')

## one_hot label
Y_one = tf.one_hot(Y, 10, axis=1)

## minimize
entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_one, logits=ful_output, name='loss')
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# ACC
Accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(softmax, 1), tf.argmax(Y_one, 1))))
# print(layer(X, [784, 2000], [2000], 'layer1')[0])
# print(layer(X, [784, 2000], [2000], 'layer1')[1])
# print(layer(X, [784, 2000], [2000], 'layer1')[2])
# print(layer(X, [784, 2000], [2000], 'layer1')[3])

# print(ful_output[0])
# print(ful_output[1])

# print(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	epoch = 40
	batch_size = 10000
	batch_count = 60000 // batch_size
	for i in range(epoch):
		print("epoch :", i)
		loss_total=0
		for i in range(batch_count):
			batch_index = i*batch_size
			img = train_images[batch_index:batch_index+batch_size]
			label = train_labels[batch_index:batch_index+batch_size]
			_,loss_val = sess.run([train_op, loss],feed_dict={X:img, Y:label})
			loss_total += loss_val
		print("loss ==> ", loss_total)

	test_img = test_images[:]
	test_label = test_labels[:]
	acc = sess.run([Accuracy], feed_dict={X:test_img, Y:test_label})
	print("Accuracy : ", acc) 

	tf.summary.scalar('loss',loss)
	tf.summary.scalar('accuracy',acc)
	summary_op = tf.summary.merge_all()
	writer = tf.summary.FileWriter('logdir',graph=tf.get_default_graph())
