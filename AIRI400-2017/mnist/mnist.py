import os
import sys
import numpy as np
import tensorflow as tf
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt


def save_image(path, image):
    scipy.misc.imsave(path, image)


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

X = tf.placeholder(shape=[None, 784], dtype=tf.float32)
Y = tf.placeholder(shape=[None], dtype=tf.uint8)
W = tf.Variable(tf.zeros([784, 10]))
B = tf.Variable(tf.zeros([10]))
Y_one = tf.one_hot(Y, 10, axis=1)

H = tf.matmul(X, W) + B
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_one, logits=H))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# acc
test_val = tf.nn.softmax(H)
Accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_val, 1), tf.argmax(Y_one, 1)), dtype=tf.float32))

#saver


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	epoch = 10
	for i in range (epoch):
		print("epoch :", i)
		batch_size = 10000
		batch_count = 60000 // batch_size
		loss_total = 0
		for i in range (batch_count):
			batch_index = i * batch_size
			img = train_images[batch_index:batch_index+batch_size]
			label = train_labels[batch_index:batch_index+batch_size]
			# one = np.zeros((batch_index+batch_size,10))
			# one[np.arange(9), label] = 1
			# label = tf.one_hot(train_labels[i:batch_index+batch_size], 10, axis=1)
			_, loss_value = sess.run([train_op, loss], feed_dict={X:img, Y:label})
		loss_total += loss_value
		print("=> loss : ", loss_total)
			# print("batch ",i,"=> loss : ", loss_total)
	# acc
	test_img = test_images[:]
	test_label = test_labels[:]
	acc = sess.run([Accuracy], feed_dict={X:test_img, Y:test_label})
	print("Accuracy : ", acc)
	
	weight = sess.run(W)
	weight_arr = np.transpose(weight)
	for i in range(10):
		w = weight_arr[i].reshape([28, 28])
		w_min = w.min()
		w_max = w.max()
		w_1 = (w - w_min) / (w_max - w_min)
		w_255 = w_1 * 255
		plt.show(w_255)
		save_image('./weight_image/filter{}.png'.format(i), w_255)
	# weight_list = list(weight_arr)
	# print(len(weight_list[0]))
	# np.save('./weight_image/weight.txt',weight_list)
