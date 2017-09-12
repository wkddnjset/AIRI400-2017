import tensorflow as tf
import numpy as np
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

train_images = (train_images[16:]).reshape([-1, 784]).astype(np.float)
train_labels = train_labels[8:].reshape([-1,]).astype(np.int)
test_images = (test_images[16:]).reshape([-1, 784]).astype(np.float)
test_labels = test_labels[8:].reshape([-1]).astype(np.int)

data = np.load('./weight_image/weight.txt.npy')

img_arr = data.reshape(-1,784)
print(img_arr[0].shape)
print(train_images[0].shape)
a = img_arr[0]*train_images[0]
print(a.shape)
print(a)
img = Image.fromarray(a.reshape(28,28))
print(img)
imgplot = plt.imshow(img)
plt.show(img)

# img = Image.fromarray(img_arr[0])
# print(img)
# imgplot = plt.imshow(img)
# plt.show(img)
# weight_arr = weight_arr.reshape((-1, 784))
# print(weight_arr.shape)

# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	saver.restore(sess, "./weight_image/my-model-5")
# 	data = sess.run(W)
# 	trans_data = data.transpose()
# 	print(trans_data[0])
# 	for i in range(0, 9):
# 		a = (trans_data[i].reshape(28,28))

		# img = Image.fromarray(a)
		# print(img)
		# imgplot = plt.imshow(img)
		# plt.show(img)