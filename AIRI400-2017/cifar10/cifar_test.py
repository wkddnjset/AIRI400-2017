import pickle
import os
import sys
import numpy as np
import tensorflow as tf
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def dataset(batch):


batch_1 = unpickle("./cifar-10-batches-py/data_batch_1")
batch_2 = unpickle("./cifar-10-batches-py/data_batch_1")
batch_3 = unpickle("./cifar-10-batches-py/data_batch_1")
batch_4 = unpickle("./cifar-10-batches-py/data_batch_1")
batch_5 = unpickle("./cifar-10-batches-py/data_batch_1")
test_batch = unpickle("./cifar-10-batches-py/test_batch")

batch_1[0]