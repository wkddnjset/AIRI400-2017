import tensorflow as tf
import data_queue
from os import walk
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import sys

files = []
num_epochs = 1
num_batch_size = 300
for (path, dir, file) in walk("./celeb_tfrec"):
    for i in range(len(file)):
        path = path+'/'+file[i]
        files.append(path)

images = data_queue.make_data_pipeline(files, num_epochs, num_batch_size)
print(images)

