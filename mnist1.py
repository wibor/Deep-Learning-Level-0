# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:40:00 2020

@author: 
"""

# get data
from keras.datasets import mnist

(train_im, train_lab), (test_im, test_lab) = mnist.load_data()
print(len(train_lab), train_lab.shape)
print(train_im.shape, train_im.ndim)
print(test_im.shape)

digit = train_im[5]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# reshape data and y to categorical
train_im = train_im.reshape((60000, 28*28)).astype('float32')/256
test_im = test_im.reshape((10000, 28*28)).astype('float32')/256

from keras.utils import to_categorical

train_lab = to_categorical(train_lab)
test_lab = to_categorical(test_lab)

# network
from keras import models, layers

network = models.Sequential()
network.add(layers.Dense(units=512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(units=10, activation='softmax'))

# rmsprop or adam: rmsprop - best!
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


# fit
network.fit(train_im, train_lab, batch_size=128, epochs=8)


# test
test_loss, test_accuracy = network.evaluate(test_im, test_lab)
print('accuracy ', test_accuracy)
# overfitting after 5th epocs

# geometric mean: bad result here :(
from math import sqrt
gm = int(sqrt(28*28*10))
