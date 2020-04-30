# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:01:50 2020

@author: 
"""

from keras.datasets import boston_housing

(train_data, train_price), (test_data, test_price) = boston_housing.load_data()

print(train_data.shape)
print(train_price.shape)

# 1. dane przeskalowac
import numpy as np
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data -= mean
x_train = train_data/std

# Sholle radzi uzywać mean+std tylko z train_data, nigdy z test_data ...?
test_data -= mean
x_test = test_data/std

# 2. model: dense, no activation function!
from keras import models, layers, losses, optimizers, metrics

model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(train_data.shape[1],)))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=optimizers.RMSprop(),
              loss=losses.mse,
              metrics=[metrics.mae])
