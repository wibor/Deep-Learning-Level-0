# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 19:01:02 2020

@author: tatus
"""

from keras.datasets import reuters

(train_data, train_lab), (test_data, test_lab) = reuters.load_data(num_words=10_000)

# 46 category need to be one_hot encoded
# network output: 46 neurons + softmax

import numpy as np
def vectorise(sequences, dimension=10000):
    #zeros((a,b)) <- tuple inside!
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorise(train_data)
x_test = vectorise(test_data)
# or use keras function:
# from keras.utils.np_utils import to_categorical
# one_hot_test_labels = to_categorical(test_labels)
y_train = vectorise(train_lab, dimension=46)
y_test = vectorise(test_lab, dimension=46)


# network
from keras import models, layers, losses, optimizers, metrics

network = models.Sequential()
network.add(layers.Dense(units=164, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(units=164, activation='relu'))
network.add(layers.Dense(units=46, activation='softmax'))

# rmsprop or adam: 
network.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                loss=losses.categorical_crossentropy, 
                metrics=[metrics.categorical_accuracy])

# ...==========================================================================
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

history = network.fit(partial_x_train,
                      partial_y_train,
                      epochs=15,
                      batch_size=512,
                      validation_data=(x_val, y_val))

print(history.history.keys()) 
acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line", r for 'red'
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history.history['categorical_accuracy']
val_acc_values = history.history['val_categorical_accuracy']

plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# =============================================================================

network.fit(x_train, y_train, batch_size=512, epochs=6)

results = network.evaluate(x_test, y_test)
print(results)