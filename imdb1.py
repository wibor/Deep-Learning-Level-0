# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:50:27 2020

@author: tatus
"""

# get data
from keras.datasets import imdb

(train_data, train_lab), (test_data, test_lab) = imdb.load_data(num_words=10_000)

print(len(train_lab), train_lab.shape)
print(train_data.shape, train_data.ndim)
print(train_data[0], train_data.dtype)
print(max([max(sequence) for sequence in train_data]))

# reshape data and y to categorical
import numpy as np
def vectorise(sequences, dimension=10000):
    #zeros((a,b)) <- tuple inside!
    results = np.zeros((len(sequences), dimension))
    
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results

x_train = vectorise(train_data)
x_test = vectorise(test_data)

y_train = np.asarray(train_lab).astype('float32')
y_test = np.asarray(test_lab).astype('float32')

# network
from keras import models, layers, losses, optimizers, metrics

network = models.Sequential()
network.add(layers.Dense(units=16, activation='tanh', input_shape=(10000,)))
network.add(layers.Dropout(rate=0.2))
network.add(layers.Dense(units=16, activation='tanh'))
network.add(layers.Dense(units=1, activation='sigmoid'))

# rmsprop or adam: 
network.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), 
                loss=losses.mse, 
                metrics=[metrics.binary_accuracy])


# fit
#network.fit(x_train, y_train, batch_size=256, epochs=10)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = network.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print(history.history.keys())                   
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # clear figure
acc_values = history.history['binary_accuracy']
val_acc_values = history.history['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# fit
network.fit(x_train, y_train, batch_size=512, epochs=3)


# test
prediction = network.predict(x_test)
print('probability ', prediction)

# again:
model = models.Sequential()
model.add(layers.Dense(12, activation='relu', input_shape=(10000,)))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(12, activation='relu'))
model.add(layers.Dropout(rate=0.2))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, batch_size=512)

results = model.evaluate(x_test, y_test)
print(results)