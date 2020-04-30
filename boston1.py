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

def build_model(units=4):
    model = models.Sequential()
    model.add(layers.Dense(units=units, activation='relu', 
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(units=units, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(),
                  loss=losses.mse,
                  metrics=[metrics.mae])
    return model

# 3. k-fold cross validation (404 lines of data -> k=4)
from keras import backend as K
# Some memory clean-up
K.clear_session()

k = 4
chunk_lenght = len(train_data)//k
#results = []
all_mae_histories = []

for i in range(k):
    print(f'processing {i} from {k} chuncs')
    test_chunk = train_data[chunk_lenght*i: chunk_lenght*(i+1)]
    test_price_chunk = train_price[chunk_lenght*i: chunk_lenght*(i+1)]
    
    train_chunk = np.concatenate((
        train_data[: chunk_lenght*i],
        train_data[chunk_lenght*(i+1): ]), axis=0)
    train_price_chunk = np.concatenate((
        train_price[: chunk_lenght*i],
        train_price[chunk_lenght*(i+1): ]), axis=0)
    
    model = build_model(64)
    history = model.fit(x=train_chunk, y=train_price_chunk, batch_size=1, epochs=200, verbose=0)
    
    #results.append(model.evaluate(x=test_chunk, y=test_price_chunk, verbose=0))
    mae_history = history.history['mean_absolute_error'] 
    all_mae_histories.append(mae_history)
    
#results = np.array(results)[:,1]
#print(np.mean(results))
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(200)]

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


model = build_model(64)
# Train it on the entirety of the data.
model.fit(x_train, train_price,
          epochs=250, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(x_test, test_price)
