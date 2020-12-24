#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 14:47:15 2019

@author: neeleshbhajantri
"""

# Read Fashion MNIST dataset

import util_mnist_reader as mnist_reader
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Your code goes here . . .
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 180
class_no = 10
epochs = 30

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
#print(x_train.shape, "This is Training")
#print(y_test.shape, "This is Testing")

y_train = keras.utils.to_categorical(y_train, class_no)
y_test = keras.utils.to_categorical(y_test, class_no)

model = Sequential()
model.add(Dense(512, activation = 'relu', input_shape = (784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(class_no, activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])

history = model.fit(x_train, y_train,  batch_size = batch_size, 
                    epochs = epochs, verbose = 1, 
                    validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test Loss", score[0])
print("Test Accuracy", score[1])
