#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:55:23 2019

@author: neeleshbhajantri
"""

# Read Fashion MNIST dataset

import util_mnist_reader as mnist_reader
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Your code goes here . . .
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

batch_size = 180
class_no = 10
epochs = 30
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
#y_train = y_train.reshape(60000, 1,)
#y_test = y_test.reshape(10000, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, class_no)
y_test = keras.utils.to_categorical(y_test, class_no)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size = 2, padding = 'same', activation = 'relu',
                 input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 2, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(class_no, activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy',optimizer = Adam(), metrics = ['accuracy'])
history = model.fit(x_train, y_train,  batch_size = batch_size, 
                    epochs = epochs, verbose = 1, 
                    validation_data = (x_test, y_test))
score = model.evaluate(x_test, y_test, verbose = 0)
print("Test Loss", score[0])
print("Test Accuracy", score[1])
