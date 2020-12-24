#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 02:06:32 2019

@author: neeleshbhajantri
"""

# Read Fashion MNIST dataset

import util_mnist_reader as mnist_reader
x_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

# Your code goes here . . .
import numpy as np
import matplotlib.pyplot as plt

m_train = 60000
m_test = 10000

mask = list(range(m_train))
x_train = x_train[mask]
y_train = y_train[mask]

mask = list(range(m_test))
x_test = x_test[mask]
y_test = y_test[mask]
#reshaping to rows
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

class NN(object):
    
    def __init__(self, in_size, h_size, out_size, std=1e-4):
        self.params = {}    
        self.params['w1'] = std * np.random.randn(in_size, h_size)   
        self.params['b1'] = np.zeros((1, h_size))
        self.params['w2'] = std * np.random.randn(h_size, out_size)   
        self.params['b2'] = np.zeros((1, out_size))
        
    def loss_func(self, x, y = None, reg = 0.0):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        N, D = x.shape
        
        score = None
        #print(w1.shape)
        h1 = self.ReLU(np.dot(x, w1) + b1)
        output = np.dot(h1, w2) + b2
        score = output
        
        if y is None:
            return score
        max_score = np.max(score, axis=1, keepdims=True)
        exp_score = np.exp(score - max_score)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        correct_N = -np.log(probs[range(N), y])
        lost_data = np.sum(correct_N) / N
        reg_loss = 0.5 * reg * np.sum(w1*w1) + 0.5 * reg * np.sum(w2*w2)
        loss = lost_data + reg_loss

        #backward propogation

        grads = {}
        dscores = probs                                 
        dscores[range(N), y] -= 1
        dscores /= N
        dw2 = np.dot(h1.T, dscores)                     
        db2 = np.sum(dscores, axis=0, keepdims=True)    
        dh1 = np.dot(dscores, w2.T)                     
        dh1[h1 <= 0] = 0
        dw1 = np.dot(x.T, dh1)                          
        db1 = np.sum(dh1, axis=0, keepdims=True)        
        dw2 += reg * w2
        dw1 += reg * w1

        grads['w1'] = dw1
        grads['b1'] = db1
        grads['w2'] = dw2
        grads['b2'] = db2

        return loss, grads
    
    def train(self, x, y, x_test, y_test, learning_rate = 1e-3, 
               learning_rate_decay = 0.95, reg = 1e-5, mu = 0.9, epoch_no = 10, 
               mu_increase = 1.0, batch_size = 200, verbose = False):
               train_no = x.shape[0]
               iterations = max(int(train_no / batch_size), 1)
               v_w2, v_b2 = 0.0, 0.0
               v_w1, v_b1 = 0.0, 0.0
               loss_history = []
               train_acc_history = []
               test_acc_history = []
               for i in range(1, epoch_no * iterations + 1):
                   x_batch = None
                   y_batch = None
                   sample_index = np.random.choice(train_no, batch_size, replace=True)   
                   x_batch = x[sample_index, :]          
                   y_batch = y[sample_index]
                   loss, grads = self.loss_func(x_batch, y=y_batch, reg=reg) 
                   loss_history.append(loss)
                   v_w2 = mu * v_w2 - learning_rate * grads['w2']
                   self.params['w2'] += v_w2   
                   v_b2 = mu * v_b2 - learning_rate * grads['b2']    
                   self.params['b2'] += v_b2   
                   v_w1 = mu * v_w1 - learning_rate * grads['w1']    
                   self.params['w1'] += v_w1   
                   v_b1 = mu * v_b1 - learning_rate * grads['b1']  
                   self.params['b1'] += v_b1

                   if verbose and i % iterations == 0:
                       epoch = i/iterations
                       train_acc = (self.predict(x_batch) == y_batch).mean()
                       test_acc = (self.predict(x_test) == y_test).mean()
                       train_acc_history.append(train_acc)
                       test_acc_history.append(test_acc)
                       learning_rate = learning_rate * learning_rate_decay
                       mu = mu * mu_increase
               return {'loss_history': loss_history, 
               'train_acc_history': train_acc_history, 
               'test_acc_history': test_acc_history}

    def predict(self, x):
        y_pred = None
        h1 = self.ReLU(np.dot(x, self.params['w1']) + self.params['b1'])
        score = np.dot(h1, self.params['w2']) + self.params['b2']
        y_pred = np.argmax(score, axis = 1)
        return y_pred
    
    def ReLU(self, x):
        return np.maximum(0,x)

in_size = x_train.shape[1]
h_size = 10
no_classes = 10
call = NN(in_size, h_size, no_classes)

history = call.train(x_train, y_train, x_test, y_test, epoch_no = 1000, batch_size = 500, 
learning_rate = 7.5e-4, learning_rate_decay = 0.9, reg = 1.0, verbose = True)
test_acc = (call.predict(x_test) == y_test).mean()
print(test_acc)
# Plot the loss function and train / validation accuracies

plt.subplot(2, 1, 1)
plt.plot(history['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')


plt.subplot(2, 1, 2)
plt.plot(history['train_acc_history'], label='train')
plt.plot(history['test_acc_history'], label='val')
#plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()
