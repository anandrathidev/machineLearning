# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 11:14:50 2017

@author: rb130
"""

import tensorflow as tf
from createTensor import create_feature_sets_and_labels
import numpy as np
#mnist = 
train_x,train_y = create_feature_sets_and_labels()
#print("===========" , train_x)

num_input = 2500
n_nodes_hl1 = 2500
n_nodes_hl2 = 5000
n_nodes_hl3 = 2500

n_classes = 2
batch_size = 15

hm_epochs = 2

x = tf.placeholder("float", [None, num_input])
#y = tf.placeholder("float", [None, n_classes])
y = tf.placeholder("float")

# Nothing changes


hidden_1_layer = {'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

output_layer = {'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}

def neural_network_model(data):
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weight']) + output_layer['bias']

    return output

def train_neural_network(x):
    # Construct model 
    logits = neural_network_model(x)
    
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      for epoch in range(hm_epochs):
        epoch_loss = 0
        i=0
        while i < len(train_x):
          start = i
          end = i+batch_size
          batch_x = np.array(train_x[start:end])
          batch_y = np.array(train_y[start:end])
          _, c = sess.run([train_op, loss_op], feed_dict={x: batch_x,
				                                              y: batch_y})          
          # _, c = sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
          
          
          # epoch_loss += c
          i+=batch_size
				
      print('Epoch', epoch+1, 'completed out of', hm_epochs,'loss:',epoch_loss)

      # Evaluate model
      prediction = tf.nn.softmax(logits)
      correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      print('Accuracy:', accuracy.eval({x:train_x, y:train_y}))

train_neural_network(x)