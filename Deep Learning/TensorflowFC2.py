# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import os
import ssl
from time import time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

# Download the dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

"""For this, start by defining two placeholders, one to hold the images, and the second to hold the two classes.
Use tf.float32 for the placeholder type.
"""

learning_rate = 0.5
batch_size = 100

# create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# each lable is one hot vector.

# correct labels
y = tf.placeholder(tf.float32, [None, 10])

# input data
x = tf.placeholder(tf.float32, [None, 784])

"""Next, define the network itself. It is up to you how many layers to use, and the number of hidden units in each layer.

You are allowed to use only the following functions:
* weight_variable
* bias_variable
* tf.nn.relu
* tf.nn.softmax
* tf.matmul
* tf.argmax
* tf.reduce_mean
* tf.cast

Please note that each layer includes not only tf.matmul, but also a bias variable.
"""

# build the net

hidden_size = 300
num_classes = 10
W_fc1 = tf.Variable(tf.random_normal([784, hidden_size], stddev=0.03), name='W_fc1')
b_fc1 = tf.Variable(tf.random_normal([hidden_size]), name='b_fc1')
h_fc1 = tf.add(tf.matmul(x, W_fc1), b_fc1)
h_fc1 = tf.nn.relu(h_fc1)

W_fc2 = tf.Variable(tf.random_normal([hidden_size, num_classes], stddev=0.03), name='W_fc2')
b_fc2 = tf.Variable(tf.random_normal([num_classes]), name='b_fc2')
y_fc2 = tf.nn.softmax(tf.add(tf.matmul(h_fc1, W_fc2), b_fc2))


"""Complete the snippet below using your own code.
define the loss function and Optimizer
"""

# define the loss function
y_clipped = tf.clip_by_value(y_fc2, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# define Optimizer
Optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_fc2, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()


"""The next code snippet trains and evaluates the network. It does this by opening a session to run the tensorflow graph that we have defined.
Complete the code at the locations marked #YOUR CODE below, in order to train the network and to evaluate its accuracy every 50 steps.
"""

with tf.Session() as sess:
  sess.run(init)

  for i in range(700):

    input_images, correct_predictions = mnist.train.next_batch(batch_size)

    batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)

    sess.run([Optimizer, cross_entropy], feed_dict={x: batch_xs, y: batch_ys})
    if i % 50 == 0:
          train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
          print("step %d, training accuracy %g" % (i, train_accuracy))

          test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
          print("Validation accuracy: %g." % test_accuracy)
