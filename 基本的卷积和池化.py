

import tensorflow as tf
import numpy as np

class FCN:
    def __init__(self,image,weights=None,sess=None):
        self.image = image
        self.weights = weights
        self.sess = sess
        self.convLayers()
        self.fcLayers()
    def conv2d(self,x,w):
        return tf.nn.conv2d(x,w,[1,1,1,1],padding="SAME")
    def max_poll_2x2(self,x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    def get_kernel(self,shape):
        return tf.Variable(tf.truncated_normal(shape,tf.float32,stddev=0.1))
    def get_bias(self,shape):
        return tf.Variable(tf.constant(0.1,tf.float32,shape=shape))
    
    def convLayers(self):
        self.parameters = []
        
        # conv1_1
        with tf.name_scope("conv1_1") as scope:
            kernel = self.get_kernel([3,3,1,32])
            bias = self.get_bias([32])
            self.conv1_1 = tf.nn.relu(self.conv2d(self.image,kernel) + bias)
            self.parameters +=[kernel,bias]
        # conv1_2
        with tf.name_scope as scope:
            kernel = self.get_kernel([3,3,32,64])
            bias = self.get_bias([64])
            self.conv1_2 = tf.nn.relu(self.conv2d(self.conv1_1,kernel) + bias)
            self.parameters +=[kernel,bias]
        # pool1
        with tf.name_scope as scope:
            self.pool1 = self.max_poll_2x2(self.conv1_2)

        # conv2_1
        with tf.name_scope as scope:
            kernel = self.get_kernel([3,3,64,128])
            bias = self.get_bias([128])
            self.conv2_1 = tf.nn.relu(self.conv2d(self.pool1,kernel) + bias)
            self.parameters +=[kernel,bias]
         # conv2_2
        with tf.name_scope as scope:
            kernel = self.get_kernel([3, 3, 128, 256])
            bias = self.get_bias([256])
            self.conv2_2 = tf.nn.relu(self.conv2d(self.conv2_1, kernel) + bias)
            self.parameters += [kernel, bias]
        # pool2
        with tf.name_scope as scope:
            self.pool2 = self.max_poll_2x2(self.conv2_2)


        #.....
  
    def fcLayers(self):
        # fc1
        with tf.name_scope as scope:
            shape = int( np.prod(self.pool2.get_shape()[1:]))   ##############
            W = tf.Variable(tf.truncated_normal([shape,1024],tf.float32,stddev=0.1))
            b = tf.Variable(tf.constant(0.1,[1024],tf.float32))
            pool2_flat = tf.reshape(self.pool2,[-1,shape])     ############
            pool2_drop = tf.nn.dropout(pool2_flat,tf.Variable(tf.constant(0.5,tf.float32)))
            self.fc1 = tf.nn.relu(tf.matmul(pool2_drop,W) + b )
            self.parameters +=[W,b]
        # fc2
        with tf.name_scope as scope:
            W = tf.Variable(tf.truncated_normal([1024,12],tf.float32,stddev=0.1))
            b = tf.Variable(tf.constant(0.1,tf.float32,[12]))
            fc1_drop = tf.nn.dropout(self.fc1,tf.Variable(tf.constant(0.5,tf.float32)))
            self.fc2 = tf.nn.relu(tf.matmul(fc1_drop,W) + b )
            self.parameters+= [W,b]
  
  
  
  
  
  
  
  
  
  
  
  