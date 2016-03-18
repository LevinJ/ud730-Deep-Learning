from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import utility.logger_tool
import logging
from A4_convolutions.main import ConvolutionNet
from A2_fullyconnected.main import ModelSrouce


class ConvolutionNetFinal(ConvolutionNet):
    def __init__(self):
        ConvolutionNet.__init__(self)
#         self.num_steps = 250 + 1
        self.num_steps = 95 * 1000 + 1
        self.train_subset = 519.090e+3
        self.keep_prob = 0.75
        return
    def setupOptimizer(self):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.05
        decay_steps = 3500
        decay_rate = 0.86
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=True)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        return
    def setupVariables(self):
        patch_size = 3
        depth = 16
        num_hidden = 705
        num_hidden_last = 205
        
        self.layerconv1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, self.num_channels, depth], stddev=0.1))
        self.layerconv1_biases = tf.Variable(tf.zeros([depth]))
        
        self.layerconv2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth * 2], stddev=0.1))
        self.layerconv2_biases = tf.Variable(tf.zeros([depth * 2]))
        
        self.layerconv3_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth * 2, depth * 4], stddev=0.03))
        self.layerconv3_biases = tf.Variable(tf.zeros([depth * 4]))
        
        self.layerconv4_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth * 4, depth * 4], stddev=0.03))
        self.layerconv4_biases = tf.Variable(tf.zeros([depth * 4]))
        
        
        self.layerconv5_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth * 4, depth * 16], stddev=0.03))
        self.layerconv5_biases = tf.Variable(tf.zeros([depth * 16]))
        
        
        self.layer3_weights = tf.Variable(tf.truncated_normal(
          [self.image_size / 7 * self.image_size / 7 * (depth * 4), num_hidden], stddev=0.03))
        self.layer3_biases = tf.Variable(tf.zeros([num_hidden]))
        
        self.layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_hidden_last], stddev=0.0532))
        self.layer4_biases = tf.Variable(tf.zeros([num_hidden_last]))
        
        self.layer5_weights = tf.Variable(tf.truncated_normal(
          [num_hidden_last, self.num_labels], stddev=0.1))
        self.layer5_biases = tf.Variable(tf.zeros([self.num_labels]))
        return
    def getTempModelOutput(self, dataset, keep_prob):
        
        conv = tf.nn.conv2d(dataset, self.layerconv1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.elu(conv + self.layerconv1_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        
        conv = tf.nn.conv2d(pool, self.layerconv2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.elu(conv + self.layerconv2_biases)
        #pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        
        
        conv = tf.nn.conv2d(hidden, self.layerconv3_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.elu(conv + self.layerconv3_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # norm1
        # norm1 = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        conv = tf.nn.conv2d(pool, self.layerconv4_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.elu(conv + self.layerconv4_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # norm1 = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        
        conv = tf.nn.conv2d(pool, self.layerconv5_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.elu(conv + self.layerconv5_biases)
        pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        # norm1 = tf.nn.lrn(pool, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
        
        shape = pool.get_shape().as_list()
        reshape = tf.reshape(pool, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.elu(tf.matmul(reshape, self.layer3_weights) + self.layer3_biases)
        
        hidden = tf.nn.dropout(hidden, keep_prob)
        
        nn_hidden_layer = tf.matmul(hidden, self.layer4_weights) + self.layer4_biases
        hidden = tf.nn.elu(nn_hidden_layer)
        
        hidden = tf.nn.dropout(hidden, keep_prob)
        
        
        return tf.matmul(hidden, self.layer5_weights) + self.layer5_biases
    
    

if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/ConvolutionNetFinal.log',filemode='w',level=logging.DEBUG)
    obj= ConvolutionNetFinal()
    trainModule = True
    
    
    if trainModule:
        obj.run(modelSrc= ModelSrouce.TRAIN_MODEL)
    else:
        obj.run(modelSrc= ModelSrouce.RESTORE_MODEL)