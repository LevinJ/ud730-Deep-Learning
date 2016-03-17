from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import utility.logger_tool
import logging
from A4_convolutions.main import ConvolutionNet


class ConvolutionNetFinal(ConvolutionNet):
    def __init__(self):
        ConvolutionNet.__init__(self)
        self.num_steps = 120 * 1000 + 1
        self.train_subset = 519.090e+3
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
    def getTempModelOutput(self, dataset, keep_prob):
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
#         res = tf.matmul(dataset, self.weights) + self.biases
#         h_layer1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        conv = tf.nn.conv2d(dataset, self.layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.layer1_biases)
        
        conv = tf.nn.conv2d(hidden, self.layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.layer2_biases)
        
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, self.layer3_weights) + self.layer3_biases)
        
        return tf.matmul(hidden, self.layer4_weights) + self.layer4_biases
    
    

if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/ConvolutionNetFinal.log',filemode='w',level=logging.DEBUG)
    obj= ConvolutionNetFinal()
    obj.run()