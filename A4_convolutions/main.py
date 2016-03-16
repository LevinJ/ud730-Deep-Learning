from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.p1_relulayer import HiddenRelu
import math
import utility.logger_tool
import logging


class ConvolutionNet(HiddenRelu):
    def __init__(self):
        self.num_channels = 1 # grayscale
        HiddenRelu.__init__(self)
        self.num_steps = 1 * 1000 + 1
        return
    def getInputData(self):
        self.tf_train_dataset = tf.placeholder(tf.float32, shape=(self.batch_size, self.image_size, self.image_size, self.num_channels))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
        return
    def setBatchSize(self):
        self.batch_size = 16
        return
    def reformatDataset(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size, self.image_size, self.num_channels)).astype(np.float32)
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
    
    def setTrainSampleNumber(self):
        self.train_subset = 200e+3
        return
#     def setDropout(self):
#         self.keep_prob = 0.9
#         return
    def setupOptimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)
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

    def setupVariables(self):
        patch_size = 5
        depth = 16
        num_hidden = 64
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        self.layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, self.num_channels, depth], stddev=0.1))
        self.layer1_biases = tf.Variable(tf.zeros([depth]))
        
        self.layer2_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, depth, depth], stddev=0.1))
        self.layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
        
        self.layer3_weights = tf.Variable(tf.truncated_normal(
        [self.image_size // 4 * self.image_size // 4 * depth, num_hidden], stddev=0.1))
        self.layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
        
        self.layer4_weights = tf.Variable(tf.truncated_normal(
        [num_hidden, self.num_labels], stddev=0.1))
        self.layer4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))
        return



if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/ConvolutionNet.log',filemode='w',level=logging.DEBUG)
    obj= ConvolutionNet()
    obj.run()
