from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import utility.logger_tool
import logging
from A4_convolutions.main import ConvolutionNet


class ConvolutionNetwithMaxPooling(ConvolutionNet):
    def getTempModelOutput(self, dataset, keep_prob):
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
#         res = tf.matmul(dataset, self.weights) + self.biases
#         h_layer1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        conv = tf.nn.conv2d(dataset, self.layer1_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.layer1_biases)
        hidden = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        
        conv = tf.nn.conv2d(hidden, self.layer2_weights, [1, 1, 1, 1], padding='SAME')
        hidden = tf.nn.relu(conv + self.layer2_biases)
        hidden = tf.nn.max_pool(hidden, [1,2,2,1], [1,2,2,1], padding='SAME')
        
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, self.layer3_weights) + self.layer3_biases)
        
        return tf.matmul(hidden, self.layer4_weights) + self.layer4_biases
    
    

if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/ConvolutionNetwithMaxPooling.log',filemode='w',level=logging.DEBUG)
    obj= ConvolutionNetwithMaxPooling()
    obj.run()