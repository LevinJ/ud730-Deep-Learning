from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.p1_relulayer import HiddenRelu
import math


class Mulilayer_HiddenRelu(HiddenRelu):
    def __init__(self):
        HiddenRelu.__init__(self)
        return
    def setupOptimizer(self):
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.5
        decay_steps = 500
        decay_rate = 0.3
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           decay_steps, decay_rate, staircase=True)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss, global_step=global_step)
        return
    def setIterationNum(self):
#         self.num_steps = 100 * 1000 + 1
        self.num_steps = 3 * 1000 + 1
        return
    def getTempModleOutput(self, dataset):
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
#         res = tf.matmul(dataset, self.weights) + self.biases
#         h_layer1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        a_layer1 = dataset
        
        z_layer2 = tf.matmul(a_layer1, self.weights_layer1 ) + self.biases_layer1
        a_layer2 = tf.nn.relu(z_layer2)
        
        z_layer3 = tf.matmul(a_layer2 , self.weights_layer2 ) + self.biases_layer2
        a_layer3 = tf.nn.relu(z_layer3)
         
         
        z_layer4 = tf.matmul(a_layer3 , self.weights_layer3 ) + self.biases_layer3
        a_layer4 = tf.nn.relu(z_layer4)
         
        z_layer5 = tf.matmul(a_layer4 , self.weights_layer4 ) + self.biases_layer4
        
       
        return z_layer5
    def setupVariables(self):
        layer1Num = self.image_size * self.image_size
        layer2Num = 1024
        layer3Num = 300
        layer4Num = 50
        layer5Num = self.num_labels
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        self.weights_layer1 = tf.Variable(tf.truncated_normal([layer1Num, layer2Num], stddev=1 / math.sqrt(float(layer1Num))))
        self.biases_layer1 = tf.Variable(tf.zeros([layer2Num]))
        
        self.weights_layer2 = tf.Variable(tf.truncated_normal([layer2Num, layer3Num], stddev=1 / math.sqrt(float(layer2Num))))
        self.biases_layer2 = tf.Variable(tf.zeros([layer3Num]))
        
        self.weights_layer3 = tf.Variable(tf.truncated_normal([layer3Num, layer4Num], stddev=1 / math.sqrt(float(layer3Num))))
        self.biases_layer3 = tf.Variable(tf.zeros([layer4Num]))
#         
        self.weights_layer4 = tf.Variable(tf.truncated_normal([layer4Num, layer5Num], stddev=1 / math.sqrt(float(layer4Num))))
        self.biases_layer4 = tf.Variable(tf.zeros([layer5Num]))
        return



if __name__ == "__main__":   
    obj= Mulilayer_HiddenRelu()
    obj.run()
