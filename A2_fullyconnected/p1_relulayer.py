from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.main import SoftmaxwithSGD
import utility.logger_tool
import logging




class HiddenRelu(SoftmaxwithSGD):  
    def __init__(self):
        SoftmaxwithSGD.__init__(self)
        self.keep_prob = 1
        return 

    def getTempModleOutput_forTrain(self, dataset):
        return self.getTempModelOutput(dataset, self.keep_prob)
        
    def getTempModelOutput(self, dataset, keep_prob):
        z_layer2 = tf.matmul(dataset, self.weights_layer1 ) + self.biases_layer1
#         a_layer2 = tf.nn.relu(z_layer2)
        a_layer2 = tf.nn.dropout(tf.nn.relu(z_layer2), keep_prob)

        
        z_layer3 = tf.matmul(a_layer2, self.weights_layer2 ) + self.biases_layer2
        return z_layer3
    def getTempModleOutput_forTest(self, dataset):    
        return self.getTempModelOutput(dataset, 1)

    def setupVariables(self):
        layer2Num = 1024
        layer3Num = self.num_labels
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        self.weights_layer1 = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, layer2Num]))
        self.biases_layer1 = tf.Variable(tf.zeros([layer2Num]))
        
        self.weights_layer2 = tf.Variable(tf.truncated_normal([layer2Num, layer3Num]))
        self.biases_layer2 = tf.Variable(tf.zeros([layer3Num]))
        return
    


if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/HiddenRelu.log',filemode='w',level=logging.DEBUG)
    obj= HiddenRelu()
    obj.run()



