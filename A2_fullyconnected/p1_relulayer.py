from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.main import SoftmaxwithSGD




class HiddenRelu(SoftmaxwithSGD):  
    def __init__(self):
        SoftmaxwithSGD.__init__(self)
        return 
    def getTempModleOutput(self, dataset):
        
        
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
#         res = tf.matmul(dataset, self.weights) + self.biases
#         h_layer1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        z_layer2 = tf.matmul(dataset, self.weights_layer1 ) + self.biases_layer1
        a_layer2 = tf.nn.relu(z_layer2)
        
        z_layer3 = tf.matmul(a_layer2, self.weights_layer2 ) + self.biases_layer2
        return z_layer3
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
    obj= HiddenRelu()
    obj.run()



