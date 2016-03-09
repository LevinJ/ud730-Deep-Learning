from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.p1_relulayer import HiddenRelu


class Dropput_HiddenRelu(HiddenRelu):
    def __init__(self):
        HiddenRelu.__init__(self)
        return
    
    def getTempModleOutput_forTrain(self, dataset):
        keep_prob = 0.5
        
        
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        z_layer2 = tf.matmul(dataset, self.weights_layer1 ) + self.biases_layer1
        a_layer2 = tf.nn.relu(z_layer2)
        a_layer2 = tf.nn.dropout(a_layer2, keep_prob)
        
        z_layer3 = tf.matmul(a_layer2, self.weights_layer2 ) + self.biases_layer2
        return z_layer3




if __name__ == "__main__":   
    obj= Dropput_HiddenRelu()
    obj.run()
