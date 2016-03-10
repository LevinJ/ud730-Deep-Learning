# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf

from A2_fullyconnected.main import SoftmaxwithSGD
from A2_fullyconnected.p1_relulayer import HiddenRelu

class SoftmaxwithSGD_L2(SoftmaxwithSGD):
    def __init__(self):
        SoftmaxwithSGD.__init__(self)
        return
    
    def addRegularization(self):
        # L2 regularization for the fully connected parameters.
        Beta  = 0.003
        regularizers = tf.nn.l2_loss(self.weights)
        # Add the regularization term to the loss.
        self.loss += Beta * regularizers
        return

class HiddenRelu_L2(HiddenRelu):
    def __init__(self):
        HiddenRelu.__init__(self)
        return
    
    def addRegularization(self):
#         Beta = 5e-4 
        Beta  = 0.001
        # L2 regularization for the fully connected parameters.
        regularizers = tf.nn.l2_loss(self.weights_layer1) + tf.nn.l2_loss(self.weights_layer2)
        # Add the regularization term to the loss.
        self.loss += Beta * regularizers
        return





if __name__ == "__main__":   
    obj= SoftmaxwithSGD_L2()
    obj.run()
    obj = HiddenRelu_L2()
    obj.run()
    





