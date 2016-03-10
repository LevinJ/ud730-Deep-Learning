from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.p1_relulayer import HiddenRelu


class Dropput_HiddenRelu(HiddenRelu):
    def __init__(self):
        HiddenRelu.__init__(self)
        return
    
    def setDropout(self):
        self.keep_prob = 0.5
        return




if __name__ == "__main__":   
    obj= Dropput_HiddenRelu()
    obj.run()
