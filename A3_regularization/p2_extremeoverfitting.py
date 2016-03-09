# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from A2_fullyconnected.p1_relulayer import HiddenRelu


class ExtremeOverfittingDemo(HiddenRelu):
    def __init__(self):
        HiddenRelu.__init__(self)
        return
    def setBatchSize(self):
        self.batch_size = 100
        return
    





if __name__ == "__main__":   
    obj= ExtremeOverfittingDemo()
    obj.run()





