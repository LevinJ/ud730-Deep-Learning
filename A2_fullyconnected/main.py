 
 
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
# import tensorflow as tf
 
 
from A1_notmnistdataset.p5_findduplication import DataExploration
 
 
class FullyConnected(DataExploration):
    def __init__(self):
        DataExploration.__init__(self)
        self.__dispDataDim()
        self.reformat()
        self.__dispDataDim()
#         self.reshapeData()
        return
    def reformat(self):
        self.train_dataset, self.train_labels = self.reformatDataset(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformatDataset(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformatDataset(self.test_dataset, self.test_labels)
        return
    def reformatDataset(self, dataset, labels):
        image_size = 28
        num_labels = 10
        dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
    
    def __dispDataDim(self):
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)
        return
    def run(self):
        
        return
 
if __name__ == "__main__":   
    obj= FullyConnected()
    obj.run()
# import tensorflow as tf
# 
# 
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))
# a = tf.constant(10)
# b = tf.constant(32)
# print(sess.run(a + b))





