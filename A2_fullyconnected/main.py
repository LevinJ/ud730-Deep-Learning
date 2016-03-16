 
 
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from utility.duration import Duration
import utility.logger_tool
import logging

# from six.moves import cPickle as pickle
# from six.moves import range
 
 
from A1_notmnistdataset.p5_findduplication import DataExploration
 
 
class ReshapeDataset(DataExploration):
    def __init__(self):
        DataExploration.__init__(self)
        self.image_size = 28
        self.num_labels = 10
        self.train_subset = 10* 1000
        return
    def reformat(self):
        self.train_dataset, self.train_labels = self.reformatDataset(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformatDataset(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformatDataset(self.test_dataset, self.test_labels)
        return
    def reformatDataset(self, dataset, labels):
        dataset = dataset.reshape((-1, self.image_size * self.image_size)).astype(np.float32)
        # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
        labels = (np.arange(self.num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels
    
    def __dispDataDim(self):
        logging.debug('Training set {} {}'.format(self.train_dataset.shape, self.train_labels.shape))
        logging.debug('Validation set {} {}'.format(self.valid_dataset.shape, self.valid_labels.shape))
        logging.debug('Test set{} {}'.format(self.test_dataset.shape, self.test_labels.shape))
        return
    def run(self):
        self.train_dataset, self.train_labels = self.train_dataset[:self.train_subset], self.train_labels [:self.train_subset]
        self.reformat()
        self.__dispDataDim()
        return

class SoftmaxwithGD(ReshapeDataset):
    def __init__(self):
        ReshapeDataset.__init__(self)
        self.num_steps = 801
        self.durationtool = Duration()
        return
   
    def getInputData(self):
        self.tf_train_dataset = tf.constant(self.train_dataset)
        self.tf_train_labels = tf.constant(self.train_labels)
        return 
    def getTempModleOutput_forTest(self, dataset):
        
        
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        partialModel = tf.matmul(dataset, self.weights) + self.biases
        return partialModel
    def getTempModleOutput_forTrain(self, dataset):
        
        
        # Training computation.
        # We multiply the inputs with the weight matrix, and add biases. We compute
        # the softmax and cross-entropy (it's one operation in TensorFlow, because
        # it's very common, and it can be optimized). We take the average of this
        # cross-entropy across all training examples: that's our loss.
        return self.getTempModleOutput_forTest(dataset)
    def setupVariables(self):
        # Variables.
        # These are the parameters that we are going to be training. The weight
        # matrix will be initialized using random valued following a (truncated)
        # normal distribution. The biases get initialized to zero.
        self.weights = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
        self.biases = tf.Variable(tf.zeros([self.num_labels]))
        return
    def addRegularization(self):
        pass
        return
    def setupLossFunction(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.getTempModleOutput_forTrain(self.tf_train_dataset), self.tf_train_labels))
        self.addRegularization()
        return
    def setupOptimizer(self):
        self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)
        return
    def prepareGraph(self):
        logging.debug("prepareGraph")
#         image_size = self.image_size
#         num_labels = self.num_labels
        
        graph = tf.Graph()
        self.graph = graph
        with graph.as_default():
            # Input data.
            # Load the training, validation and test data into constants that are
            # attached to the graph.
            self.getInputData()
#             tf_train_dataset, tf_train_labels = self.getInputData()
            tf_valid_dataset = tf.constant(self.valid_dataset)
            tf_test_dataset = tf.constant(self.test_dataset)
            
            self.setupVariables()
            
            self.setupLossFunction()
            # Optimizer.
            # We are going to find the minimum of this loss using gradient descent.
            self.setupOptimizer()
            
            # Predictions for the training, validation, and test data.
            # These are not part of training, but merely here so that we can report
            # accuracy figures as we train.
            train_prediction = tf.nn.softmax(self.getTempModleOutput_forTest(self.tf_train_dataset))
            valid_prediction = tf.nn.softmax(self.getTempModleOutput_forTest(tf_valid_dataset))
            test_prediction = tf.nn.softmax(self.getTempModleOutput_forTest(tf_test_dataset))
            
            self.train_prediction = train_prediction
            self.valid_prediction= valid_prediction
            self.test_prediction = test_prediction
               
        return
    def accuracy(self, predictions, labels):
        return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
    def computeGraph(self):
        logging.debug("computeGraph")
        with tf.Session(graph=self.graph) as session:
            # This is a one-time operation which ensures the parameters get initialized as
            # we described in the graph: random weights for the matrix, zeros for the
            # biases. 
            tf.initialize_all_variables().run()
            logging.debug('Initialized')
            for step in range(self.num_steps):
                # Run the computations. We tell .run() that we want to run the optimizer,
                # and get the loss value and the training predictions returned as numpy
                # arrays.
                _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction])
                if (step % 100 == 0):
                    logging.debug('Loss at step %d: %f' % (step, l))
                    logging.debug('Training accuracy: %.1f%%' % self.accuracy(
                    predictions, self.train_labels))
                    # Calling .eval() on valid_prediction is basically like calling run(), but
                    # just to get that one numpy array. Note that it recomputes all its graph
                    # dependencies.
                    logging.debug('Validation accuracy: %.1f%%' % self.accuracy(self.valid_prediction.eval(), self.valid_labels))
        
        
        
            logging.debug('Test accuracy: %.1f%%' % self.accuracy(self.test_prediction.eval(), self.test_labels))
        return
    def run(self):
        self.durationtool.start()
        ReshapeDataset.run(self)
        self.prepareGraph()
        self.computeGraph()
        self.durationtool.end()
        return
    
class SoftmaxwithSGD(SoftmaxwithGD):  
    def __init__(self):
        self.setBatchSize()
        SoftmaxwithGD.__init__(self)
        self.num_steps = 3001
        return 
    def getInputData(self):
        self.tf_train_dataset = tf.placeholder(tf.float32,shape=(self.batch_size, self.image_size * self.image_size))
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.num_labels))
        return

    def setBatchSize(self):
        self.batch_size = 128
        return
    def computeGraph(self):
        logging.debug("computeGraph")
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            logging.debug("Initialized")
            for step in range(self.num_steps):
                # Pick an offset within the training data, which has been randomized.
                # Note: we could use better randomization across epochs.
#                 if step % 100 == 0:
#                     logging.debug("iteration {}:{}".format(step, self.num_steps))
#                 offset = (step * self.batch_size) % (self.train_labels.shape[0] - self.batch_size)
#                 # Generate a minibatch.
#                 batch_data = self.train_dataset[offset:(offset + self.batch_size), :]
#                 batch_labels = self.train_labels[offset:(offset + self.batch_size), :]
                _positions = np.random.choice(self.train_dataset.shape[0], size=self.batch_size, replace=False)
                batch_data = self.train_dataset[_positions, :]
                batch_labels = self.train_labels[_positions, :]
                # Prepare a dictionary telling the session where to feed the minibatch.
                # The key of the dictionary is the placeholder node of the graph to be fed,
                # and the value is the numpy array to feed to it.
                feed_dict = {self.tf_train_dataset : batch_data, self.tf_train_labels : batch_labels}
                _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 250 == 0):
                    logging.debug("Minibatch loss at step %d/%d: %f" % (step, self.num_steps,l))
                    logging.debug("Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels))
                    logging.debug("Validation accuracy: %.1f%%" % self.accuracy(self.valid_prediction.eval(), self.valid_labels))
            res = self.accuracy(self.test_prediction.eval(), self.test_labels)
            logging.debug("Test accuracy: %.1f%%" % res)
            logging.debug("Incorrectly labelled test sample number: {}".format(self.test_labels.shape[0] * (100- res)/float(100)))
        return
    
    
if __name__ == "__main__":   
    _=utility.logger_tool.Logger(filename='logs/SoftmaxwithSGD.log',filemode='w',level=logging.DEBUG)
#     obj= SoftmaxwithSGD()
    obj = SoftmaxwithGD()
    obj.run()





