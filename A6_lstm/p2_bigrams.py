# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

batch_size=64
num_unrollings=10
vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

class DataExploration(object):
    def __init__(self):
        self.creatDataset()
        return
    def maybe_download(self, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        url = 'http://mattmahoney.net/dc/'
        if not os.path.exists(filename):
            filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception(
                'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename
    def run(self):
#         print(self.char2id('a'), self.char2id('z'), self.char2id(' '), self.char2id('x'))
#         print(self.id2char(1), self.id2char(26), self.id2char(0))
        return
    def creatDataset(self):
        filename = self.maybe_download('text8.zip', 31344016)
        text = self.read_data(filename)
        print('Data size %d' % len(text))
        valid_size = 1000
        self.valid_text = text[:valid_size]
        self.train_text = text[valid_size:]
        train_size = len(self.train_text)
        print(train_size, self.train_text[:100])
        print(valid_size, self.valid_text[:100])
        self.valid_size = valid_size
        return
    def read_data(self, filename):
        f = zipfile.ZipFile(filename)
        for name in f.namelist():
            text = tf.compat.as_str(f.read(name))
        f.close()
        return text


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()
        return
    def char2id(self,char):
        if char in string.ascii_lowercase:
            return ord(char) - first_letter + 1
        elif char == ' ':
            return 0
        else:
            print('Unexpected character: %s' % char)
            return 0
    def id2char(self, dictid):
        if dictid > 0:
            return chr(dictid + first_letter - 1)
        else:
            return ' '
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            #get the word by current curosr
            batch[b, self.char2id(self._text[self._cursor[b]])] = 1.0
            #move the current cursor to next word
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for _step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches
    def characters(self, probabilities):
        """Turn a 1-hot encoding or a probability distribution over the possible
        characters back into its (most likely) character representation."""
        return [self.id2char(c) for c in np.argmax(probabilities, 1)]
    def batches2string(self, batches):
        """Convert a sequence of batches back into their (most likely) string
        representation."""
        s = [''] * batches[0].shape[0]
        for b in batches:
            s = [''.join(x) for x in zip(s, self.characters(b))]
        return s
    def run(self):
        print(self.batches2string(self.next()))
        print(self.batches2string(self.next()))
        return
    
    
    
    
class LSTMModle:
    def __init__(self):
        self.num_nodes = 64 
        self.num_steps = 7001
        self.summary_frequency = 10
        return
    def logprob(self, predictions, labels):
        """Log-probability of the true labels in a predicted batch."""
        predictions[predictions < 1e-10] = 1e-10
        return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]
    def sample_distribution(self, distribution):
        """Sample one element from a distribution assumed to be an array of normalized
        probabilities.
        """
        r = random.uniform(0, 1)
        s = 0
        for i in range(len(distribution)):
            s += distribution[i]
            if s >= r:
                return i
        return len(distribution) - 1
    def sample(self, prediction):
        """Turn a (column) prediction into 1-hot encoded samples."""
        p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
        p[0, self.sample_distribution(prediction[0])] = 1.0
        return p
    def random_distribution(self,):
        """Generate a random column of probabilities."""
        b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
        return b/np.sum(b, 1)[:,None]
    def setupParameters(self):
        # Parameters:
        num_nodes = self.num_nodes
        # Forget gate: input, previous output, and bias.
        self._fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        self._fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        self._fb = tf.Variable(tf.zeros([1, num_nodes]))
        # Input gate: input, previous output, and bias.
        self._ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        self._im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        self._ib = tf.Variable(tf.zeros([1, num_nodes]))
        # Memory cell: input, state and bias.                             
        self._cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        self._cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        self._cb = tf.Variable(tf.zeros([1, num_nodes]))
        # Output gate: input, previous output, and bias.
        self._ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
        self._om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
        self._ob = tf.Variable(tf.zeros([1, num_nodes]))
        # Variables saving state across unrollings.
        self._saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        self._saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
        # Classifier weights and biases.
        self._w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
        self._b = tf.Variable(tf.zeros([vocabulary_size]))
        return
    def setupOptimizer(self):
        global_step = tf.Variable(0)
        self.learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
        optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)
        self.optimizer = optimizer
        return
    def setupInputData(self):
        # Input data.
        self.train_data = list()
        for _ in range(num_unrollings + 1):
            self.train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
        return
    def setupRepresentationandLoss(self):
        self.setupParameters()
        self.setupInputData()
        train_inputs = self.train_data[:num_unrollings]
        train_labels = self.train_data[1:]  # labels are inputs shifted by one time step.
        # Unrolled LSTM loop.
        outputs = list()
        output = self._saved_output
        state = self._saved_state
        for i in train_inputs:
            output, state = self.setupLSTM_cell(i, output, state)
            outputs.append(output)
        # State saving across unrollings.
        with tf.control_dependencies([self._saved_output.assign(output),self._saved_state.assign(state)]):
            # Classifier.
            self.logits = tf.nn.xw_plus_b(tf.concat(0, outputs), self._w, self._b)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, tf.concat(0, train_labels)))
        return
    def setupLSTM_cell(self,i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        forget_gate = tf.sigmoid(tf.matmul(i, self._fx) + tf.matmul(o, self._fm) + self._fb)
        input_gate = tf.sigmoid(tf.matmul(i, self._ix) + tf.matmul(o, self._im) + self._ib)
        update = tf.matmul(i, self._cx) + tf.matmul(o, self._cm) + self._cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        
        output_gate = tf.sigmoid(tf.matmul(i, self._ox) + tf.matmul(o, self._om) + self._ob)
        output = output_gate * tf.tanh(state)
        
        return output, state
    def buildGraph(self):
        num_nodes = self.num_nodes
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.setupRepresentationandLoss()
            self.setupOptimizer()
            # Predictions.
            self.train_prediction = tf.nn.softmax(self.logits)
            # Sampling and validation eval: batch 1, no unrolling.
            self.sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
            saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
            saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
            
            self.reset_sample_state = tf.group(
              saved_sample_output.assign(tf.zeros([1, num_nodes])),
              saved_sample_state.assign(tf.zeros([1, num_nodes])))
            sample_output, sample_state = self.setupLSTM_cell(self.sample_input, saved_sample_output, saved_sample_state)
            
            with tf.control_dependencies([saved_sample_output.assign(sample_output),saved_sample_state.assign(sample_state)]):
                self.sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, self._w, self._b))
        return
    def parepareData(self):
        data = DataExploration()
        self.train_batches= BatchGenerator(data.train_text, batch_size, num_unrollings)
        self.valid_batches = BatchGenerator(data.valid_text, 1, 1)
        return
    def EvaluateModel_Step_Minibatch(self, step, mean_loss, lr, batches, predictions):
        if not step % self.summary_frequency == 0:
            return
        if step > 0:
            mean_loss = mean_loss / self.summary_frequency
        # The mean loss is an estimate of the loss over the last few batches.
        print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
        mean_loss = 0
        labels = np.concatenate(list(batches)[1:])
        print('Minibatch perplexity: %.2f' % float(np.exp(self.logprob(predictions, labels))))
        return
    def EvaluateModel_Step_Sample(self, step, mean_loss, lr, batches, predictions):
        if not step % (self.summary_frequency * 10) == 0:
            return
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
            feed = self.sample(self.random_distribution())
            sentence = self.train_batches.characters(feed)[0]
            self.reset_sample_state.run()
            for _ in range(79):
                prediction = self.sample_prediction.eval({self.sample_input: feed})
                feed = self.sample(prediction)
                sentence += self.train_batches.characters(feed)[0]
            print(sentence)
        print('=' * 80)
        return
    def evaluate_Final(self):
        # Measure validation set perplexity.
        self.reset_sample_state.run()
        valid_logprob = 0
        for _ in range(self.valid_size):
            b = self.valid_batches.next()
            predictions = self.sample_prediction.eval({self.sample_input: b[0]})
            valid_logprob = valid_logprob + self.logprob(predictions, b[1])
        print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / self.valid_size)))
        return
    def trainModel(self):
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            print('Initialized')
            mean_loss = 0
            for step in range(self.num_steps):
                batches = self.train_batches.next()
                feed_dict = dict()
                for i in range(num_unrollings + 1):
                    feed_dict[self.train_data[i]] = batches[i]
                _, l, predictions, lr = session.run( [self.optimizer, self.loss, self.train_prediction, self.learning_rate], feed_dict=feed_dict)
                mean_loss += l
                self.EvaluateModel_Step_Minibatch(step, mean_loss, lr, batches, predictions)
                self.EvaluateModel_Step_Sample(step, mean_loss, lr, batches, predictions)
        return
    def run(self):
        self.parepareData()
        self.buildGraph()
        self.trainModel()
        return
    
    
if __name__ == "__main__":   
#     data = DataExploration()
#     train_text = data.train_text
#     obj= BatchGenerator(train_text, batch_size, num_unrollings)
#     obj.run()
#     valid_text = data.valid_text
#     obj = BatchGenerator(valid_text, 1, 1)
#     for _step in range(100):
#         obj.run()
    obj = LSTMModle()
    obj.run()
    
    
    
    
    
    