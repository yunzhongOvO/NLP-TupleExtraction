# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 01:01:13 2019

@author: DJF
"""
from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""#"1, 2, 3, 4"

import numpy as np
import tensorflow as tf

##config
n_features = 250 #the input length of a sentence
n_classes  = 3
vocabulary_size = 2841 #len(word2idx)+1
embedding_size = 100
n_hidden_units = 100
batch_size = 50
learning_rate = 0.003
hidden_size = 150
# dropout = 0.5

class MyNetwork():
    def __init__(self):
        #input: X, y(class)
        self.X = tf.placeholder(tf.int32, [None, n_features], name='X')
        self.y = tf.placeholder(tf.int32, [None, n_features], name='y')
        #variables
        self.W = tf.Variable(
                initial_value=tf.constant(0.0, shape=[2*n_hidden_units, n_classes]),
                dtype=tf.float32, name='weight')
        self.b = tf.Variable(
                initial_value=tf.constant(0.0, shape=[n_classes]),
                dtype=tf.float32, name='bais')
        self.embeddings = tf.Variable(
                initial_value=tf.random_uniform([vocabulary_size, embedding_size], minval=-1, maxval=1),
                dtype=tf.float32, name='embeddings')
        self.is_train = False
    
    def softmax(self, logits):
        return tf.nn.softmax(logits, -1)
    
    def get_network(self):
        valid_embeddings = tf.nn.embedding_lookup(self.embeddings, self.X)
        #LSTM cell
        if self.is_train is False:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units)
        else:
            cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, reuse=True)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units, reuse=True)
        #initial state
        init_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
        init_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
        #BiLSTM
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, valid_embeddings,
                initial_state_fw=init_state_fw, initial_state_bw=init_state_bw)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw, output_bw], axis=-1)
        matri_output = tf.reshape(output, [-1, 2*n_hidden_units])
        #hidden = tf.matmul(matri_output, self.W) + self.b
        #hidden = tf.reshape(hidden, [-1, n_features, n_classes])
        #self.y_pred = tf.argmax(self.softmax(hidden), axis=-1)
        #return self.y_pred, hidden
        matri_unary_score = tf.matmul(matri_output, self.W) + self.b
        unary_score = tf.reshape(matri_unary_score, [batch_size, n_features, n_classes])
        sequence_len = np.full(batch_size, n_features, dtype=np.int32)
        seq_length = tf.constant(sequence_len, dtype=tf.int32)
        if self.is_train is False:
            self.is_train = True
            with tf.variable_scope('crf') as scope:
                hidden, transition_params = tf.contrib.crf.crf_log_likelihood(unary_score, self.y, seq_length)
        else:
            with tf.variable_scope('crf') as scope:
                scope.reuse_variables()
                hidden, transition_params = tf.contrib.crf.crf_log_likelihood(unary_score, self.y, seq_length)
        self.y_pred, best_score = tf.contrib.crf.crf_decode(unary_score, transition_params, seq_length)
        return self.y_pred, hidden
    
    def get_loss(self, hidden):
        #onehot_labels = tf.one_hot(self.y, depth=n_classes)
        #self.loss = tf.losses.softmax_cross_entropy(onehot_labels, hidden)
        #return tf.reduce_mean(self.loss)
        return tf.reduce_mean(-hidden)
    
    def generate_feed_dict(self, data_x, data_y=None):
        feed_dict = {}
        feed_dict[self.X] = data_x
        feed_dict[self.y] = data_y
        return feed_dict

#optimizer
def get_optimizer(learning_rate, optim=None):
    # get optimizer for training, AdamOptimizer as default
    if optim == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optim == 'adadelta':
        return tf.train.AdadeltaOptimizer()
    else:  # if optim == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

class MyModel():
    def __init__(self):
        self.saver = None
        self.sess = tf.Session()
        self.network = MyNetwork()
        self.train_output = self.get_model_train() #y_pred, loss
        self.test_output  = self.get_model_test() #y_pred
        self.optimizer = get_optimizer(learning_rate, 'adam').minimize(self.train_output[-1])
        self.sess.run(tf.global_variables_initializer())
        self.init_saver()
    
    def errors(self, y_pred, y_truth=None):
        if y_truth is None:
            y_truth = self.y
            not_equal_counts = tf.abs(y_pred - y_truth)
            return tf.reduce_mean(not_equal_counts)
        else:
            not_equal_counts = abs(y_pred - y_truth)
            return np.mean(not_equal_counts)
    
    def get_model_train(self):
        with tf.name_scope('train'):
            y_pred, hidden = self.network.get_network()
            loss = self.network.get_loss(hidden)
            return y_pred, loss
    
    def get_model_test(self):
        with tf.name_scope('test'):
            y_pred, hidden = self.network.get_network()
            return y_pred
    
    def init_saver(self, var_list=None):
        if self.saver is not None:
            return
        self.saver = tf.train.Saver(
            var_list = tf.global_variables(),
            reshape=True,
            sharded=False,
            restore_sequentially=True,
            write_version=tf.train.SaverDef.V2)
    
    def save_model(self, path):
        if self.saver is None:
            self.init_saver()
        # save session in file
        self.saver.save(self.sess, save_path=path)
    
    def load_model(self, path):
        if self.saver is None:
            self.init_saver()
        self.saver.restore(self.session, path)
    
    def call_model(self, data_x, data_y=None, mode='train'):
        # generate data for placeholder
        if mode == 'test':
            ret = self.sess.run(  # return y_pred
                self.test_output,
                feed_dict=self.network.generate_feed_dict(data_x, data_y))
        else:  # mode == 'train'
            _, ret = self.sess.run(  # return y_pred, loss
                [self.optimizer, self.train_output], 
                feed_dict=self.network.generate_feed_dict(data_x, data_y))
        return ret

if __name__=="__main__":
    # call model class for a instance
    model = MyModel()

    path = 'C:/Users/25535/python_work/AI/NLP-TupleExtraction/'
    #path = './'
    train_set_x = np.load(path+'train_data_X.npy')
    train_set_y = np.load(path+'train_data_Y.npy')

    epoch = 0
    n_epochs = 20
    print("Now training.")
    while epoch < n_epochs:
        # draw a figure every 'draw_freq' times
        # print error/cost per epoch
        for i in range(int(len(train_set_x)/batch_size)):
            start = i*batch_size
            end = start+batch_size
            train_pred, loss = model.call_model(
                    train_set_x[start:end], train_set_y[start:end], 'train')
            train_error = model.errors(
                    y_pred=train_pred, y_truth=train_set_y[start:end])
        
        #for i in range(int(len(test_set_x)/batch _size)):
        #    start = i*batch_size
        #    end = start+batch_size
        #    test_pred = model.call_model(
        #            test_set_x[start:end],  test_set_y[start:end], 'test')
        #    if i == 0:
        #        test_y = test_pred
        #    else:
        #        test_y = np.vstack([test_y, test_pred])
        #test_error = model.errors(
        #    y_pred=test_y, y_truth=test_set_y)
        print ("epoch is %d, train error %f" % (epoch, train_error))
        epoch += 1
    model.save_model(path+'model.ckpt')