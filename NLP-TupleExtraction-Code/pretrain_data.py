# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 00:37:54 2019

@author: DJF
"""
import json
import numpy as np

#load data
train_sentence_list_fname = 'C:/Users/25535/python_work/AI/NLP-TupleExtraction/assignment_training_data_word_segment.json'
test_sentence_list_fname = 'C:/Users/25535/python_work/AI/NLP-TupleExtraction/assignment_test_data_word_segment.json'
train_sentence_list = json.load(open(train_sentence_list_fname , 'r'))
test_sentence_list = json.load(open(test_sentence_list_fname , 'r'))

n_features = 250
def gen_train_data(sentence):
    if sentence['times'] and sentence['attributes'] and sentence['values']:
        valid_idx = []
        for i in range(len(sentence['results'])):
            if i == 0:
                valid_idx = sentence['results'][0]
            else:
                valid_idx = np.hstack([valid_idx, sentence['results'][i]])
        X = [i+1 for i in sentence['indexes']]
        y = np.zeros(len(X), dtype=int)
        for i in valid_idx:
            y[i] = 1
        X = np.hstack([X, np.zeros(n_features-len(X), dtype=int)])
        y = np.hstack([y, np.zeros(n_features-len(y), dtype=int)+[2]])
    else:
        X = None
        y = None
    return X, y

def gen_test_data(sentence):
    if sentence['times'] and sentence['attributes'] and sentence['values']:
        X = [i+1 for i in sentence['indexes']]
        X = np.hstack([X, np.zeros(n_features-len(X), dtype=int)])
    else:
        X = None
    return X

for i in range(len(train_sentence_list)):
    Xt, yt = gen_train_data(train_sentence_list[i])
    if Xt is None or yt is None:
        continue
    if i == 0:
        X = Xt
        y = yt
    else:
        X = np.vstack([X, Xt])
        y = np.vstack([y, yt])

np.save('train_data_X.npy', X)
np.save('train_data_Y.npy', y)

for i in range(len(test_sentence_list)):
    Xt = gen_test_data(test_sentence_list[i])
    if Xt is None:
        continue
    if i == 0:
        X = Xt
    else:
        X = np.vstack([X, Xt])
np.save('test_data_X.npy', X)