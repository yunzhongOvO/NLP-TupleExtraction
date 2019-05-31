# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 21:19:07 2019

@author: DJF
"""
import json
import numpy as np

#load data
sentence_list_fname = 'C:/Users/25535/python_work/AI/NLP-TupleExtraction/assignment_training_data_word_segment.json'
sentence_list = json.load(open(sentence_list_fname , 'r'))
sentence_y = np.load('y_pred.npy')
y_train = np.load('data_Y.npy')
y_truth = np.load('test_data_Y.npy')
y_pred = np.load('y_pred.npy')

#match for sentence[k]
def RuleMatch(sentence, y):
    #find valid info
    times, attributes, values = [], [], []
    for i in sentence['times']:
        if y[i] == 1:
            times.append(i)
    for i in sentence['attributes']:
        if y[i] == 1:
            attributes.append(i)
    for i in sentence['values']:
        if y[i] == 1:
            values.append(i)
    X = np.hstack([times, attributes, values])
    Xt = np.hstack([times, attributes, values])
    X.sort()
    for i in range(len(X)):
        if X[i] in times:
            Xt[i] = 0
        elif X[i] in attributes:
            Xt[i] = 1
        elif X[i] in values:
            Xt[i] = 2
    
    result = []
    i = 0
    while True:
        ctime = 0
        t, a, v = [], [], []
        while i < len(Xt):
            if Xt[i] == 0 and ctime == 0:
                while(i < len(Xt) and Xt[i] == 0):
                    t.append(X[i])
                    i = i + 1
                ctime = 1
            elif Xt[i] == 1:
                a.append(X[i])
                i = i + 1
            elif Xt[i] == 2:
                v.append(X[i])
                i = i + 1
            else:
                if len(a) == 1 and len(t)*len(a) == len(v):
                    for j in range(len(v)):
                        result.append([t[j%len(t)], a[j%len(a)], v[j%len(v)]])
                elif len(t)*len(a) == len(v):
                    for j in range(len(a)):
                        for k in range(len(t)):
                            result.append([t[k], a[j], v[j*len(t)+k]])
                break
        if i == len(Xt):
            if len(a) == 1 and len(t)*len(a) == len(v):
                for j in range(len(v)):
                    result.append([t[j%len(t)], a[j%len(a)], v[j%len(v)]])
            elif len(t)*len(a) == len(v):
                for j in range(len(a)):
                    for k in range(len(t)):
                        result.append([t[k], a[j], v[j*len(t)+k]])
            break
    return result, Xt, X

TP = FP = FN = 0
for i in range(2100, 3000):
    result, Xt, X = RuleMatch(sentence_list[i], y_pred[i-2100])
    for j in range(len(result)):
        if result[j] in sentence_list[i]['results']:
            TP = TP + 1
        else:
            FP = FP + 1
    for j in range(len(sentence_list[i]['results'])):
        if sentence_list[i]['results'][j] not in result:
            FN = FN + 1
P = TP/(TP+FP)
R = TP/(TP+FN)
F = 2*P*R/(P+R)
print(P, R, F)