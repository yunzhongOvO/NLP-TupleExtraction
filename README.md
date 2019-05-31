# NLP-TupleExtraction
> Use biLSTM model to extract correct triples from a sentence

## Problem Defination
> Given a sentence that describes the financial situation of a company in the past few years.

#### Technical routes
- Methods based on rules
- Methods based on training
- Both above

## Example
> From the sentence as
>   "2013年度、2014年度和2015年1-6月，发行人应付账款余额分别为2306万元、635万元和70118万元。"

We extract some entities, include **time**, **attribute**, and **value**.

- **times**:  2013年度、2014年度、2015年1-6月
- **attributes**: 应收账款 (attributes might be not complete)
- **values**: 2306万元、635万元、70118万元

> We define a triple is consisted of three ordered components: [time, attribute, value]. If the sentence states that at time t, the value of attribute a was v, we say triple [t, a, v] is a correct triple.

All correct triples in current sentence:

- 【2013年度、应付账款、2306万元】
- 【2014年度、应付账款、635万元】
- 【2015年1-6月、应付账款、70118万元】



## Import packages

```python
# coding: utf-8
# ============================================================================
#   Copyright (C) 2017 All rights reserved.
#
#   filename : Assignment_template.py
#   author   : chendian / okcd00@qq.com
#   date     : 2018-11-15
#   desc     : Tensorflow Tuple Extraction Tutorial
# ============================================================================

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["CUDA_VISIBLE_DEVICES"] = "2" (if you want to use GPU2)

import sys
import math
import json
import pickle
import numpy as np
import tensorflow as tf
```
## Loading training data and testing data
> 使用包含标注信息的训练数据来Train模型参数， 在测试数据上预测出结果用于打分

### 数据格式
**introduction for keys**
```
# Every string is Unicode string.
{
    "sentenceId": 'unique id',
    "sentence": 'A Unicode string',
    "words": 'A list, that is the result of word segmentation of the sentence',
    "indexes": 'convert each word in “words” to an index in vocabulary,\ 
                notice that all times are converted to the same index,\ 
                same as attributes and values',
    "times": 'which words might representing time',
    "attributes": 'which words might representing attribute',
    "values": 'which words might representing value', 
    "results": 'correct triples in this sentence that can be composed by words in “times”, “attributes” and “values”.',
}
```

**Train dataset json file**

```python
[
    {
        "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
        "sentence": "2013年度、2014年度、2015年度，公司融资租赁业务收入分别为58,821.17万元、\
                     104,388.84万元和147,579.60万元，分别占发行人营业收入的59.21%、66.59%和66.78%。",
        "words": [
            "2013年度",  "、",  "2014年度",  "、",  "2015年度",  "，",  "公司",  "融资",  "租赁",  "业务",  "收入",
            "分别",  "为",  "58,821.17万元",  "、",  "104,388.84万元",  "和",  "147,579.60万元",  "，",  "分别", 
            "占",  "发行",  "人",  "营业收入",  "的",  "59.21%",  "、",  "66.59%",  "和",  "66.78%",  "。"
        ],
        "indexes": [
            0, 6, 0, 6, 0, 7, 13, 104, 146, 33, 1, 11, 8, 2, 6, 
            2, 9, 2, 7, 11, 14, 17, 18, 1, 12, 2, 6, 2, 9, 2, 10 
        ],  
        "times": [0, 2, 4 ],      # that is ["2013年度", "2014年度", "2015年度"]
        "attributes": [23, 10 ],  # that is ["收入", "营业收入"]
        "values": [13, 15, 17, 25, 27, 29 ], 
        # that is ["58,821.17万元", "104,388.84万元", "147,579.60万元", "59.21%", "66.59%", "66.78%"]
        "results": [
            [0, 10, 13 ],  # ["2013年度", "收入", "58,821.17万元"]
            [2, 10, 15 ],  # ["2014年度", "收入", " 104,388.84万元"]
            [4, 10, 17 ]   # ["2015年度", "收入", " 147,579.60万元"]
        ], 
    },
    { ... }, 
    ...
]
```

**Test dataset json file**

```python
[
    {
        "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
        "indexes": [
            0, 6, 0, 6, 0, 7, 13, 104, 146, 33, 1, 11, 8, 2, 6, 
            2, 9, 2, 7, 11, 14, 17, 18, 1, 12, 2, 6, 2, 9, 2, 10 
        ],  
        "times": [0, 2, 4 ],  
        "attributes": [23, 10 ],
        "values": [13, 15, 17, 25, 27, 29 ],
    }, 
    { ... }, 
    ...
]
```

**Your output json file**

```python
[
    {
        "sentenceId": "eadf4c4d7eaa6cb767fa6d8c02555f5-eb85e9fb6ec57b2dd9ba53a8cc4b1625b18",
        "results": [
            [0, 10, 13 ],
            [4, 10, 17 ], 
            [2, 10, 15 ]],
    },
    { ... }, 
    ...
]
```

```python
# How to load training data in Python:
import json
sentence_list_fname = 'xxx'  # ‘assignment_training_data_word_segment.json’ here
sentence_list = json.load(open(sentence_list_fname , ‘r’))

# How to load vocabulary in Python:

# import pickle (in python 3.x)
import cPickle as pickle # (in python 2.x)

voc_dict_fname = 'xxx'  # provided file is ‘voc.pkl’ here
voc_dict = pickle.load(open(voc_dict_fname, ‘rb’))
idx2word, word2idx = voc_dict[‘idx2word’, ‘word2idx’] 
# idx2word[index] is a word
# word2idx[word] is an index
```

## Define network
> 定义你自己的网络与计算图，多思考如何构建网络才能让模型参数更好地学到 Triple Matching 的规律。
HINT: 不仅仅是网络，数据的预处理或是后处理也可以纳入考虑的范畴

```python
class MyNetwork():
    def __init__(self, other_params=None):
        # Define your placeholders with type, size and name.
        """
        self.X = tf.placeholder(tf.float32, [None, n_features], name='X')
        self.y = tf.placeholder(tf.int32, [None, n_classes], name='y')
        """
        self.init_variables(n_features, n_classes)
        
    def init_variables(self, n_features, n_classes):
        # Define your variables
        # HINT: or you can directly use existed functions.
        """
        self.W = tf.Variable(
            initial_value=tf.constant(0.0, shape=[n_features, n_classes]),
            dtype=tf.float32, name='weight')
        self.b = tf.Variable(
            initial_value=tf.constant(0.0, shape=[n_classes]),
            dtype=tf.float32, name='bias')
        """
        pass
    
    def sigmoid(self, logits):
        return tf.nn.sigmoid(hidden)
    
    def softmax(self, logits):
        # softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)
        return tf.nn.softmax(logits, -1)
    
    def prob_layer(self, hidden, mask=None, expand_dim=False):
        # calculate probability from hidden
        prob = self.sigmoid(hidden) * mask
        if expand_dim:
            # twins_prob = tf.concat([prob, 1.0 - prob], -1)
            return tf.expand_dims(prob, -1)
        return prob
    
    def get_network(self):
        # hidden = how_to_get_my_hidden_value()
        # probs = self.prob_layer(hidden)
        # self.y_pred = how_to_get_predict_value(probs)
        # return self.y_pred, hidden
        pass
    
    def get_loss(self, hidden):
        self.loss = 0.0
        # my_loss_function = ...
        # labels, logits = ..., ...
        # self.loss = my_loss_function(labels, logits)
        return tf.reduce_mean(self.loss)
    
    def generate_feed_dict(self, data):
        feed_dict = {}
        # feed_dict[self.X] = data['data_x']
        # feed_dict[self.y] = data['data_y']
        return feed_dict
```

## Define optimizer
> 因为深度学习常见的是对于梯度的优化，也就是说，优化器最后其实就是各种对于梯度下降算法的优化。

常见的优化器有 SGD，RMSprop，Adagrad，Adadelta，Adam 等，
下面给出各个Optimizer的初始化示例以供选择，
此处实例中默认使用的是 Adam（也是目前最为广泛使用的一个），
大多数机器学习的任务就是最小化Loss，在定义好Loss的情况下，后面的工作就交给优化器处理即可.

```python
def get_optimizer(learning_rate, optim=None):
    # get optimizer for training, AdamOptimizer as default
    optim = self.options.get('optimizer', 'N/A')
    if optim == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif optim == 'rmsprop':
        return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    elif optim == 'adadelta':
        return tf.train.AdadeltaOptimizer()
    else:  # if optim == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
```

```python
class MyModel():
    self.sess = tf.Session()
    self.network = MyNetwork()  # Classifier
    self.train_output = self.get_model_train()  # y_pred, loss
    self.test_output = self.get_model_test()  # y_pred
    
    self.optimizer = get_optimizer(learning_rate, 'adam').minimize(self.train_output[-1])
    self.sess.run(tf.global_variables_initializer())
    self.init_saver()
    
    def errors(self, y_pred, y_truth=None):
        err = 0.0
        # err = calculate_errors(y_pred, y_truth)
        return err
    
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
    
    def call_model(self, data, mode='train'):
        # generate data for placeholder
        if mode == 'test':
            ret = self.sess.run(  # return y_pred
                test_output,
                feed_dict=self.network.gen_input(data))
        else:  # mode == 'train'
            _, ret = self.sess.run(  # return y_pred, loss
                [optimizer, train_output], 
                feed_dict=self.network.gen_input(data))
        return ret
```

## Something about training and cross validation

When you want to evaluate your model and choose hyper-parameters,
cross validation(CV) is usually useful.

**K-fold CV**:

Split all data you have into k folds.

If you set a hyper-parameter, to evaluate your model, you should do something like this:

```
for i in range(k):  
    model = # train on all folds except the ith fold.  
    errors[i] = # model prediction error on the ith fold (use measures like F1, precision, cost)  
error = mean(errors)
```

And you can try many hyper-parameter settings and choose the best one.

## Start Training
> 对于各自的模型，选择与处理输入数据（data），设置参数，交由模型进行训练
保存效果最好的模型参数，确认可以正确的进行读取与预测

```python
if __name__=="__main__":
    # call model class for a instance
    model = MyModel()

    # train_set_x, train_set_y, test_set_x,  test_set_y = ...
    # HINT: X-Fold Cross Validation.
    
    epoch = 0
    print("Now training.")
    while epoch < n_epochs:
        # draw a figure every 'draw_freq' times
    
        # print error/cost per epoch
        train_pred, loss = model.call_model(
            train_set_x, train_set_y, 'train')
        train_error = model.errors(
            y_pred=train_pred, y_truth=train_set_y)
    
        test_pred = model.call_model(
            test_set_x,  test_set_y, 'test')
        test_error = model.errors(
            y_pred=test_pred, y_truth=test_set_y)
    
        print ("epoch is %d, train error %f, test error %f" % (
            epoch, train_error, test_error))
        epoch += 1
```

## Answer Checking
> Source Code 选自2017年人工智能基础课 韩柔刚 同学，
  作简要格式修改后放在这里，便于各位同学测试自己模型效果使用，
  便于测试或用于为不擅长写error计算函数的同学作简要替代

```python
import json
import numpy

info = json.load(open("data.json", 'r'))  # original data json
answers = json.load(open("output.json", 'r'))  # your prediction

n_info = len(info)
n_answers = len(answers)

prec, recall = 0., 0.
post, negt, posf = 0, 0, 0

n = n_answers

if n_info == n_answers:
    print ("No missing or flowing problem")
else:
    if n_info < n_answers:
        n = n_info
        print ("flowing problems")
    else :
        n = n_answers
        print ("missing problems")

for i in numpy.arange(0, n):
    for r in info[i]["results"]:
        if (r in answers[i]["results"]):
            post = post + 1
        else :
            posf = posf + 1
    for r in answers[i]["results"]:
        if (r not in info[i]["results"]):
            negt = negt + 1

recall = post * 1. / (post + posf)
prec   = post * 1. / (post + negt)
correct = post * 1. / (post + posf + negt)
f1 = (prec * recall) * 2. / (prec + recall)

print("recall:%.2lf prec:%.2lf correct:%.2lf F1:%.2lf" % (recall, prec, correct, f1))
```

## Self-learning material
- Python @ COGS18
  - Introduction to Python (COGS18) is a course offered by the Department of Cognitive Science of UC San Diego, taught by Tom Donoghue. It is a hands-on programming course, focused on teaching students in Cognitive Science and related disciplines an introduction on how to productively use Python.
  - https://cogs18.github.io/intro/
- More Python Packages @ 机器之心
  - Python 成功和受欢迎的原因之一是存在强大的库，这些库使 Python 极具创造力且运行快速。然而，使用 Pandas、Scikit-learn、Matplotlib 等常见库在解决一些特殊的数据问题时可能并不实用，本文介绍的这些非常见库可能更有帮助。
  - https://mp.weixin.qq.com/s/cLCfdaCMub0xEtbgv4b-cQ
- Tensorflow @ Google
  - TensorFlow 最近提供了官方中文版教程（Tutorials）和中文版指南（Guide）。其中教程主要介绍了 TensorFlow 的基本概念，以及各种基础模型的简单实现方法，这些模型基本上都是用 Keras 等易于理解的高阶 API 完成。而指南则深入介绍了 TensorFlow 的工作原理，包括高阶 API、Estimator、低阶 API 和 TensorBoard 等。 ——by 机器之心
  - https://tensorflow.google.cn/tutorials/?hl=zh-cn

## Random grouping
```python
import numpy as np
# for random groups generating
​```python
a = np.arange(1, 51)
np.random.shuffle(a)
a
```
```txt
OUTPUT:
array([42, 27, 41, 13, 46, 47, 23, 10, 17, 19, 37, 45, 31, 26, 36, 12,  2,
       32, 43, 48,  5, 18, 21,  4, 22, 40, 25, 39, 33,  1, 49, 28, 38,  3,
       15,  6, 24, 14, 35,  7, 16, 11, 34, 29,  9, 50,  8, 30, 44, 20])
```

