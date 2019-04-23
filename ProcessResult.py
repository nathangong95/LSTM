import os
import sys
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd() + '/Modules/')
import utils

""" data map:
0: head x
1: head y
2: left hand x
3: left hand y
4: right hand x
5: right hand y
6: left elbow x
7: left elbow y
8: right elbow x
9: right elbow y
10: left shoulder x
11: left shoulder y
12: right shoulder x
13: right shoulder y
GT1:NY531
GT2:RCH1
GT3:RCH3
"""


def custom_lstm(ht, ct, data, W, U, b):
    """
    :param ht: previous hidden state (h, 1)
    :param ct: Previous cell state (h, 1)
    :param data: new input data (d, 1)
    :param W: parameter for ht (h, 4*h)
    :param U: parameter for data (d, 4*h)
    :param b: bias (4*h, )
    :return: current hidden state and cell state
    """
    global ht_, ct_
    h, _ = ht.shape
    W_i = W[:, :h]
    W_f = W[:, h: h * 2]
    W_c = W[:, h * 2: h * 3]
    W_o = W[:, h * 3:]

    U_i = U[:, :h]
    U_f = U[:, h: h * 2]
    U_c = U[:, h * 2: h * 3]
    U_o = U[:, h * 3:]

    b_i = b[:h]
    b_f = b[h: h * 2]
    b_c = b[h * 2: h * 3]
    b_o = b[h * 3:]

    ft = hard_sigmoid(W_f.T.dot(ht) + U_f.T.dot(data) + b_f.reshape((h, 1)))
    it = hard_sigmoid(W_i.T.dot(ht) + U_i.T.dot(data) + b_i.reshape((h, 1)))
    ct_bar = np.tanh(W_c.T.dot(ht) + U_c.T.dot(data) + b_c.reshape((h, 1)))
    ct_ = ft * ct + it * ct_bar
    ot = hard_sigmoid(W_o.T.dot(ht) + U_o.T.dot(data) + b_o.reshape((h, 1)))
    ht_ = ot * np.tanh(ct_)
    return ht_, ct_


def hard_sigmoid(x):
    output = 0.2 * x + 0.5
    output[x < -2.5] = 0
    output[x > 2.5] = 1
    return output


def custom_dense(ht, Dw, Db):
    """
    :param ht: previous hidden state (h, 1)
    :param Dw: Dense weight (h, d)
    :param Db: Dense bias (d, )
    :return: output: (d,)
    """
    h, d = Dw.shape
    return Dw.T.dot(ht).reshape((d,)) + Db


def percentage_in_range(GT, PD, r):
    rate = []
    rate_navie = []
    GT = np.reshape(GT, (GT.shape[0], 7, 2))
    PD = np.reshape(PD, (PD.shape[0], 7, 2))
    for i in range(7):
        hit = 0
        hit_navie = 0
        for j in range(1, GT.shape[0]):
            distance = ((GT[j, i, 0] - PD[j, i, 0]) ** 2 + (GT[j, i, 1] - PD[j, i, 1]) ** 2) ** 0.5
            distance_navie = ((GT[j, i, 0] - GT[j - 1, i, 0]) ** 2 + (GT[j, i, 1] - GT[j - 1, i, 1]) ** 2) ** 0.5
            if distance <= r:
                hit += 1
            if distance_navie <= r:
                hit_navie += 1
        rate.append(hit / (GT.shape[0] - 1))
        rate_navie.append(hit_navie / (GT.shape[0] - 1))
    return rate, rate_navie  # list with len 7


def plot_result(data_path, model_path):
    data = np.genfromtxt(data_path, delimiter=',')[:14, :]
    model = load_model(model_path)
    U, W, b, Dw, Db = model.get_weights()
    h, _ = W.shape
    n = data.shape[1]
    ht = np.zeros((h, 1))
    ct = np.zeros((h, 1))
    model_output = np.zeros((n, 14))
    for i in range(1, n):
        ht, ct = custom_lstm(ht, ct, (2 * data[:, i - 1] / 255 - 1).reshape((14, 1)), W, U, b)
        model_output[i, :] = (custom_dense(ht, Dw, Db).T+1)*255/2
    ground_truth = data.T
    name = ['head', 'l_hand', 'r_hand', 'l_elbow', 'r_elbow', 'l_shoulder', 'r_shoulder']
    r = np.arange(0, 30, 1)
    result = []
    for rr in r:
        result.append(percentage_in_range(ground_truth, model_output, rr))
    result = np.asarray(result)
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(r, result[:, 0, i], r, result[:, 1, i])
        plt.gca().legend(('LSTM', 'Naive'))
        plt.ylabel(name[i])
        plt.xlabel('distance')
    plt.show()


test_data_path = 'Data/NY531/test/part1.csv'
model_path = 'Result/2019414352/model.h5'
plot_result(test_data_path, model_path)
# window_size = 2
# d = 4
# hidden_unit = 3
# data = np.array([[1, 2, 3, 4], [3, 4, 5, 6]]).reshape((1, 2, 4))
# x = model.predict(data)
# ht = np.zeros((3, 1))
# ct = np.zeros((3, 1))
# U, W, b, Dw, Db = model.get_weights()
# ht_, ct_ = custom_lstm(ht, ct, data[0,0,:].reshape((4,1)), W, U, b)
# print(custom_dense(ht_, Dw, Db))
# ht_, ct_ = custom_lstm(ht_, ct_, data[0,1,:].reshape((4,1)), W, U, b)
# output = custom_dense(ht_, Dw, Db)
# 1
