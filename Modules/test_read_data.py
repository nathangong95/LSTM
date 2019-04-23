from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense, Input
from keras import optimizers
import numpy as np

window_size = 2
d = 4
hidden_unit = 3
data = Input(shape=(window_size, d))
lstm, state_h, state_c = LSTM(hidden_unit, return_sequences=True, return_state=True)(data)
output = Dense(d, activation='linear')(lstm)
model = Model(inputs=data, outputs=output)
sgd = optimizers.SGD(lr=0.001)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['mse'])


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

    ft = hard_sigmoid(W_f.T.dot(ht)+U_f.T.dot(data)+b_f.reshape((h, 1)))
    it = hard_sigmoid(W_i.T.dot(ht) + U_i.T.dot(data) + b_i.reshape((h, 1)))
    ct_bar = np.tanh(W_c.T.dot(ht) + U_c.T.dot(data) + b_c.reshape((h, 1)))
    ct_ = ft*ct + it*ct_bar
    ot = hard_sigmoid(W_o.T.dot(ht) + U_o.T.dot(data) + b_o.reshape((h, 1)))
    ht_ = ot*np.tanh(ct_)
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

window_size = 2
d = 4
hidden_unit = 3
data = np.array([[1, 2, 3, 4], [3, 4, 5, 6]]).reshape((1, 2, 4))
x = model.predict(data)
ht = np.zeros((3, 1))
ct = np.zeros((3, 1))
U, W, b, Dw, Db = model.get_weights()
ht_, ct_ = custom_lstm(ht, ct, data[0,0,:].reshape((4,1)), W, U, b)
print(custom_dense(ht_, Dw, Db))
ht_, ct_ = custom_lstm(ht_, ct_, data[0,1,:].reshape((4,1)), W, U, b)
output = custom_dense(ht_, Dw, Db)
1