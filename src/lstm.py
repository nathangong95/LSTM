from keras.models import Model
from keras.layers import Embedding
from keras.utils import to_categorical, plot_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Input
from keras import optimizers
import os
import numpy as np


class Modeltrainer:
    def __init__(self, data, window_size, hidden_units=30):
        """
            data: list of N*7*2
            label: n*4
        """

        self.window_size = window_size
        self.hidden_units = hidden_units
        self.model = None
        # n*window_size*d
        self.data = self.process_data(data, window_size)

    def build_model(self):
        data = Input(shape=(self.window_size, self.data.shape[2]))
        lstm_layer = LSTM(self.hidden_units)(data)
        output = Dense(4, activation='sigmoid')(lstm_layer)
        model = Model(inputs=data, outputs=output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
        print(model.summary())
        self.model = model

    @staticmethod
    def process_data(data, window_size):

        return data