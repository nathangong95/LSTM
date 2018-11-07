from keras.models import Model
from keras.layers import Embedding
from keras.utils import to_categorical, plot_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Input
import os
import numpy as np

class Model1:
	def __init__(self, data_shape, hidden_units=30):
		""" data: n*window_size*d
			label: n*4
		"""
		self.data_shape=data_shape
		self.hidden_units=hidden_units

	def build_model(self):
		(_,window_size,d)=self.data_shape
		data=Input(shape=(window_size,d))
		lstm=LSTM(self.hidden_units)(data)
		output=Dense(4,activation='sigmoid')(lstm)
		model=Model(inputs=data, outputs=output)
		#plot_model(model, to_file='lstm_model.png')
		model.compile(loss='categorical_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])
		print(model.summary())
		return model
class onestepModel:
	def __init__(self, data_shape, hidden_units=30):
		self.data_shape=data_shape
		self.hidden_units=hidden_units
	def build_model(self):
		(_,window_size,d)=self.data_shape
		data=Input(shape=(window_size,d))
		lstm=LSTM(self.hidden_units)(data)
		output=Dense(d,activation='linear')(lstm)
		model=Model(inputs=data, outputs=output)
		plot_model(model, to_file='onse_step_prediction.png')
		model.compile(loss='mean_squared_error',
			optimizer='adadelta',
			metrics=['mse'])
		print(model.summary())
		return model




