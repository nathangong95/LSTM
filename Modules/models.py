from keras.models import Model
from keras.layers import Embedding
from keras.utils import to_categorical, plot_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Input
import os
import numpy as np

class Model1:
	def __init__(self, data, label, hidden_units=30, epochs=3, batch_size=32):
		""" data: n*window_size*d
			label: n*4
		"""
		self.train_data=data
		self.train_label=label
		self.hidden_units=hidden_units
		self.epoch=epochs
		self.batch_size=batch_size
		self.model=None

	def build_model(self):
		_,window_size,d=self.train_data[0].shape
		data=Input(shape=(window_size,d))
		lstm=LSTM(self.hidden_units)(data)
		output=Dense(4,activation='sigmoid')(lstm)
		model=Model(inputs=data, outputs=output)
		#plot_model(model, to_file='lstm_model.png')
		model.compile(loss='categorical_crossentropy',
			optimizer='rmsprop',
			metrics=['accuracy'])
		print(model.summary())
		self.model=model
	def train_model1(self):
		for train_dat, train_lab in zip(self.train_data, self.train_label):
			self.model.fit(train_dat,train_lab,batch_size=self.batch_size,epochs=self.epoch)
	def predict(self, test_data):
		return self.model.predict(test_data)
	def save_model(self):
		self.model.save('model1.h5')




