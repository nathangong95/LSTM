from keras.models import Model
from keras.layers import Embedding
from keras.utils import to_categorical, plot_model
from keras.layers import LSTM
from keras.layers import Dense, Dropout, Input
import os
import numpy as np
""" 
"""
class LSTM_model1:
    def __init__(self,step_size=5,batch_size=32,window_size=20,hidden_units=30,epoch=5):
        """ Constructor
        Args:
            step_size (int): space that the slide window takes
            batch_size (int): training batch_size
            window_size (int): the size of the slide window

        """
        self.train_data=None
        self.train_label=None
        self.test_data=None
        self.test_label=None
        self.model=None
        self.step_size=step_size
        self.batch_size=batch_size
        self.window_size=window_size
        self.hidden_units=hidden_units
        self.epoch=epoch

    def load_14Ddata(self, data_path):
        filelist=os.listdir(data_path)
        train_data=[]
        train_label=[]
        filelist.sort()
        for file in filelist:
            data=np.genfromtxt(data_path+file,delimiter=',')
            a,s=data.shape
            train_data.append(np.reshape(data[:a-2,:].T,(s,a-2)))
            train_label.append(to_categorical(np.reshape(data[a-2,:],(s,))))
        # deal with the step size here
        new_train_data,new_train_label=self.stack_data(train_data,train_label)
        length_folder=len(new_train_data)
        train_length=length_folder-length_folder/3
        self.train_data=new_train_data[:train_length]
        self.test_data=new_train_data[train_length:]
        self.train_label=new_train_label[:train_length]
        self.test_label=new_train_label[:train_length]
        #return new_train_data,new_train_label

    def stack_data(self,train_data,train_label):
        new_train_data=[]
        new_train_label=[]
        for train_dat, train_lab in zip(train_data,train_label):
            s,d=train_dat.shape
            new_train_dat=[]
            new_train_lab=[]
            for i in range(s):
                if i%self.step_size==0:
                    window=[]
                    for j in range(self.window_size):
                        if (i-self.window_size+j+1)>0:
                            window.append(train_dat[i-self.window_size+j+1,:])
                        else:
                            window.append(np.reshape(np.zeros((1,d)),(d,)))
                    window=np.asarray(window)
                    new_train_dat.append(window)
                    new_train_lab.append(train_lab[i])
            new_train_data.append(np.asarray(new_train_dat))
            new_train_label.append(np.asarray(new_train_lab))
        return new_train_data,new_train_label

    def build_model1(self):
        _,_,d=self.train_data[0].shape
        data=Input(shape=(self.window_size,d))
        lstm=LSTM(self.hidden_units)(data)
        output=Dense(4,activation='sigmoid')(lstm)
        model=Model(inputs=data, outputs=output)
        plot_model(model, to_file='lstm_model.png')
        model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
        print(model.summary())
        plot_model(model, to_file='lstm_model.png')
        self.model=model

    def train_model1(self):
        for train_dat, train_lab in zip(self.train_data, self.train_label):
            self.model.fit(train_dat,train_lab,batch_size=self.batch_size,epochs=self.epoch)




        
