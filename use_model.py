import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import model1

data_path=os.getcwd()+'/Data/'
model=model1.LSTM_model1()
model.load_14Ddata(data_path)
print(len(model.train_data))
print(len(model.train_label))
print(model.train_data[0].shape)
print(model.train_label[0].shape)
model.build_model1()
model.train_model1()