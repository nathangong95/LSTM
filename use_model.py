import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models

data_path=os.getcwd()+'/Data/'
data,label=utils.load_data(data_path)
print(len(data))
print(len(label))
print(data[0].shape)
print(label[0].shape)
model1=models.Model1(data[:4],label[:4])
model1.build_model()
model1.train_model1()
print(model1.predict(data[5]))
model1.save_model()