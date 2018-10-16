import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models

data_path=os.getcwd()+'/Data/'
data,label=utils.load_data(data_path,step_size=5,window_size=20)
#print(len(data))
#print(len(label))
#print(data[0].shape)
#print(label[0].shape)
model1=models.Model1(data[0].shape, hidden_units=30)
model=model1.build_model()
model=utils.train_model(model,data[:4],label[:4], batch_s=32, epo=1)
predict=utils.predict(model,data[5])
print(utils.correlation(label[5], predict))
utils.save_model(model, 'model1.h5')