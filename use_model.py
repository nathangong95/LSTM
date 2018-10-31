import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models

data_path=os.getcwd()+'/Data/'
data,label=utils.load_data(data_path,step_size=5,window_size=20)
model1=models.Model1(data[0].shape, hidden_units=30)
model=model1.build_model()
model, call_back=utils.train_model(model,data[:4],label[:4], batch_s=32, epo=50)
utils.save_result(model, data, label, call_back)