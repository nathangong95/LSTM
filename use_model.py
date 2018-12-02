import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models

data_path=os.getcwd()+'/KFData/train/'
#data,label=utils.load_data(data_path,step_size=1,window_size=20)
#model1=models.Model1(data[0].shape, hidden_units=30)
#model=model1.build_model()
#print(data[0].shape)
#print(label[0].shape)
#model, call_back=utils.train_model(model,data[:4],label[:4], batch_s=1, epo=1)
#utils.save_result(model, data, label, call_back)

#for one step prediction
train_data,train_label=utils.load_data_one_step_prediction(data_path,step_size=1,window_size=30)
train_data,train_label=utils.normalize(train_data,train_label)
print(len(train_data))
print(len(train_label))
print(train_data[0].shape)
print(train_label[0].shape)
model2=models.onestepModel(train_data[0].shape, hidden_units=120)
model=model2.build_model()
model, call_back=utils.train_model(model,train_data,train_label,batch_s=16,epo=50)
data_path=os.getcwd()+'/KFData/test/'
test_data,test_label=utils.load_data_one_step_prediction(data_path,step_size=1,window_size=30)
test_data,test_label=utils.normalize(test_data,test_label)
utils.save_result_one_step_prediction(model, test_data, test_label, call_back)