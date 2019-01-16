import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models
import numpy as np
""" data map:
0: head x
1: head y
2: left hand x
3: left hand y
4: right hand x
5: right hand y
6: left elbow x
7: left elbow y
8: right elbow x
9: right elbow y
10: left shoulder x
11: left shoulder y
12: right shoulder x
13: right shoulder y
GT1:NY531
GT2:RCH1
GT3:RCH3
"""
data_path=os.getcwd()+'/Data/NY531/train/'

train_data,_=utils.load_data_one_step_prediction(data_path,step_size=1,window_size=1,moving_only=False)
print(len(train_data))
train_data=np.concatenate(train_data)
s,_,_=train_data.shape
train_data=np.reshape(train_data, (s,14))
print(train_data.shape)