import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models
from keras.models import load_model
import os
import numpy as np


data_path=os.getcwd()+'/Data/'
data,label=utils.load_data(data_path)

model = load_model('model1.h5')
print(model.predict(data[5]))