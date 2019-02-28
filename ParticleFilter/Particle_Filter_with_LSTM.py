import sys
import os
sys.path.insert(0, os.getcwd() + '/../Modules/')
import utils
from keras.models import load_model
import numpy as np

which_model = '20192261822(return sequance)'
model_path = '../Result/' + which_model + '/model.h5'
model = load_model(model_path)
window_size = model.layers[0].output_shape[1]
joints = np.load('Data/QR_train_GT.npy')
joints = np.reshape(joints, (joints.shape[0], 14))
joints, label = utils.stack_data_one_step_prediction([joints], 1, window_size, False)
joints, label = utils.normalize(joints, label)
predict = model.predict(joints[0])
GT = np.zeros((label[0].shape[0], 14))
PD = np.zeros((label[0].shape[0], 14))
for i in range(label[0].shape[0]):
    GT[i, :] = label[0][i, window_size - 1, :]
    PD[i, :] = predict[i, window_size - 1, :]
GT = np.reshape(GT, (GT.shape[0], 7, 2))
PD = np.reshape(PD, (PD.shape[0], 7, 2))
GT = utils.recover_from_normalize(GT)
PD = utils.recover_from_normalize(PD)
print(GT[0,:])
print(PD[0,:])
R = utils.estimateR_with_LSTM(GT, PD)
print(R)
