import sys
import os
sys.path.insert(0, os.getcwd() + '/Modules/')
import utils
import models

data_path = os.getcwd() + '/Data/NY531/train/'

# for one step prediction
window_size = 100
dimension = 14
step_size = 1
batch_size = 32
epoch = 10000
Num_data = 53860
a = 6  # parameter to set the No of hidden units
hidden_unit = int(Num_data / (a * (window_size * dimension + dimension)))
print('number of hidden unit is ' + str(hidden_unit))
hidden_unit = 200
train_data, train_label = utils.load_data_one_step_prediction(data_path, step_size=step_size, window_size=window_size,
                                                              moving_only=False)
train_data, train_label = utils.normalize(train_data, train_label)
print(len(train_data))
print(len(train_label))
print(train_data[0].shape)
print(train_label[0].shape)
model2 = models.onestepModel(train_data[0].shape, hidden_units=hidden_unit)
model = model2.build_model()
model, call_back = utils.train_model(model, train_data, train_label, batch_s=batch_size, epo=epoch)
data_path = os.getcwd() + '/Data/NY531/test/'
test_data, test_label = utils.load_data_one_step_prediction(data_path, step_size=step_size, window_size=window_size,
                                                            moving_only=False)
print(len(test_data))
print(len(test_label))
test_data, test_label = utils.normalize(test_data, test_label)
utils.save_result_one_step_prediction(model, test_data, test_label, call_back)
