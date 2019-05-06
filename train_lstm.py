import sys
import os

sys.path.insert(0, os.getcwd() + '/src/')
# import lstm
import util
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model

import sys
import os
sys.path.insert(0, os.getcwd() + '/Modules/')
import utils
import models

train_file = open('Data/train_data', 'r')
train_data = train_file.read()
train_file.close()
test_file = open('Data/test_data', 'r')
test_data = test_file.read()
test_file.close()

joints_names = ["head", "l_wrist", "r_wrist", "l_elbow", "r_elbow", "l_shoulder", "r_shoulder"]
subjects = ['S1', 'S2']
actions = ['Box', 'Gestures', 'ThrowCatch']  # , 'Gestures', 'ThrowCatch', 'Box']
cameras = ['C1']

names = []
joints = []
sections = []
for subject in subjects:
    for action in actions:
        for camera in cameras:
            train_joint, train_name = util.extract_humanEva_label(train_data, subject, action, camera, joints_names)
            test_joint, test_name = util.extract_humanEva_label(test_data, subject, action, camera, joints_names)
            name = train_name + test_name
            joint = train_joint + test_joint
            joint = [x for _, x in sorted(zip(name, joint))]
            name.sort()
            names.append(name)
            joints.append(np.asarray(joint).reshape((len(name), 14)))
            sections.append(subject+'_'+action)
            # np.save(subject+'_'+action+'.npy', joint)


# for i in range(len(joint)):
#     image_path = '../data/data/images/HumanEva/'+subjects[0]+'/'+actions[0]+'/'+cameras[0]+'/'+str(name[i])+'.jpg'
#     image = cv2.imread(image_path)
#     print(image.shape)
#     plt.imshow(image)
#     plt.scatter(joint[i][:,0], joint[i][:,1])
#     plt.pause(0.0001)
#     plt.clf()
def interp(data):
    # data: n*14
    n, _ = data.shape
    new_data = np.zeros((n, 14))
    for i in range(1, n):
        new_data[i - 1] = (data[i - 1] + data[i]) / 2.0
    new_data[n - 1] = data[n - 1]
    return new_data


def flip(data):
    # data: n*14
    n, _ = data.shape
    data_copy = data.reshape((n, 7, 2))
    new_data = np.zeros((n, 7, 2))
    for i in range(n):
        new_data[i, 0, :] = data_copy[i, 0, :]
        new_data[i, 1, :] = data_copy[i, 2, :]
        new_data[i, 2, :] = data_copy[i, 1, :]
        new_data[i, 3, :] = data_copy[i, 4, :]
        new_data[i, 4, :] = data_copy[i, 3, :]
        new_data[i, 5, :] = data_copy[i, 6, :]
        new_data[i, 6, :] = data_copy[i, 5, :]
    new_data = new_data.reshape((n, 14))
    return new_data


interp_joints = []
filp_joints = []
interp_flip_joints = []
for joint in joints:
    interp_joints.append(interp(joint))
    filp_joints.append(flip(joint))
    interp_flip_joints.append(flip(interp(joint)))

previous_name = np.inf
data_list = []
name_list = []
interp_data_list = []
filp_data_list = []
interp_flip_data_list = []
name = None
data = None
interp_data = None
filp_data = None
interp_flip_data = None
for i in range(len(names)):
    for j in range(len(names[i])):
        if int(names[i][j]) != previous_name + 1:
            if data is not None:
                data_list.append(data)
            if interp_data is not None:
                interp_data_list.append(interp_data)
            if filp_data is not None:
                filp_data_list.append(filp_data)
            if interp_flip_data is not None:
                interp_flip_data_list.append(interp_flip_data)
            if name is not None:
                name_list.append(name)
            data = joints[i][j]
            interp_data = interp_joints[i][j]
            filp_data = filp_joints[i][j]
            interp_flip_data = interp_flip_joints[i][j]
            previous_name = int(names[i][j])
            name = np.array([[int(names[i][j])]])
        else:
            data = np.vstack((data, joints[i][j]))
            interp_data = np.vstack((interp_data, interp_joints[i][j]))
            filp_data = np.vstack((filp_data, filp_joints[i][j]))
            interp_flip_data = np.vstack((interp_flip_data, interp_flip_joints[i][j]))
            previous_name = int(names[i][j])
            name = np.vstack((name, np.array([[previous_name]])))

data_list = data_list + interp_data_list + filp_data_list + interp_flip_data_list

window_size = 30
total_data = np.zeros((1, window_size, 14))
total_label = np.zeros((1, window_size, 14))
for data in data_list:
    if data.shape[0] < window_size + 1:
        continue
    num_data = data.shape[0] - window_size
    batch_data = np.zeros((num_data, window_size, 14))
    batch_label = np.zeros((num_data, window_size, 14))
    for i in range(num_data):
        batch_data[i] = data[i:i+window_size]
        batch_label[i] = data[i+1:i+1+window_size]
    # print(total_data.shape)
    # print(batch_data.shape)
    total_data = np.concatenate((total_data, batch_data), axis=0)
    total_label = np.concatenate((total_label, batch_label), axis=0)
total_data = total_data[1:]
total_label = total_label[1:]
print(total_data.shape)
print(total_label.shape)

# train_data = []
# train_label = []
# train_data.append(total_data[0:1000])
# train_label.append(total_label[0:1000])
#
# train_data.append(total_data[1000:7500])
# train_label.append(total_label[1000:7500])
#
# test_data = []
# test_label = []
# test_data.append(total_data[7500:])
# test_label.append(total_label[7500:])
#
#
# train_data, train_label = utils.normalize(train_data, train_label)
# test_data, test_label = utils.normalize(test_data, test_label)
#
# hidden_unit = 150
# batch_size = 32
# epoch = 10000
# model = models.onestepModel(train_data[0].shape, hidden_units=hidden_unit)
# model = model.build_model()
# model = load_model('Result/2019551452best/model.h5')
# model, call_back = utils.train_model(model, train_data, train_label, batch_s=batch_size, epo=epoch)
# utils.save_result_one_step_prediction(model, test_data, test_label, call_back)