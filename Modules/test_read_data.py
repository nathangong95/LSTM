import numpy as np
from operator import itemgetter

train_file = open('train_data', 'r')
train_data = train_file.read()


def extract_humanEva_label(file_data, which_subject, which_action, which_cam):
    idx = [10, 11, 15, 16, 20, 21, 25, 26, 30, 31, 35, 36, 40, 41, 45, 46, 50, 51, 55, 56, 60, 61, 65, 66, 70, 71, 75,
           76]
    file_data = file_data.split('\n')
    joints_data = []
    for dat in file_data:
        splited = dat.split(',')
        names = splited[0].split('/')
        if len(names) == 7:
            if names[3] == which_subject and names[4] == which_action and names[5] == which_cam:
                joints = itemgetter(*idx)(splited)
                joints = [float(i) for i in joints]
                joints = np.asarray(joints)
                joints = np.reshape(joints, (28, 1))
                joints = np.reshape(joints, (14, 2))
                joints_data.append(joints)
    joints_data = np.asarray(joints_data)
    return joints_data


joints = extract_humanEva_label(train_data, 'S1', 'Box', 'C1')

# JOINTS = ("head", "neck", "thorax", "pelvis", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist", "l_knee", "l_ankle", "r_knee", "r_ankle")
