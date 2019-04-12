import sys
import os
sys.path.insert(0, os.getcwd() + '/src/')
import lstm
import utils

train_file = open('Data/train_data', 'r')
test_file = open('Data/test_data', 'r')
train_data = train_file.read()
test_data = train_file.read()

joints_names = ["head", "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist"]

joints = utils.extract_humanEva_label(train_data, 'S1', 'Box', 'C1', joints_names)
print(joints.shape)

