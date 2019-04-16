import sys
import os
sys.path.insert(0, os.getcwd() + '/src/')
# import lstm
import utils
import matplotlib.pyplot as plt
import cv2

train_file = open('Data/train_data', 'r')
train_data = train_file.read()
train_file.close()
test_file = open('Data/test_data', 'r')
test_data = test_file.read()
test_file.close()

joints_names = ["head", "l_wrist", "r_wrist", "l_elbow", "r_elbow", "l_shoulder",  "r_shoulder"]
subjects = ['S3']
actions = ['ThrowCatch']#, 'Gestures', 'ThrowCatch', 'Box']
cameras = ['C1']

for subject in subjects:
    for action in actions:
        for camera in cameras:
            train_joint, train_name = utils.extract_humanEva_label(train_data, subject, action, camera, joints_names)
for subject in subjects:
    for action in actions:
        for camera in cameras:
            test_joint, test_name = utils.extract_humanEva_label(test_data, subject, action, camera, joints_names)
name = train_name + test_name
joint = train_joint + test_joint
joint = [x for _,x in sorted(zip(name,joint))]
name.sort()
for i in range(len(joint)):
    image_path = '../data/data/images/HumanEva/'+subjects[0]+'/'+actions[0]+'/'+cameras[0]+'/'+str(name[i])+'.png'
    image = cv2.imread(image_path)
    plt.imshow(image)
    plt.scatter(joint[i][:,0], joint[i][:,1])
    plt.pause(0.0001)
    plt.clf()

