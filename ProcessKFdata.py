import sys
import os
import numpy as np 
import h5py
import scipy.io as sio
#import tables
path=os.getcwd()+'/PatientPose_train-test/'

file=h5py.File(path+'NY531/train/kf_training_joints.mat','r')
NY531_train=file['detections']['manual']['locs']
NY531_train=np.asarray(NY531_train)
file.close()

file=h5py.File(path+'RCH1/train/kf_training_joints.mat','r')
RCH1_train=file['detections']['manual']['locs']
RCH1_train=np.asarray(RCH1_train)
file.close()

file=h5py.File(path+'RCH3/train/kf_training_joints.mat','r')
RCH3_train=file['detections']['manual']['locs']
RCH3_train=np.asarray(RCH3_train)
file.close()

NY531_test=sio.loadmat(path+'NY531/test/gt.mat')
NY531_test=np.asarray(NY531_test['joints'])

RCH1_test=sio.loadmat(path+'RCH1/test/gt.mat')
RCH1_test=np.asarray(RCH1_test['joints_gt'])

RCH3_test=sio.loadmat(path+'RCH3/test/gt.mat')
RCH3_test=np.asarray(RCH3_test['joints_gt'])

print(NY531_test.shape)

print(NY531_train.shape)

NY531_test=np.reshape(NY531_test,(14,3000))
RCH1_test=np.reshape(RCH1_test,(14,1000))
RCH3_test=np.reshape(RCH3_test,(14,1000))

NY531_train=np.reshape(NY531_train,(500,14)).T
RCH1_train=np.reshape(RCH1_train,(500,14)).T
RCH3_train=np.reshape(RCH3_train,(500,14)).T

save_path=os.getcwd()+'/KFData'
np.savetxt(save_path+'/train/NY531.csv', NY531_train, delimiter=",")
np.savetxt(save_path+'/train/RCH1.csv', RCH1_train, delimiter=",")
np.savetxt(save_path+'/train/RCH3.csv', RCH3_train, delimiter=",")

np.savetxt(save_path+'/test/NY531.csv', NY531_test, delimiter=",")
np.savetxt(save_path+'/test/RCH1.csv', RCH1_test, delimiter=",")
np.savetxt(save_path+'/test/RCH3.csv', RCH3_test, delimiter=",")