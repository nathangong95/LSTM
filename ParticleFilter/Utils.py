import sys
import os
import numpy as np 
import h5py
import scipy.io as sio
from matplotlib import pyplot as plt
def estimateQR_with_a(joints, est):
	#13494,7,2
	#print(joints[0,:,:])
	#print(est[0,:,:])
	s=joints.shape[0]
	F=np.array([[1.0,0.0,1.0,0.0,0.5,0.0],
		[0.0,1.0,0.0,1.0,0.0,0.5],
		[0.0,0.0,1.0,0.0,1.0,0.0],
		[0.0,0.0,0.0,1.0,0.0,1.0],
		[0.0,0.0,0.0,0.0,1.0,0.0],
		[0.0,0.0,0.0,0.0,0.0,1.0]])

	X=np.zeros((6,7,s))
	for i in range(2,s):
		for j in range(7):
			X[0,j,i]=joints[i,j,0]
			X[1,j,i]=joints[i,j,1]
			X[2,j,i]=joints[i,j,0]-joints[i-1,j,0]
			X[3,j,i]=joints[i,j,1]-joints[i-1,j,1]
			X[4,j,i]=joints[i,j,0]-joints[i-1,j,0]-joints[i-1,j,0]+joints[i-2,j,0]
			X[5,j,i]=joints[i,j,1]-joints[i-1,j,1]-joints[i-1,j,1]+joints[i-2,j,1]
	for j in range(7):
		X[0,j,0]=joints[0,j,0]
		X[1,j,0]=joints[0,j,1]

		X[0,j,1]=joints[1,j,0]
		X[1,j,1]=joints[1,j,1]

		X[2,j,1]=joints[1,j,0]-joints[0,j,0]
		X[3,j,1]=joints[1,j,1]-joints[0,j,1]

	R=np.zeros((6,6,7))
	Q=np.zeros((2,2,7))
	for j in range(7):
		for i in range(2,s):
			#print(F.shape)
			#print(X[:,j,i].shape)
			jointji=np.reshape(X[:,j,i],(6,1))
			jointji_1=np.reshape(X[:,j,i-1],(6,1))
			tmp=jointji-F@jointji_1
			#print(tmp.shape)
			R[:,:,j]+=tmp@tmp.T
		R[:,:,j]/=(s-2)
		for i in range(s):
			Q[:,:,j]+=np.reshape((est[i,j,:]-joints[i,j,:]),(2,1))@np.reshape((est[i,j,:]-joints[i,j,:]),(1,2))
		Q[:,:,j]=Q[:,:,j]/float(s)
	return R,Q

joints=np.load('QR_train_GT.npy')
est=np.load('QR_train_PD.npy')
R,Q=estimateQR_with_a(joints,est)
np.save('R_with_a.npy',R)
''' load and save every frame in each individual file
file=h5py.File('Heatmap.mat','r')
NY531=file['heatmapResized']
NY531=np.asarray(NY531)
for i in range(3000):
	print(i)
	NY531_train=NY531[i,:,:,:]
	np.save(str(i+1)+'.npy',NY531_train)

'''
''' load each individual file and extract the joint locations
heatmap_path='../Heatmaps/'
joints=np.zeros((3000,7,2))
for i in range(3000):
	heatmap=np.load(heatmap_path+str(i+1)+'.npy')
	for j in range(7):
		result = np.where(heatmap[j,:,:] == np.amax(heatmap[j,:,:]))
		joints[i,j,0]=result[0]
		joints[i,j,1]=result[1]
#print(joints[:,:,1])
np.save('jointsfromheatmap.npy',joints)
'''

''' transfer kalman filter result in to npy file
from scipy.io import loadmat
file = loadmat('joints_kalman.mat')
NY531=file['joints_kalman']
NY531=np.asarray(NY531)
print(NY531.shape)
joints=np.zeros((3000,7,2))
for i in range(3000):
	for j in range(7):
		joints[i,j,0]=NY531[0,j,i]
		joints[i,j,1]=NY531[1,j,i]
np.save('joints_kalman.npy',joints)
'''
'''
from scipy.io import loadmat
#file=h5py.File('../Data/QREstimation/NY531_training_estimates.mat','r')
file = loadmat('../Data/QREstimation/NY531_training_estimates.mat')
NY531=file['joints']
NY531=np.asarray(NY531)
print(NY531.shape)
joints=np.zeros((13494,7,2))
for i in range(13494):
	for j in range(7):
		joints[i,j,0]=NY531[0,j,i]
		joints[i,j,1]=NY531[1,j,i]
np.save('QR_train_PD.npy',joints)
'''
