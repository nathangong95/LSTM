import sys
import os
import numpy as np 
import h5py
import scipy.io as sio
from matplotlib import pyplot as plt
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

