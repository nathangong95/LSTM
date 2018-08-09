import scipy.io as spio
import h5py
import numpy as np
import os
import sys
sys.path.insert(0,os.getcwd()+'/Modules/')
import simjoints as sj
'''
data_path=os.getcwd()+'/Data'
Data = spio.loadmat(data_path+'/Folder.mat', squeeze_me=True,struct_as_record=True)
data=[]
for i in range(6):
	_,_,s=Data['Folder'+str(i+1)].shape
	tmp=np.zeros((16,s))
	tmp[:14,:]=np.reshape(Data['Folder'+str(i+1)],(14,s))
	data.append(tmp)


train_label=np.load(data_path+'/train_label.npy')
for i in range(6):
	data[i][14,:]=train_label[i]
f=[]
f.append(h5py.File(data_path+'/t1.mat'))
f.append(h5py.File(data_path+'/t2.mat'))
f.append(h5py.File(data_path+'/t3.mat'))
f.append(h5py.File(data_path+'/t4.mat'))
f.append(h5py.File(data_path+'/t5.mat'))
f.append(h5py.File(data_path+'/t6.mat'))
for i in range(6):
	data[i][15,:]=f[i]['color']['kT'][0]/1000
for i in range(6):
	np.savetxt(data_path+'/part'+str(i+1)+'.csv', data[i], delimiter=",")
'''
# add feature part
data_path=os.getcwd()+'/Data/'
filelist=os.listdir(data_path)
j=0
for file in filelist:
    Data=np.genfromtxt(data_path+file,delimiter=',')
    _,s=Data.shape
    Data=Data.T
    newData=np.zeros((s,24))
    newData[:,:14]=Data[:,:14]
    for i in range(s):
        newData[i,14]=((Data[i,2]-Data[i,6])**2+(Data[i,3]-Data[i,7])**2)**0.5#Left lower
        newData[i,15]=((Data[i,4]-Data[i,8])**2+(Data[i,5]-Data[i,9])**2)**0.5#Right lower
        newData[i,16]=((Data[i,6]-Data[i,10])**2+(Data[i,7]-Data[i,11])**2)**0.5#Left Upper
        newData[i,17]=((Data[i,8]-Data[i,12])**2+(Data[i,9]-Data[i,13])**2)**0.5#Right Upper
        
        v1=[Data[i,2]-Data[i,6],Data[i,3]-Data[i,7]]
        v2=[Data[i,10]-Data[i,6],Data[i,11]-Data[i,7]]
        newData[i,18]=sj.get_elbow_angle(v1,v2)#left elbow
        if np.isnan(newData[i,18]):
            newData[i,18]=newData[i-1,18]

        v1=[Data[i,4]-Data[i,8],Data[i,5]-Data[i,9]]
        v2=[Data[i,12]-Data[i,8],Data[i,13]-Data[i,9]]
        newData[i,19]=sj.get_elbow_angle(v1,v2)#right elbow
        if np.isnan(newData[i,19]):
            newData[i,19]=newData[i-1,19]

        v1=[Data[i,12]-Data[i,10],Data[i,13]-Data[i,11]]
        v2=[Data[i,6]-Data[i,10],Data[i,7]-Data[i,11]]
        newData[i,20]=sj.get_axillary_angle(v1,v2)#left shoulder
        if np.isnan(newData[i,20]):
            newData[i,20]=newData[i-1,20]
        
        v1=[Data[i,10]-Data[i,12],Data[i,11]-Data[i,13]]
        v2=[Data[i,8]-Data[i,12],Data[i,9]-Data[i,13]]
        newData[i,21]=sj.get_axillary_angle(v1,v2)#right shoulder
        if np.isnan(newData[i,21]):
            newData[i,21]=newData[i-1,21]

        newData[i,22]=Data[i,14]
        newData[i,23]=Data[i,15]
    np.savetxt(data_path+'/part'+str(j+1)+'(22D).csv', newData.T, delimiter=",")
    j+=1