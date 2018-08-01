import scipy.io as spio
import h5py
import numpy as np
data_path='/home/zhengzheng/NathanCode/LSTM-master/Data'
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