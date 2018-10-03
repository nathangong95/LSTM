import scipy.io as spio
import h5py
import numpy as np
import os
import sys
sys.path.insert(0,os.getcwd()+'/Modules/')
import MJD
data_path=os.getcwd()+'/Data22D/'

option=0
Window_size=20
num_joints=7

if option ==0:
	save_path=os.getcwd()+'/MJDdata0/'
	filelist=os.listdir(data_path)
	filelist.sort()
	j=1
	for file in filelist:
		data=np.genfromtxt(data_path+file,delimiter=',')
		print(data.shape)
		R,B=MJD.mjd(data,Window_size,option)
		new_data=[]
		for Rs, Bs, dat in zip(R,B,data.T):
			Rs=np.reshape(Rs,(Window_size*num_joints,))
			Bs=np.reshape(Bs,(Window_size*num_joints,))
			new_dat=list(Rs)+list(Bs)
			new_dat=new_dat+list(dat)
			new_data.append(new_dat)
		new_data=np.asarray(new_data).T
		print(new_data.shape)
		np.savetxt(save_path+'/part'+str(j)+'(mjd_option0).csv', new_data, delimiter=",")
		j+=1


