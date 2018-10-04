import scipy.io as spio
import h5py
import numpy as np
import os
import sys
sys.path.insert(0,os.getcwd()+'/Modules/')
import simjoints as sj

# add feature part
data_path=os.getcwd()+'/Data/'
def make_22D_data(data_path): 
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

def load_data(data_path,step_size=1,window_size=20):
    filelist=os.listdir(data_path)
    train_data=[]
    train_label=[]
    filelist.sort()
    for file in filelist:
        data=np.genfromtxt(data_path+file,delimiter=',')
        a,s=data.shape
        train_data.append(np.reshape(data[:a-2,:].T,(s,a-2)))
        train_label.append(to_categorical(np.reshape(data[a-2,:],(s,))))
    # deal with the step size here
    new_train_data,new_train_label=stack_data(train_data,train_label,step_size,window_size)
    length_folder=len(new_train_data)
    train_length=length_folder-length_folder/3
    self.train_data=new_train_data[:train_length]
    self.test_data=new_train_data[train_length:]
    self.train_label=new_train_label[:train_length]
    self.test_label=new_train_label[:train_length]
    #return new_train_data,new_train_label

def stack_data(train_data,train_label,step_size,window_size):
    new_train_data=[]
    new_train_label=[]
    for train_dat, train_lab in zip(train_data,train_label):
        s,d=train_dat.shape
        new_train_dat=[]
        new_train_lab=[]
        for i in range(s):
            if i%self.step_size==0:
                window=[]
                for j in range(self.window_size):
                    if (i-self.window_size+j+1)>0:
                        window.append(train_dat[i-self.window_size+j+1,:])
                    else:
                        window.append(np.reshape(np.zeros((1,d)),(d,)))
                window=np.asarray(window)
                new_train_dat.append(window)
                new_train_lab.append(train_lab[i])
        new_train_data.append(np.asarray(new_train_dat))
        new_train_label.append(np.asarray(new_train_lab))
    return new_train_data,new_train_label
    