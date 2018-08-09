""" 
This module handles the basic data processing for LSTM training and LSTM training itself
training funciton is the highest level function of this module
Author: Chenghao Gong
Date: 7/20/2018
Version: 1.0
"""
import numpy as np
import keras.utils
import scipy.io as spio
from keras import utils as np_utils
from keras.utils import to_categorical
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import matplotlib.pyplot as plt
import itertools
import os
import h5py
import simjoints as sj

def loadData(data_path):
    filelist=os.listdir(data_path)
    train_data=[]
    train_label=[]
    f=[]
    filelist.sort()
    for file in filelist:
        data=np.genfromtxt(data_path+file,delimiter=',')
        a,s=data.shape
        train_data.append(np.reshape(data[:a-2,:].T,(s,1,a-2)))
        train_label.append(to_categorical(np.reshape(data[a-2,:],(s,))))
        f.append(np.reshape(data[a-1,:],(s,)))
    return train_data, train_label, f
'''
def loadData2(data_path):
    """ function to load training data, training label, and time stamp
    Args:
        datapath (str): path to the data folder
    Returns:

    """
    Data = spio.loadmat(data_path+'/Folder.mat', squeeze_me=True,struct_as_record=True)
    train_data=[]
    train_data.append(Data['Folder1'])
    train_data.append(Data['Folder2'])
    train_data.append(Data['Folder3'])
    train_data.append(Data['Folder4'])
    train_data.append(Data['Folder5'])
    train_data.append(Data['Folder6'])
    train_label=np.load(data_path+'/train_label.npy')
    for i in range(6):
        a,b,s=train_data[i].shape
        train_data[i]=np.reshape(train_data[i],(14,s))
        train_data[i]=train_data[i].T
        train_data[i]=np.reshape(train_data[i],(s,1,14))
    for i in range(6):
        train_label[i]=to_categorical(train_label[i])

    f=[]
    f.append(h5py.File(data_path+'/t1.mat'))
    f.append(h5py.File(data_path+'/t2.mat'))
    f.append(h5py.File(data_path+'/t3.mat'))
    f.append(h5py.File(data_path+'/t4.mat'))
    f.append(h5py.File(data_path+'/t5.mat'))
    f.append(h5py.File(data_path+'/t6.mat'))
    f.append(h5py.File(data_path+'/t7.mat'))
    return train_data, train_label, f
'''

def addFeature(self,Data):
        """ Functions that add more feature on the existing data
        Args:
            Data (s*14 nparray): original 14 dimensional data
        Returns:
            (s*22 nparray): new 22 dimensional data
        """
        s,a=Data.shape
        newData=np.zeros((s,22))
        newData[:,:14]=Data
        for i in range(s):
            newData[i,14]=((Data[i,2]-Data[i,6])**2+(Data[i,3]-Data[i,7])**2)**0.5#Left lower
            newData[i,15]=((Data[i,4]-Data[i,8])**2+(Data[i,5]-Data[i,9])**2)**0.5#Right lower
            newData[i,16]=((Data[i,6]-Data[i,10])**2+(Data[i,7]-Data[i,11])**2)**0.5#Left Upper
            newData[i,17]=((Data[i,8]-Data[i,12])**2+(Data[i,9]-Data[i,13])**2)**0.5#Right Upper
        
            v1=[Data[i,2]-Data[i,6],Data[i,3]-Data[i,7]]
            v2=[Data[i,10]-Data[i,6],Data[i,11]-Data[i,7]]
            newData[i,18]=sj.get_elbow_angle(v1,v2)#left elbow

            v1=[Data[i,4]-Data[i,8],Data[i,5]-Data[i,9]]
            v2=[Data[i,12]-Data[i,8],Data[i,13]-Data[i,9]]
            newData[i,19]=sj.get_elbow_angle(v1,v2)#right elbow

            v1=[Data[i,12]-Data[i,10],Data[i,13]-Data[i,11]]
            v2=[Data[i,6]-Data[i,10],Data[i,7]-Data[i,11]]
            newData[i,20]=sj.get_axillary_angle(v1,v2)#left shoulder
        
            v1=[Data[i,10]-Data[i,12],Data[i,11]-Data[i,13]]
            v2=[Data[i,8]-Data[i,12],Data[i,9]-Data[i,13]]
            newData[i,21]=sj.get_axillary_angle(v1,v2)#right shoulder
        return newData

def trainLSTM(train_data,train_label,Hidden_unit,batch_s,epoch):
    """ Low level train funciton
    Args:
        trainin
    input: training data, training label, params
    output: model
    """
    _,_,s=train_data[0].shape
    model = Sequential()
    model.add(LSTM(Hidden_unit, input_shape=(1, s)))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='sigmoid'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    for i in range(len(train_data)):
        model.fit(train_data[i],train_label[i], batch_size=batch_s, epochs=epoch)
    return model

def toIntegerLabel(l):
    '''
    low level helper function to transfer one hot label to integer label
    input: one hot label
    output: integer label
    '''
    label=[]
    s,_=l.shape
    for i in range(s):
        label.append(l[i].tolist().index(max(l[i].tolist())))
    return label
def toOnOffSet(integerLabel):
    '''
    low level helper function to transfer integer label to on/off set
    input: integer label
    output: list of on/off set
    '''
    onoff=[]
    lmsk = [(el==1) or (el==3) for el in integerLabel]
    Rmsk = [(el==2) or (el==3) for el in integerLabel]
    i=0
    while i<len(lmsk):
        if lmsk[i]:
            temp=[]
            temp.append('l')
            temp.append(i)
            for j in range(i,len(lmsk)):
                if not lmsk[j]:
                    temp.append(j-1)
                    onoff.append(temp)
                    i=j
                    break
        i+=1
    i=0
    while i<len(Rmsk):
        if Rmsk[i]:
            temp=[]
            temp.append('r')
            temp.append(i)
            for j in range(i,len(Rmsk)):
                if not Rmsk[j]:
                    temp.append(j-1)
                    onoff.append(temp)
                    i=j
                    break
        i+=1
    return onoff
def toPanda_test(model,train_data,whichFolder,f):
    '''
    function to take testing data and model and output the panda prediction
    input: model, testing data, folder and time stamp information
    output: pandas data frame with the prediction result
    '''
    IntegerLabel=toIntegerLabel(model.predict(train_data[whichFolder]))
    OnOff=toOnOffSet(IntegerLabel)
    df=[]
    for i in range(len(OnOff)):
        OnOff[i][1]=f[whichFolder][OnOff[i][1]]/1000
        OnOff[i][2]=f[whichFolder][OnOff[i][2]]/1000
        df.append(pd.DataFrame(data={'onset':[OnOff[i][1]],  'offset':[OnOff[i][2]],  'label':[OnOff[i][0]]}))
    if len(df)==0:
        df.append(pd.DataFrame(data={'onset':[0],  'offset':[0],  'label':['b']}))
    df = pd.concat(df, axis=0)
    df.sort_values(by=('onset'), inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def toPanda_train(train_label,whichFolder,f):
    '''
    function that find the pandas data frame for training data
    input: training label and folder and time stamp
    output: pandas data frame of training data
    '''
    IntegerLabel=toIntegerLabel(train_label[whichFolder])
    OnOff=toOnOffSet(IntegerLabel)
    df=[]
    for i in range(len(OnOff)):
        OnOff[i][1]=f[whichFolder][OnOff[i][1]]/1000
        OnOff[i][2]=f[whichFolder][OnOff[i][2]]/1000
        df.append(pd.DataFrame(data={'onset':[OnOff[i][1]],  'offset':[OnOff[i][2]],  'label':[OnOff[i][0]]}))
    df = pd.concat(df, axis=0)
    df.sort_values(by=('onset'), inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def score_periods(df_manual, df_pred, params=None, err1thresh=2, err2thresh=5):
    '''
    function to find df_result
    input: pandas training and testing data frames and threshold for err1 and err2
    output: df_result
    '''
    # initialize storage variables/arrays
    err1 = np.empty(len(df_manual)) # err1
    err2 = np.empty(len(df_manual)) # err2
    detected = np.zeros(len(df_manual), dtype=bool) # detected
    
    # Loop through each manual
    for i, (label, offset, onset) in df_manual.iterrows():
                
        # search for nearest neighbor in df_pred
        pred_onsets = df_pred['onset'].values
        pred_offsets = df_pred['offset'].values
        
        i_onset_match = np.argmin(np.abs(pred_onsets-onset))
        i_offset_match = np.argmin(np.abs(pred_offsets - offset))
        
        if label=='b' or i_onset_match != i_offset_match:
            err1[i] = np.nan
            err2[i] = np.nan
            detected[i] = False
            continue
        
        # calculate onset and offset error (err1, err2)
        err1[i] = onset - pred_onsets[i_onset_match]
        err2[i] = offset - pred_offsets[i_offset_match]
        if np.abs(err1[i])>err1thresh or np.abs(err2[i])>err2thresh:
            detected[i] = False
        else:
            detected[i]=True

    df_return = df_manual.copy()
    df_return['err1'] = err1
    df_return['err2'] = err2
    df_return['detected'] = detected
    
    return df_return

def trainning(train_data, train_label,f,params):
    '''
    high level training function
    input: training data, training label, time stamps, training params
    output: df_result and df_pred
    '''
    result=[]
    pred=[]
    for i in range(4):
        train_set=[]
        train_set_label=[]
        test_folder=i
        for j in range(4):
            if j!=i:
                train_set.append(train_data[j])
                train_set_label.append(train_label[j])
        model=trainLSTM(train_set, train_set_label, params[0],params[1],params[2])
        df_manual=toPanda_train(train_label,test_folder,f)
        df_pred=toPanda_test(model,train_data,test_folder,f)
        df_results = score_periods(df_manual, df_pred)
        pred.append(df_pred)
        result.append(df_results)
    return result,pred
def evaResult(results):
    '''
    high level evaluation function
    input: df_result
    output: hit rate, err1, err2
    '''
    total_event=0.0
    total_detected=0.0
    total_err1=0.0
    total_err2=0.0
    for result in results:
        total_event+=len(result)
        total_detected+=sum(result['detected'])
        for i in range(len(result)):
            if result['detected'][i]:
                total_err1+=np.abs(result['err1'][i])
                total_err2+=np.abs(result['err2'][i])
    return (total_detected/total_event,total_err1/total_event,total_err2/total_event)

def demo():
    return None
if __name__ == "__main__":
    demo()