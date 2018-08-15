import matplotlib.pyplot as plt
import numpy as np
import os
from keras.utils import to_categorical
import sys
import pandas as pd
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

def get_average(data,windowsize,index):
	s,_=data.shape
	if index<=window_size:
		output=data[0,:]
		for i in range(1,index+window_size):
			output+=data[i,:]
		output=output/(index+window_size)
	elif s<=index+window_size:
		output=data[index-window_size]
		for i in range(index-window_size+1,s):
			output+=data[i,:]
		output=output/(s-index+window_size)
	else:
		output=data[index-window_size]
		for i in range(index-window_size+1,index+window_size):
			output+=data[i,:]
		output=output/(2*window_size+1)
	return output


data_path="/home/zhengzheng/NathanCode/LSTM-master/Data22D/"
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

integerLabel=toIntegerLabel(train_label[0])
data=train_data[0]
s,_,_=data.shape
data=np.reshape(data, (s,22))
for i in range(1,6):
	s,_,_=train_data[i].shape
	integerLabel+=toIntegerLabel(train_label[i])
	data=np.concatenate((data,np.reshape(train_data[i],(s,22))),axis=0)
onoff=toOnOffSet(integerLabel)
#print(onoff)
#print(data.shape)
window_size=10
plot_data_on_l=[]
plot_data_off_l=[]
plot_data_on_r=[]
plot_data_off_r=[]
for of in onoff:
	if of[0]=='l':
		plot_data_on_l.append(get_average(data,window_size,of[1]))
		plot_data_off_l.append(get_average(data,window_size,of[2]))
	if of[0]=='r':
		plot_data_on_r.append(get_average(data,window_size,of[1]))
		plot_data_off_r.append(get_average(data,window_size,of[2]))

plot_data_off_l=np.asarray(plot_data_off_l)
plot_data_on_l=np.asarray(plot_data_on_l)
plot_data_off_r=np.asarray(plot_data_off_r)
plot_data_on_r=np.asarray(plot_data_on_r)
print(plot_data_off_l.shape)
print(plot_data_on_l.shape)
print(plot_data_off_r.shape)
print(plot_data_on_r.shape)
print(len(onoff))
########################plotting#############################
save_path="/home/zhengzheng/NathanCode/LSTM-master/Plots/"
for i in range(22):
    fig=plt.figure(figsize=(8, 12))
    ax1=plt.subplot(411)
    ax1.plot(plot_data_on_l[:,i])
    ax1.set_title("left onset")
    ax2=plt.subplot(412)
    ax2.plot(plot_data_off_l[:,i])
    ax2.set_title("left offset")
    ax3=plt.subplot(413)
    ax3.plot(plot_data_on_r[:,i])
    ax3.set_title("right onset")
    ax4=plt.subplot(414)
    ax4.plot(plot_data_off_r[:,i])
    ax4.set_title("right offset")
    ax4.set_xlabel("Events")
    fig.savefig(save_path+str(i+1)+".svg")