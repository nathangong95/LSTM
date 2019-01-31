import sys
import os
sys.path.insert(0,os.getcwd()+'/Modules/')
import utils
import models
import numpy as np
import matplotlib.pyplot as plt
""" data map:
0: head x
1: head y
2: left hand x
3: left hand y
4: right hand x
5: right hand y
6: left elbow x
7: left elbow y
8: right elbow x
9: right elbow y
10: left shoulder x
11: left shoulder y
12: right shoulder x
13: right shoulder y
GT1:NY531
GT2:RCH1
GT3:RCH3
"""

data_path=os.getcwd()+'/Data/NY531/train/'
moving_list=utils.load_moving_data(data_path,1)
data=[]
for l in moving_list:
	tmp=[]
	for i in range(7):
		tmp.append(l[:,i*2:(i+1)*2])
	data.append(tmp)
#print(data[0][0].shape)
def average_speed(data, step):
	down_sampled=[]
	for dat in data:
		if dat[0].shape[0]>step:
			seven=[]
			for i in range(7):
				seven.append(dat[i][::step,:])
			down_sampled.append(seven)
	speed=[]
	for i in range(7):
		tmp=[]
		for j in range(len(down_sampled)):
			for k in range(1,down_sampled[j][i].shape[0]):
				tmp.append(((down_sampled[j][i][k,0]-down_sampled[j][i][k-1,0])**2+(down_sampled[j][i][k,1]-down_sampled[j][i][k-1,1])**2)**0.5)
		speed.append(tmp)
	all_speed=[]
	for i in range(7):
		all_speed.append(float(sum(speed[i]))/len(speed[i]))
	return all_speed
titles=['head','l_hand','r_hand','l_elbow','r_elbow','l_shoulder','r_shoulder']
speed=[]
for i in range(1,15):
	speed.append(average_speed(data,i))
speed=np.asarray(speed)
ax=[]
for i in range(7):
	ax.append(plt.subplot(4,2,i+1))
	plt.plot(speed[:,i])
	ax[i].set_ylim([0,20])
	plt.xlabel('Step Size')
	plt.ylabel('Average_speed')
	plt.title(titles[i])
plt.show()
print(speed.shape)
'''

train_data,_=utils.load_data_one_step_prediction(data_path,step_size=1,window_size=1,moving_only=False)
print(len(train_data))
train_data=np.concatenate(train_data)
s,_,_=train_data.shape
train_data=np.reshape(train_data, (s,14))
print(train_data.shape)
data=[]
for i in range(7):
	data.append(train_data[:,i*2:(i+1)*2])
speed=[]
for i in range(7):
	temp=[]
	for j in range(1,data[i].shape[0]):
		temp.append(((data[i][j,0]-data[i][j-1,0])**2+(data[i][j,1]-data[i][j-1,1])**2)**0.5)
	speed.append(np.asarray(temp))
print(speed[0].shape)
titles=['head','l_hand','r_hand','l_elbow','r_elbow','l_shoulder','r_shoulder']
ax=[]
for i in range(7):
	ax.append(plt.subplot(4,2,i+1))
	plt.hist(speed[i], bins=np.linspace(0,15,100),rwidth=1)
	ax[i].set_ylim([0,60000])
	plt.xlabel('speed')
	plt.ylabel('Number of Occurance')
	plt.title(titles[i])
plt.show()

#################################################
def average_speed(data,step):
	down_sampled=[]
	for i in range(7):
		down_sampled.append(data[i][::step,:])
	speed=[]
	for i in range(7):
		temp=[]
		for j in range(1,down_sampled[i].shape[0]):
			temp.append(((down_sampled[i][j,0]-down_sampled[i][j-1,0])**2+(down_sampled[i][j,1]-down_sampled[i][j-1,1])**2)**0.5)
		speed.append(np.asarray(temp))
	all_speed=[]
	for i in range(7):
		all_speed.append(float(sum(speed[i]))/len(speed[i]))
	return all_speed
speed=[]
for i in range(1,15):
	speed.append(average_speed(data,i))
speed=np.asarray(speed)
ax=[]
for i in range(7):
	ax.append(plt.subplot(4,2,i+1))
	plt.plot(speed[:,i])
	ax[i].set_ylim([0,6])
	plt.xlabel('Step Size')
	plt.ylabel('Average_speed')
	plt.title(titles[i])
plt.show()
print(speed.shape)
'''