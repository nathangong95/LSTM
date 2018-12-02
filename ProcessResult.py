import os
import sys
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
which_result='20181212319'
path=os.getcwd()+'/Result/'+which_result+'/'
GT1=np.genfromtxt(path+'GroundTruth1.csv',delimiter=',')
GT2=np.genfromtxt(path+'GroundTruth2.csv',delimiter=',')
GT3=np.genfromtxt(path+'GroundTruth3.csv',delimiter=',')

PD1=np.genfromtxt(path+'Prediction1.csv',delimiter=',')
PD2=np.genfromtxt(path+'Prediction2.csv',delimiter=',')
PD3=np.genfromtxt(path+'Prediction3.csv',delimiter=',')

def percentage_in_range(GT,PD,r):
	rate=[]
	GT=np.reshape(GT, (GT.shape[0],7,2))
	PD=np.reshape(PD, (PD.shape[0],7,2))
	for i in range(7):
		hit=0
		for j in range(GT.shape[0]):
			distance=((GT[j,i,0]-PD[j,i,0])**2+(GT[j,i,1]-PD[j,i,1])**2)**0.5
			if distance<=r:
				hit+=1
		rate.append(hit/GT.shape[0])
	return rate#list with len 7

r=np.arange(0,30,0.1)
head=[]
l_hand=[]
r_hand=[]
l_elbow=[]
r_elbow=[]
l_shoulder=[]
r_shoulder=[]
for i in range(len(r)):
	print(i)
	rate=percentage_in_range(GT1,PD1,r[i])
	head.append(rate[0])
	l_hand.append(rate[1])
	r_hand.append(rate[2])
	l_elbow.append(rate[3])
	r_elbow.append(rate[4])
	l_shoulder.append(rate[5])
	r_shoulder.append(rate[6])
plt.subplot(421)
plt.plot(r,head)
plt.title('NY531')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand)
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand)
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow)
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow)
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder)
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder)
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()

r=np.arange(0,30,0.1)
head=[]
l_hand=[]
r_hand=[]
l_elbow=[]
r_elbow=[]
l_shoulder=[]
r_shoulder=[]
for i in range(len(r)):
	print(i)
	rate=percentage_in_range(GT2,PD2,r[i])
	head.append(rate[0])
	l_hand.append(rate[1])
	r_hand.append(rate[2])
	l_elbow.append(rate[3])
	r_elbow.append(rate[4])
	l_shoulder.append(rate[5])
	r_shoulder.append(rate[6])
plt.subplot(421)
plt.plot(r,head)
plt.title('RCH1')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand)
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand)
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow)
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow)
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder)
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder)
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()

r=np.arange(0,30,0.1)
head=[]
l_hand=[]
r_hand=[]
l_elbow=[]
r_elbow=[]
l_shoulder=[]
r_shoulder=[]
for i in range(len(r)):
	print(i)
	rate=percentage_in_range(GT3,PD3,r[i])
	head.append(rate[0])
	l_hand.append(rate[1])
	r_hand.append(rate[2])
	l_elbow.append(rate[3])
	r_elbow.append(rate[4])
	l_shoulder.append(rate[5])
	r_shoulder.append(rate[6])
plt.subplot(421)
plt.plot(r,head)
plt.title('RCH3')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand)
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand)
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow)
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow)
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder)
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder)
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()
