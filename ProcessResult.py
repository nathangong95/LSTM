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
	rate_navie=[]
	GT=np.reshape(GT, (GT.shape[0],7,2))
	PD=np.reshape(PD, (PD.shape[0],7,2))
	for i in range(7):
		hit=0
		hit_navie=0
		for j in range(1,GT.shape[0]):
			distance=((GT[j,i,0]-PD[j,i,0])**2+(GT[j,i,1]-PD[j,i,1])**2)**0.5
			distance_navie=((GT[j,i,0]-GT[j-1,i,0])**2+(GT[j,i,1]-GT[j-1,i,1])**2)**0.5
			if distance<=r:
				hit+=1
			if distance_navie<=r:
				hit_navie+=1
		rate.append(hit/(GT.shape[0]-1))
		rate_navie.append(hit_navie/(GT.shape[0]-1))
	return rate, rate_navie#list with len 7

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
	rate,rate_navie=percentage_in_range(GT1,PD1,r[i])
	head.append([rate[0],rate_navie[0]])
	l_hand.append([rate[1],rate_navie[1]])
	r_hand.append([rate[2],rate_navie[2]])
	l_elbow.append([rate[3],rate_navie[3]])
	r_elbow.append([rate[4],rate_navie[4]])
	l_shoulder.append([rate[5],rate_navie[5]])
	r_shoulder.append([rate[6],rate_navie[6]])
head=np.asarray(head)
l_hand=np.asarray(l_hand)
r_hand=np.asarray(r_hand)
l_elbow=np.asarray(l_elbow)
r_elbow=np.asarray(r_elbow)
l_shoulder=np.asarray(l_shoulder)
r_shoulder=np.asarray(r_shoulder)
plt.subplot(421)
plt.plot(r,head[:,0],r,head[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.title('NY531')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand[:,0],r,l_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand[:,0],r,r_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow[:,0],r,l_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow[:,0],r,r_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder[:,0],r,l_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder[:,0],r,r_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()


head=[]
l_hand=[]
r_hand=[]
l_elbow=[]
r_elbow=[]
l_shoulder=[]
r_shoulder=[]
for i in range(len(r)):
	print(i)
	rate,rate_navie=percentage_in_range(GT2,PD2,r[i])
	head.append([rate[0],rate_navie[0]])
	l_hand.append([rate[1],rate_navie[1]])
	r_hand.append([rate[2],rate_navie[2]])
	l_elbow.append([rate[3],rate_navie[3]])
	r_elbow.append([rate[4],rate_navie[4]])
	l_shoulder.append([rate[5],rate_navie[5]])
	r_shoulder.append([rate[6],rate_navie[6]])
head=np.asarray(head)
l_hand=np.asarray(l_hand)
r_hand=np.asarray(r_hand)
l_elbow=np.asarray(l_elbow)
r_elbow=np.asarray(r_elbow)
l_shoulder=np.asarray(l_shoulder)
r_shoulder=np.asarray(r_shoulder)
plt.subplot(421)
plt.plot(r,head[:,0],r,head[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.title('RCH1')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand[:,0],r,l_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand[:,0],r,r_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow[:,0],r,l_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow[:,0],r,r_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder[:,0],r,l_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder[:,0],r,r_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()


head=[]
l_hand=[]
r_hand=[]
l_elbow=[]
r_elbow=[]
l_shoulder=[]
r_shoulder=[]
for i in range(len(r)):
	print(i)
	rate,rate_navie=percentage_in_range(GT3,PD3,r[i])
	head.append([rate[0],rate_navie[0]])
	l_hand.append([rate[1],rate_navie[1]])
	r_hand.append([rate[2],rate_navie[2]])
	l_elbow.append([rate[3],rate_navie[3]])
	r_elbow.append([rate[4],rate_navie[4]])
	l_shoulder.append([rate[5],rate_navie[5]])
	r_shoulder.append([rate[6],rate_navie[6]])
head=np.asarray(head)
l_hand=np.asarray(l_hand)
r_hand=np.asarray(r_hand)
l_elbow=np.asarray(l_elbow)
r_elbow=np.asarray(r_elbow)
l_shoulder=np.asarray(l_shoulder)
r_shoulder=np.asarray(r_shoulder)
plt.subplot(421)
plt.plot(r,head[:,0],r,head[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.title('RCH3')
plt.ylabel('head accuracy')
plt.xlabel('distance')
plt.subplot(422)
plt.plot(r,l_hand[:,0],r,l_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l hand accuracy')
plt.xlabel('distance')
plt.subplot(423)
plt.plot(r,r_hand[:,0],r,r_hand[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r hand accuracy')
plt.xlabel('distance')
plt.subplot(424)
plt.plot(r,l_elbow[:,0],r,l_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l elbow accuracy')
plt.xlabel('distance')
plt.subplot(425)
plt.plot(r,r_elbow[:,0],r,r_elbow[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r elbow accuracy')
plt.xlabel('distance')
plt.subplot(426)
plt.plot(r,l_shoulder[:,0],r,l_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('l shoulder accuracy')
plt.xlabel('distance')
plt.subplot(427)
plt.plot(r,r_shoulder[:,0],r,r_shoulder[:,1])
plt.gca().legend(('LSTM','Naive'))
plt.ylabel('r shoulder accuracy')
plt.xlabel('distance')
plt.show()
