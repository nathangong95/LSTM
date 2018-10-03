import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
14: left lower length
15: right lower length
16: left upper length
17: right upper length
18: left elbow angle
19: right elbow angle
20: left shoulder angle
21: right shoulder angle
22: label
23: time stamp
"""
def subtractor(data):
	""" Helper function to subtract each joint with the center
	Args:
		data (14*n array): input joints position data
	returns:
	"""
	output=[]
	for dat in data:
		center=((dat[10]+dat[12])/2.0, (dat[11]+dat[13])/2.0)
		for i in range(14):
			if i%2==0:
				dat[i]-=center[0]
			else:
				dat[i]-=center[1]
		output.append(dat)
	return np.asarray(output)
def polar(centered_data):
	output=[]
	for data in centered_data.T:
		temp=[]
		for i in range(7):
			x=data[2*i]
			y=data[2*i+1]
			rho = np.sqrt(x**2 + y**2)
			phi = np.arctan2(y, x)
			temp.append([rho,phi])
		output.append(temp)
	return output
def mjd(data, window_size=20, option=0):
	num_joints=7
	centered_data = subtractor(data)
	polar_data=polar(centered_data)
	assert(option==0 or option==1)
	if option==0:
		matrix_listR=[]
		matrix_listB=[]
		for i in range(len(polar_data)):
			windowR=np.zeros((num_joints,window_size))
			windowB=np.zeros((num_joints,window_size))
			if i < window_size:
				matrix_listR.append(windowR)
				matrix_listB.append(windowB)
				continue
			for j in range(num_joints):
				for k in range(window_size):
					windowR[j,k]=polar_data[i-window_size+k][j][1]
					windowB[j,k]=polar_data[i-window_size+k][j][0]
			matrix_listR.append(windowR)
			matrix_listB.append(windowB)
	else:
		matrix_listR=[]
		matrix_listB=[]
		for i in range(len(polar_data)):
			if i==0:
				continue
			if i%window_size!=0:
				continue
			windowR=np.zeros((num_joints,window_size))
			windowB=np.zeros((num_joints,window_size))
			for j in range(num_joints):
				for k in range(window_size):
					windowR[j,k]=polar_data[i-window_size+k][j][1]
					windowB[j,k]=polar_data[i-window_size+k][j][0]
			matrix_listR.append(windowR)
			matrix_listB.append(windowB)
	return matrix_listR, matrix_listB

def demo(data_path):
	filelist=os.listdir(data_path)
	train_data=[]
	train_label=[]
	filelist.sort()
	print(filelist)
	data=np.genfromtxt(data_path+filelist[0],delimiter=',')
	R,B=mjd(data,20,1)
	num_joint,window_size=R[1].shape
	G=[np.zeros((num_joint,window_size))]*(len(R))
	ims = []
	fig = plt.figure()
	for i in range(len(R)):
		image=np.zeros((num_joint,window_size,3))
		image[:,:,0]=R[i]
		image[:,:,1]=G[i]
		image[:,:,2]=B[i]
		img=plt.imshow(image,animated=True)
		ims.append([img])
		print(i)
	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
	plt.show()

if __name__=="__main__":
	data_path="/home/nathan/WorkSpace/LSTM/Data22D/"
	demo(data_path)