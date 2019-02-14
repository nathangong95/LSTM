import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import cv2

heatmap_path='../../Heatmaps/'
Q=np.load('Data/Q.npy')
print(Q.shape)
R=np.load('Data/R.npy')
print(R.shape)
joints=np.load('Data/jointsfromheatmap.npy')
print(joints.shape)

def normalize_heatmap(heatmap):
	"""
		heatmap:256*256
	"""
	minval=min(heatmap.flatten())
	heatmap=heatmap-minval
	s=heatmap.sum(axis=0).sum()
	normalized_heatmap=heatmap/float(s)
	return normalized_heatmap
def sampling_from_heatmap(normalized_heatmap):
	"""
		heatmap:256*256
	"""
	col_prob=np.sum(normalized_heatmap,axis=0)
	thres = float(np.random.uniform(0,1,1))
	sum_prob=0
	y=0
	for i in range(normalized_heatmap.shape[0]):
		sum_prob+=col_prob[i]
		if sum_prob>=thres:
			y=i
			break
	row_prob=np.sum(normalized_heatmap,axis=1)
	thres = float(np.random.uniform(0,1,1))
	sum_prob=0
	x=0
	for i in range(normalized_heatmap.shape[0]):
		sum_prob+=row_prob[i]
		if sum_prob>=thres:
			x=i
			break
	return [x,y]
def importance_sampling(particles, weight):
	"""
		particles: list of num_particles
		weight: list of num_particles
	"""
	thres = float(np.random.uniform(0,1,1))
	sum_prob=0
	#print('aaaaaaaaaaaaaaaaaaaaaa')
	#print(sum(weight))
	for i in range(len(weight)):
		sum_prob+=weight[i]
		if sum_prob>=thres:
			return particles[i]
def particle_filter(joints, heatmap_path, Q, num_particles):
	"""
		joints: 3000*7*2
		Q: 4*4*7
	"""
	dt=1
	A=np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
	B=np.array([[1,0,0,0],[0,1,0,0]])
	joints_particle=np.zeros((3000,7,2))
	for j in range(1,7):
		Q_=Q[:,:,j]
		for i in range(joints.shape[0]):
			print(i)
			heatmaps=np.load(heatmap_path+str(i+1)+'.npy')#7*256*256
			heatmap=normalize_heatmap(heatmaps[j,:,:])
			if i == 0:
				particles=[]
				for k in range(num_particles):
					pos=sampling_from_heatmap(heatmap)
					pos.append(0)
					pos.append(0)
					particles.append(np.reshape(np.asarray(pos),(4,1)))
				weight=[]
				for part in particles:
					weight.append(heatmap[part[0,0],part[1,0]])#
				weight=weight/sum(weight)
				joints_particle[0,j,:]=joints[0,j,:]#Change here for different first frame method
			else:
				new_particles=[]
				for k in range(num_particles):
					particle_k=importance_sampling(particles, weight)
					new_particles.append(np.reshape(np.random.multivariate_normal(np.reshape(np.matmul(A,particle_k),(4,)), Q_, 1),(4,1)))
				for k in range(num_particles):
					new_particles[k][0,0]=int(new_particles[k][0,0])
					new_particles[k][1,0]=int(new_particles[k][1,0])
					if new_particles[k][0,0]>255:
						new_particles[k][0,0]=255
					if new_particles[k][1,0]>255:
						new_particles[k][1,0]=255
					if new_particles[k][0,0]<0:
						new_particles[k][0,0]=0
					if new_particles[k][1,0]<0:
						new_particles[k][1,0]=0
				particles=new_particles
				weight=[]
				for part in particles:
					weight.append(heatmap[int(part[0,0]),int(part[1,0])])#
				weight=weight/sum(weight)
				sum_x=0
				sum_y=0
				for x in range(len(particles)):
					sum_x+=particles[x][0]
					sum_y+=particles[x][1]
				joints_particle[i,j,0]=sum_x/float(num_particles)
				joints_particle[i,j,1]=sum_y/float(num_particles)
			
			#uncomment to visualize each step
			
			print('joints is')
			print(joints_particle[i,j,:])
			print('heatmap is')
			print(np.where(heatmap==heatmap.max()))
			fig=plt.figure()
			ax = fig.gca()
			ax.imshow(heatmap)
			for part in particles:
				ax.plot(part[1],part[0], marker='o', markersize=3, color="red")
			#ax.colorbar()
			plt.show()
			
	return joints_particle


joints_particlefilter=particle_filter(joints, heatmap_path, Q, 100)
np.save('joints_particle.npy',joints_particlefilter)