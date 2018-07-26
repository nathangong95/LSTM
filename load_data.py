""" Module that implements the load_data class
Author: Chenghao Gong
Date: 7/20/2018
Version: 1.0
"""
import numpy as np
import simjoints as sj
class load_data:
	""" This class handle the load_data object
	Params: data_path (str): the path to the data folder
	"""
	def _int_(self,data_path):
		""" Constructor
		Args:
			data_path (str): the path to the data folder
		Returns:
			None
		"""
		self.data_path=data_path
	def load(self):
		"""
		"""
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

	def split(self):
