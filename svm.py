from sklearn import svm

# SVM classifier
# Data formats:
# 	tr = [int timestamp,float ear,int labled_blinks]
# 	ts = [int timestamp,float ear]

TIMESTAMP = 0
EAR = 1
LABELED_BLINK = 2

import numpy as np 
import scipy.stats

class SVM:
	tr_vectors = None
	ts_vectors = None
	internal_model = None
	
	def __init__(self,tr,ts = None):
		# window size = 2 * F_VECTOR_LENGTH + 1 
		F_VECTOR_LENGTH = 2
		
		if(len(tr) < 2*F_VECTOR_LENGTH + 1 or len(ts) < 2*F_VECTOR_LENGTH + 1):
			raise ValueError("training or testing data too small")

		# normalize data with Z-SCORE
		tr[:,1] = scipy.stats.zscore(tr[:,1])
		ts[:,1] = scipy.stats.zscore(ts[:,1])
	
		# SET UP TRAINING
		# Maps indicies to blink or not
		blinkMap = np.array(tr[:,2],copy=True)
		
		# Mapping frames to their classification
		self.tr_vectors = []
		
		# extract blinks 
		for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			if(tr[i][LABELED_BLINK] == 1):
				blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
				self.tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],1))
				
		# extract non blinks
		for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			if(blinkMap[i] == 0):
				if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
					self.tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],0))
		
		# -----------------------------------------------------------------------------
		
		# SET UP TESTING
		self.ts_vectors = []
		for i in range(F_VECTOR_LENGTH,len(ts)-F_VECTOR_LENGTH):
			self.ts_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]))
			
		# clear memory
		del blinkMap
		
	# train on feature vectors
	def train(self):
		labels = [x[1] for x in self.tr_vectors]
		
		features = []
		for feature in [x[0] for x in self.tr_vectors]:
			features.append([x[1] for x in feature])

		self.internal_model = svm.SVC(gamma='scale')
		self.internal_model.fit(features,labels)
		
	
	# output classification on test data
	def classify(self):  	
		if(self.internal_model is None):
			raise ValueError("Need to train model first")
		
		# get ear from test_features
		test_features = []
		for feature_list in self.ts_vectors:
			test_features.append([x[1] for x in feature_list])
	
		prediction = self.internal_model.predict(test_features)
		print(prediction)
		
def test():
	# test training pre-processing
	t1 = np.array([
		  [50,1,0],
		  [51,1,1],
		  [52,1,0],
		  [53,1,1],
		  [54,1,0],
		  # --- 
		  [55,1,0],
		  [56,1,0],
		  [57,0,0],
		  [58,0,0],
		  [59,0,0],
		  # --- 
		  [60,0,0],
		  [61,0,0],
		  [62,0,0]])
		  
	# test slicing
	# testSlicing()
	svm = SVM(t1,t1[:,0:2])
	svm.train()
	svm.classify()
	
	
test()
