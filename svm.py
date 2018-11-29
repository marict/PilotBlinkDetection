import numpy as np
import math 
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from numpy import nan

# Other features to consider:
	# Derivative of EAR
	# Euclidean distance between eyebrow and eye

# SVM classifier
# Data formats:
# 	tr = [int timestamp,float ear,int labled_blinks]
# 	classification = [int timestamp,float ear]

TIMESTAMP = 0
EAR = 1
LABELED_BLINK = 2

basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"
vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"

detector2Path = preTrainedPath + "haarcascade_frontalface_default.xml"
detector3Path = preTrainedPath + "mmod_human_face_detector.dat"
shapePredPath = preTrainedPath + "shape_predictor_68_face_landmarks.dat"


class SVM:
	tr_vectors = None
	ts_vectors = None
	internal_model = None
	F_VECTOR_LENGTH = 3
	pp = pprint.PrettyPrinter(indent=4)
	
	def __init__(self,tr):
		# window size = 2 * F_VECTOR_LENGTH + 1 
		F_VECTOR_LENGTH = self.F_VECTOR_LENGTH
		
		if(len(tr) < 2*F_VECTOR_LENGTH + 1):
			raise ValueError("training data too small")
		

		# remove nan entries
		tr = tr[~np.isnan(tr).any(axis=1)]
		# normalize
		tr[:,1] = scipy.stats.zscore(tr[:,1])
		
		# calculate and store the derivative to add to the feature vectors
		dr = scipy.stats.zscore(np.gradient(tr[:,1]))

		print(tr.shape)
		tr = np.hstack((tr,dr.reshape((-1,1))))
		
		print(tr.shape)
	
		# SET UP TRAINING
		# Maps indicies to blink or not
		blinkMap = np.array(tr[:,2],copy=True)
		
		# Mapping frames to their classification
		self.tr_vectors = []
		
		numblinks = 0
		
		# extract blinks 
		for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			if(tr[i][LABELED_BLINK] == 1):
				blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
				# BALANCING DATA
				for z in range(100):
					numblinks += 1
					self.tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],1))
				
		# extract non blinks
		for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			if(blinkMap[i] == 0):
				if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
					self.tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],0))
		
		# clear memory
		del blinkMap
		
		#print(numblinks)
		# print(len(self.tr_vectors))
		
		#print(self.tr_vectors[0][0])
		
		
		
	
			
	
	# extracts the labels and features from sample array
	def get_labels_features(self):
		labels = [x[1] for x in self.tr_vectors]
		
		features = []
		for feature in [x[0] for x in self.tr_vectors]:
			features.append([x[1] for x in feature] + [x[3] for x in feature])
			
		self.pp.pprint(features[0])
		
		return labels,features
		
	# train on feature vectors
	def train(self):
		labels,features = self.get_labels_features()

		self.internal_model = svm.SVC(gamma='scale')
		self.internal_model.fit(features,labels)
		
	# cross validate
	def cv(self,fold=10):
		print("cross validating")
		
		labels,features = self.get_labels_features()		
		
		model_svm_linear = make_pipeline(svm.SVC(C=1, kernel='linear'))
		
		#Metric scores
		scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro'}
		  
		score_lr = cross_validate(model_svm_linear, features, labels, scoring=scoring, cv=fold, return_train_score=True,verbose =True)
		self.pp.pprint(score_lr)
	   
	# output classification on test data
	def classify(self,c_vector):  	
		if(self.internal_model is None):
			raise ValueError("Need to train model first")
		
		F_VECTOR_LENGTH = self.F_VECTOR_LENGTH
		
		# extract features
		c_vectors = []
		for i in range(F_VECTOR_LENGTH,len(c_vector)-F_VECTOR_LENGTH):
			c_vectors.append((c_vector[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]))	
		
		# get ear from test_features
		c_features = []
		for feature_list in c_vectors:
			c_features.append([x[1] for x in feature_list])
	
		prediction = self.internal_model.predict(c_features)
		print(prediction)
		
	# save weights
	def save(self,path):
		if(self.internal_model is None):
			raise ValueError("Need to train model first")
			
		date = datetime.datetime.today().strftime('%Y_%m_%d')
		print("saving model to " + path + date)
		joblib.dump(self.internal_model, path + date + ".joblib") 

	# load weights 
	def load(self,path_and_name):
		print("loading model from " + path_and_name)
		self.internal_model = joblib.load(path_and_name)
		
def test():
	# test training pre-processing
	# t1 = np.array([
		  # [50,1,0],
		  # [51,1,1],
		  # [52,1,0],
		  # [53,1,1],
		  # [54,1,0],
		  # # --- 
		  # [50,1,0],
		  # [51,1,1],
		  # [52,1,0],
		  # [53,1,1],
		  # [54,1,0],
		  # # --- 
		  # [55,1,0],
		  # [56,1,0],
		  # [57,0,0],
		  # [58,0,0],
		  # [59,0,0],
		  # # ---
		  # [55,1,0],
		  # [56,1,0],
		  # [57,0,0],
		  # [58,0,0],
		  # [59,0,0],
		  # # --- 
		  # [60,0,0],
		  # [61,0,0],
		  # [62,0,0]])
		  
	# test slicing
	# testSlicing()
	# svm = SVM(t1)
	#svm.train()
	# svm.cv(2)
	#svm.classify(t1[:,0:2])
	
	# current working directory	
	# cwd = os.path.dirname(os.path.realpath(__file__))
	# model_path = cwd + "\\saved_models\\"
	# svm.save(model_path)
	# svm2 = SVM(t1)
	# svm2.load("D:\\blink-detection\\saved_models\\2018_11_21.joblib")
	# svm2.cv(2)
	
	# # test training pre-processing
	# t2 = np.array([
		  # [50,nan,1],
		  # [51,1,1],
		  # [52,1,0],
		  # [53,1,1],
		  # [54,nan,0],
		  # # --- 
		  # [50,1,0],
		  # [51,1,1],
		  # [52,1,1],
		  # [53,1,1],
		  # [54,1,1],
		  # # --- 
		  # [55,1,0],
		  # [56,1,0],
		  # [57,nan,0],
		  # [58,nan,0],
		  # [59,nan,0],
		  # # ---
		  # [55,1,1],
		  # [56,1,1],
		  # [57,0,0],
		  # [58,0,0],
		  # [59,0,0],
		  # # --- 
		  # [60,0,0],
		  # [61,0,0],
		  # [62,0,0]])
		  
	# svm2 = SVM(t2)
	
	# try training svm on some actual data
	# pull out data and combine
	
	print("loading in ears")
	txt1 = csvPath + "planesweater1_ears.csv"
	ears = pd.read_csv(txt1,sep=',',header=None).values

	print("loading in labels")
	txt2 = csvPath + "planesweater1_labels.csv"
	labels = pd.read_csv(txt2,sep=',',header=None).values
	
	t3 = np.hstack((ears,labels))
	svm3 = SVM(t3)
	svm3.cv(10)
	
test()
