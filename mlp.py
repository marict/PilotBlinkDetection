# try using an mlp for the series analysis

import numpy as np
import math 
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from numpy import nan
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense

TIMESTAMP = 0
EAR = 1
GRADIENT = 2
LABELED_BLINK = 3
F_VECTOR_LENGTH = 3

basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"
vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"

detector2Path = preTrainedPath + "haarcascade_frontalface_default.xml"
detector3Path = preTrainedPath + "mmod_human_face_detector.dat"
shapePredPath = preTrainedPath + "shape_predictor_68_face_landmarks.dat"	



def extract_features_labels(tr):

	# normalize
	tr[:,0] = scipy.stats.zscore(tr[:,EAR])
	tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])

	# Maps indicies to blink or not
	blinkMap = np.array(tr[:,LABELED_BLINK],copy=True)

	# Mapping frames to their classification
	tr_vectors = []
	numblinks = 0

	# extract blinks 
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
		if(tr[i][LABELED_BLINK] == 1):
			blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
			# BALANCING DATA
			for z in range(100):
				numblinks += 1
				tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],1))
			
	# extract non blinks
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
		if(blinkMap[i] == 0):
			if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
				tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],0))

	pd.Series(blinkMap.flatten()).plot()
	
	X = []
	y = []
	for chunk, label in tr_vectors:
		y.append(label)
		# gradient in
		X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
	
	return X,y

def extract_features(tr):

	# normalize
	tr[:,0] = scipy.stats.zscore(tr[:,EAR])
	tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])
	
	tr_vectors = []
	# extract features with step length
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			tr_vectors.append(tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
	
	X = []
	for chunk in tr_vectors:
		X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
	X = np.array(X)
	
	y = (tr[:,LABELED_BLINK].astype(int))[F_VECTOR_LENGTH:len(tr)-F_VECTOR_LENGTH]
	
	return X,y
	
def extract_features_chunked(tr):
	# normalize
	tr[:,0] = scipy.stats.zscore(tr[:,EAR])
	tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])
	
	tr_vectors = []
	# extract features with step length
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			tr_vectors.append(tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
	
	X = []
	for chunk in tr_vectors:
		X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
	X = np.array(X)
	
	y = (tr[:,LABELED_BLINK].astype(int))[F_VECTOR_LENGTH:len(tr)-F_VECTOR_LENGTH]
	
	return X,y
	
def windowize_labels(labels):
	new_labels = np.zeros(labels.size)
	for i in range(F_VECTOR_LENGTH,len(labels)-F_VECTOR_LENGTH):
		if labels[i] == 1:
			new_labels[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1 

	return new_labels

print("loading in ears")
txt1 = csvPath + "planesweater1_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

print("loading in labels")
txt2 = csvPath + "planesweater1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw = np.hstack((ears,labels))

# remove nan entries
raw = raw[~np.isnan(raw).any(axis=1)]

print("extracting features")

# get features
y = []
X = []
for chunk, label in tr_vectors:
	y.append(label)
	# gradient in
	X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
	
X = np.array(X)
y = np.array(y)
print(y.reshape((-1,1)).shape)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=14))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X,y,epochs=50,verbose=1)
model.save_weights("D:\\blink-detection\\saved_models\\model.hdf5")

# now try on original video

y_predict = (np.asarray(model.predict(X_test)) > 0.5).astype(int)
print(np.unique(y_predict))
print("--")
print(np.unique(y_test))
recall = recall_score(y_test,y_predict)
precision = precision_score(y_test,y_predict)



# ---------------------- TEST VIDEO ------------------------

def test_model(fname):
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=14))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	model.load_weights("D:\\blink-detection\\saved_models\\model.hdf5", by_name=False)

	# import test video
	txt1 = csvPath + fname + "_ears.csv"
	ears3 = (pd.read_csv(txt1,sep=',',header=None).values)
	gradient = np.gradient(ears3[:,EAR]).reshape((-1,1))
	test_data = np.hstack((ears3,gradient))

	txt2 = csvPath + fname + "_labels.csv"
	labels3 = pd.read_csv(txt2,sep=',',header=None).values.reshape((-1,1))

	print(test_data.shape)
	print(labels3.shape)

	test_data = np.hstack((test_data,labels3))

	# remove nan entries
	test_data = test_data[~np.isnan(test_data).any(axis=1)]

	X_test,y_test = extract_features(test_data)

	y_predict = (np.asarray(model.predict(X_test)) > 0.5).astype(int)


	y_test = windowize_labels(y_test)
	pd.Series(y_predict.flatten()).plot()
	pd.Series(y_test.flatten()).plot()

	recall = recall_score(y_test,y_predict)
	precision = precision_score(y_test,y_predict)
	print("recall = " + str(recall))
	print("precision = " + str(precision))

	plt.show()
	
#fname = "planesweater3"
#test_model(fname)

# fname = "planesweater1"
# test_model(fname)

# timestamps = (ears[F_VECTOR_LENGTH:len(ears)-F_VECTOR_LENGTH])[:,TIMESTAMP].reshape((-1,1))











