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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
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
	print("extracing features, labels")

	# normalize
	tr[:,EAR] = scipy.stats.zscore(tr[:,EAR])
	tr[:,GRADIENT] = scipy.stats.zscore(tr[:,GRADIENT])

	# Maps indicies to blink or not
	blinkMap = np.array(tr[:,LABELED_BLINK],copy=True)

	# Mapping frames to their classification
	tr_vectors = []

	num_blinks = 0

	# extract blinks 
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
		if(tr[i][LABELED_BLINK] == 1):
			blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
			tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],1,i))	
			num_blinks += 1

	# extract non blinks
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
		if(blinkMap[i] == 0):
			if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
				tr_vectors.append((tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],0,i))
			
	tr_vectors = sorted(tr_vectors,key=lambda x: x[2])

	X = []
	y = []
	for chunk, label, index in tr_vectors:
		y.append(label)
		# gradient in
		X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
		
	X = np.asarray(X)
	y = np.asarray(y)
	
	return np.hstack((X,y.reshape((-1,1))))

def extract_features(tr):

	# normalize
	tr[:,EAR] = scipy.stats.zscore(tr[:,EAR])
	tr[:,GRADIENT] = scipy.stats.zscore(tr[:,GRADIENT])
	
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
	
def extract_features_naive(tr):
	# normalize
	tr[:,0] = scipy.stats.zscore(tr[:,EAR])
	tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])
	
	tr_vectors = []
	# extract features with step length
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH):
			tr_vectors.append(tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1],tr[i][LABELED_BLINK])
	
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

def test_model(fname):

	print("testing model")
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=14))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	model.load_weights("D:\\blink-detection\\saved_models\\model2.hdf5", by_name=False)

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

	pd.Series(y_predict.flatten()).plot()
	pd.Series(y_test.flatten()).plot()

	recall = recall_score(y_test,y_predict)
	precision = precision_score(y_test,y_predict)
	print("recall = " + str(recall))
	print("precision = " + str(precision))

	plt.show()
	
def under_sample_balance(extracted_data):
	X,y = X_y(extracted_data)
	
	to_balance = np.hstack((X,y.reshape((-1,1))))

	# balance
	non_blinks = to_balance[to_balance[:,14] == 0]
	blinks = to_balance[to_balance[:,14] == 1]
	idx = np.random.randint(len(non_blinks),size=len(blinks))
	non_blinks = non_blinks[idx]

	to_balance = np.vstack((blinks,non_blinks))
	X = to_balance[:,range(14)]
	y = to_balance[:,14]
	
	return np.hstack((X,y.reshape((-1,1))))

# pulls off the last column as labels, the rest as feature vectors
def X_y(data):
	n = data.shape[1]
	X = data[:,range(n-1)]
	y = data[:,n-1]
	return X,y
	
# gives a test case
def test_tr():
	a = np.arange(300)
	b = np.zeros(300)
	b[4] = 1.0
	tr = np.vstack((a,a*10,a*100,b)).T
	
	return tr
	
	
# print("loading in ears")
txt1 = csvPath + "planesweater1_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

print("loading in labels")
txt2 = csvPath + "planesweater1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw = np.hstack((ears,labels))

# remove nan entries
tr = raw[~np.isnan(raw).any(axis=1)]

# ----- get x,y ---- 

# normalize
tr[:,0] = scipy.stats.zscore(tr[:,EAR])
tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])

extracted_data = extract_features_labels(tr)
sampled_data = under_sample_balance(extracted_data)

np.random.shuffle(sampled_data)
X,y = X_y(sampled_data)

# -----------

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=14))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X,y,epochs=300,verbose=0)
model.save_weights("D:\\blink-detection\\saved_models\\model2.hdf5")
# model.load_weights("D:\\blink-detection\\saved_models\\model2.hdf5")

y_predict = (model.predict(X) > 0.5).astype(int)
pd.Series(y_predict.flatten()).plot()
pd.Series(y.flatten()).plot()
plt.show()

recall = recall_score(y,y_predict)
precision = precision_score(y,y_predict)
accuracy = accuracy_score(y,y_predict)

print("total blinks predicted: " + str(np.sum(y_predict)))
print(recall)
print(precision)
print(accuracy)


# STORAGE ---------------


# Over sample:
# balance_num = int(abs(len(tr_vectors)-num_blinks)/num_blinks)
# n = len(tr_vectors)
# new_rows = []
# for i in range(n):
	# chunk, label, index = tr_vectors[i]
	# if(label == 1):
		# for j in range(balance_num):
			# new_rows.append(tr_vectors[i])
# tr_vectors += new_rows








