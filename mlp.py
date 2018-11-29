# try using an mlp for the series analysis

import numpy as np
import math 
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
from numpy import nan
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

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


print("loading in ears")
txt1 = csvPath + "planesweater1_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))


print("loading in labels")
txt2 = csvPath + "planesweater1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
tr = np.hstack((ears,labels))

# remove nan entries
tr = tr[~np.isnan(tr).any(axis=1)]

print("extracting features")

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

	# clear memory
	del blinkMap
	
	return tr_vectors

def extract_features(tr):
	# remove nan entries
	tr = tr[~np.isnan(tr).any(axis=1)]
	# normalize
	tr[:,0] = scipy.stats.zscore(tr[:,EAR])
	tr[:,1] = scipy.stats.zscore(tr[:,GRADIENT])
	
	tr_vectors = []
	# extract features 
	for i in range(F_VECTOR_LENGTH,len(tr)-F_VECTOR_LENGTH,F_VECTOR_LENGTH*2+1):
		tr_vectors.append(tr[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
	
	return tr_vectors

tr_vectors = extract_features_labels(tr)

# get features
y = []
X = []
for chunk, label in tr_vectors:
	y.append(label)
	# gradient in
	X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
	# gradient out
	#X.append([x[0] for x in chunk])
	# only gradient 
	#X.append([x[1] for x in chunk])
	
X = np.array(X)
y = np.array(y)
print(y.reshape((-1,1)).shape)


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.20,random_state=42)
print(X_train)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))


# define model
from keras.models import Sequential
from keras.layers import Dense
print("defining model")
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=14))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit(X_train,y_train,epochs=50,verbose=1)

y_predict = (np.asarray(model.predict(X_test)) > 0.5).astype(int)
print(np.unique(y_predict))
print("--")
print(np.unique(y_test))
recall = recall_score(y_test,y_predict)
precision = precision_score(y_test,y_predict)

print(recall)
print(precision)

# ---------------------- TEST VIDEO ------------------------

# import test video
txt1 = csvPath + "planesweater3_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

# remove nan entries
ears = ears[~np.isnan(ears).any(axis=1)]

tr_vectors = extract_features(ears)

X_test = []
for chunk in tr_vectors:
	X_test.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
X_test = np.array(X_test)
y_predict = (np.asarray(model.predict(X_test)) > 0.5).astype(int)

timestamps = (ears[F_VECTOR_LENGTH:len(ears)-F_VECTOR_LENGTH])[:,TIMESTAMP].reshape((-1,1))


# y_predict_padded = np.insert(y_predict,list(range(len(y_predict))),[0,0,0])
y_predict_padded = np.zeros(timestamps.size)
y_predict_padded[0::7] = y_predict.flatten()

print(timestamps.shape)
print(y_predict_padded.shape)

ts_predictions = np.hstack((timestamps,y_predict_padded.reshape((-1,1))))

print(ts_predictions)
np.savetxt(csvPath + "planesweater3_pred.csv", ts_predictions, delimiter=",")










