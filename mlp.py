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

LABELED_BLINK = 2
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
# get rid of timestamp 
ears = ears[:,1]
ears = np.vstack((ears,np.gradient(ears))).T
print(ears.shape)

print("loading in labels")
txt2 = csvPath + "planesweater1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
tr = np.hstack((ears,labels))

print("extracting features")
# window size = 2 * F_VECTOR_LENGTH + 1 
F_VECTOR_LENGTH = F_VECTOR_LENGTH

# remove nan entries
tr = tr[~np.isnan(tr).any(axis=1)]
# normalize
tr[:,0] = scipy.stats.zscore(tr[:,0])
tr[:,1] = scipy.stats.zscore(tr[:,1])

# Maps indicies to blink or not
blinkMap = np.array(tr[:,2],copy=True)

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

print(tr_vectors[0])
# get features
y = []
X = []
for chunk, label in tr_vectors:
	y.append(label)
	# gradient in
	X.append([x[0] for x in chunk] + [x[1] for x in chunk])
	# gradient out
	#X.append([x[0] for x in chunk])
	# only gradient 
	#X.append([x[1] for x in chunk])
	
X = np.array(X)
y = np.array(y)
print(y.reshape((-1,1)).shape)


X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.20,random_state=41)
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

# import test video
txt1 = csvPath + "planesweater3_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values

ears = ears[~np.isnan(ears).any(axis=1)]


