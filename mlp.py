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

# Keras stuff
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import LSTM, Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras import regularizers

from data_funcs import *

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
	
	new_labels = np.zeros(labels.size)
	for i in range(F_VECTOR_LENGTH,len(labels)-F_VECTOR_LENGTH):
		if labels[i] == 1:
			new_labels[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1 

	return new_labels

# gives a test case
def test_tr():
	a = np.arange(300)
	b = np.zeros(300)
	b[4] = 1.0
	tr = np.vstack((a,a*10,a*100,b)).T
	
	return tr

def score(model,X,y):
	y_predict = (model.predict(X) > 0.5).astype(int)
	
	recall = recall_score(y,y_predict)
	precision = precision_score(y,y_predict)
	accuracy = accuracy_score(y,y_predict)
	
	print("\trecall: " + str(recall))
	print("\tPrecision: " + str(precision))
	print("\tAccuracy: " + str(accuracy))
	
	return y_predict

def model1():
	model = Sequential()
	model.add(Dense(100,input_dim=11))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	return model
	
def model2():
	model = Sequential() 
	model.add(Dense(64, input_dim=14)) 
	model.add(BatchNormalization()) 
	model.add(keras.layers.ReLU()) 
	model.add(Dense(2)) 
	model.add(Activation('softmax'))
	model.compile(optimizer='adam', loss='mse')
	return model
	
	
# # MY DATA ------------------
# # print("loading in ears")
# # txt1 = csvPath + "planesweater1_ears.csv"
# # ears = pd.read_csv(txt1,sep=',',header=None).values
# # gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
# # ears = np.hstack((ears,gradient))

# # print("loading in labels")
# # txt2 = csvPath + "planesweater1_labels.csv"
# # labels = pd.read_csv(txt2,sep=',',header=None).values
# # raw = np.hstack((ears,labels))

# STEVEN'S DATA ----------------

print("loading in ears")
txt1 = csvPath + "1_EAR.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values[0:18200]
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

print("loading in labels")
txt2 = csvPath + "1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:18200]
raw = np.hstack((ears,labels))

# remove nan entries
tr = raw[~np.isnan(raw).any(axis=1)]

# VALIDATE ON EXTRACTED DATA 
raw_features = extract_features_labels_raw(tr)

X,y = X_y(raw_features)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
X_train,y_train = X_y(extract_features_labels_true(np.hstack((X_train,y_train.reshape(-1,1)))))

model = model1()
model.fit(X_train,y_train,epochs=500)
y_predict = score(model,X_test,y_test)

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()








