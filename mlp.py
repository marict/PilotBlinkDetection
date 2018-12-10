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
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras.layers.recurrent import LSTM, GRU
from keras import regularizers

from data_funcs import *

# # raw is the raw windows
# def extract_features_labels_true(raw, use_gradient = False):
	# blinkMap = np.zeros(len(raw))
	# true_values = []
	# label = raw.shape[1] - 1
	
	# # mark off no 0 zones
	# for i in range(len(raw)):
		# if raw[i][label] == 1:
			# blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
	
	# true_values = []
	# # remove non-blinks that are within the no 0 zones
	# for i in range(len(raw)):
		# if raw[i][label] == 0:
			# if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
				# true_values.append(raw[i])
		# else:
			# true_values.append(raw[i])
				
	# return np.asarray(true_values)

# def extract_features_labels_raw(ts,use_gradient = False):

	# # normalize
	# ts[:,EAR] = scipy.stats.zscore(ts[:,EAR])
	# ts[:,GRADIENT] = scipy.stats.zscore(ts[:,GRADIENT])
	
	# tr_vectors = []
	# # extract features with step length
	# for i in range(F_VECTOR_LENGTH,len(ts)-F_VECTOR_LENGTH):
			# tr_vectors.append(ts[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
		
	# X = []
	# for chunk in tr_vectors:
		# if(use_gradient):
			# X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
		# else:
			# X.append([x[EAR] for x in chunk])
	# X = np.array(X)
	
	# y = (ts[:,LABELED_BLINK].astype(int))[F_VECTOR_LENGTH:len(ts)-F_VECTOR_LENGTH]
	# return np.hstack((X,y.reshape((-1,1))))
	
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
	
# # pull out blink number of non blinks
# def under_sample_balance(extracted_data):
	# X,y = X_y(extracted_data)
	# label = X.shape[1]
	
	# to_balance = np.hstack((X,y.reshape((-1,1))))

	# # balance
	# non_blinks = to_balance[to_balance[:,label] == 0]
	# blinks = to_balance[to_balance[:,label] == 1]
	# idx = np.random.randint(len(non_blinks),size=len(blinks))
	# non_blinks = non_blinks[idx]

	# to_balance = np.vstack((blinks,non_blinks))
	# X = to_balance[:,range(label)]
	# y = to_balance[:,label]
	
	# return np.hstack((X,y.reshape((-1,1))))

# # copy non blink number of blinks
# def over_sample_balance(extracted_data):
	# X,y = X_y(extracted_data)
	# to_balance = np.hstack((X,y.reshape((-1,1))))

	# # balance
	# non_blinks = to_balance[to_balance[:,14] == 0]
	# blinks = to_balance[to_balance[:,14] == 1]
	
	# balance_num = int(len(non_blinks)/len(blinks))
	# to_balance = np.concatenate((blinks,non_blinks),axis=0)
	
	# for i in range(balance_num):
		# to_balance = np.concatenate((to_balance,blinks),axis=0)
	
	# X = to_balance[:,range(14)]
	# y = to_balance[:,14]
	
	# return np.hstack((X,y.reshape((-1,1))))
	
	
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

def score(model,X,y):
	y_predict = (model.predict(X) > 0.5).astype(int)
	
	# true positives
	
	# false negatives 

	recall = recall_score(y,y_predict)
	precision = precision_score(y,y_predict)
	accuracy = accuracy_score(y,y_predict)
	
	print("\trecall: " + str(recall))
	print("\tPrecision: " + str(precision))
	print("\tAccuracy: " + str(accuracy))
	
	return y_predict

def model1():
	model = Sequential()
	model.add(Dense(100, activation='relu', input_dim=10))
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
	
TIMESTAMP = 0
EAR = 0
GRADIENT = 1
LABELED_BLINK = 2
F_VECTOR_LENGTH = 2

	
# print("loading in ears")
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
raw_features = extract_features_labels_raw(tr,True)
X,y = X_y(extract_features_labels_true(raw_features))

model = model1()
model.fit(X,y,epochs=50,verbose=1)

y_predict = score(model,X,y)

pd.Series(y_predict.flatten()).plot()
pd.Series(y.flatten()).plot()
plt.show()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)

raw_train = np.hstack((X_train,y_train.reshape((-1,1))))

# extract true values from train data
extracted_train = extract_features_labels_true(raw_train)
model = model1()
X_train,y_train = X_y(under_sample_balance(extracted_train))
print("training")
model.fit(X_train,y_train,epochs=100,verbose=0)
y_predict = score(model,X_test,y_test)

pd.Series(y_predict.flatten()).plot()
pd.Series(y_test.flatten()).plot()
plt.show()








