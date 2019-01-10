# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
from operator import itemgetter
from numpy import nan

import numpy as np
import argparse
import imutils
from imutils.video import VideoStream
import time
import dlib
import cv2
import sys,os
import importlib
import pdb

import numpy as np
import scipy
import os
import cv2
import dlib

# try using an svm for the series analysis
import numpy as np
import math
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
import pickle
import pdb
import matplotlib.pyplot as plt

from scipy.stats import uniform, randint
from numpy import *

# sklearn
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.base import clone
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Keras stuff
# import keras
# from keras.models import Sequential
# from keras.layers.core import Dense, Dropout, Activation, Flatten
# from keras.layers.normalization import BatchNormalization
# from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
# from keras.optimizers import RMSprop, Adam, SGD, Nadam
# from keras.layers.advanced_activations import *
# from keras.layers import LSTM, Convolution1D, MaxPooling1D, AtrousConvolution1D
# from keras import regularizers
# from keras.wrappers.scikit_learn import KerasClassifier
# import functools
# from keras import backend as K

# GLOBALS
# basePath = "C:\\Users\\Paul\\Desktop\\Research\\PilotBlinkDetection\\"
# basePath = "D:\\blink-detection\\"
basePath = "D:\\Research\\PilotBlinkDetection\\"
vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"
savedModelPath = basePath + "saved_models\\"

detector2Path = preTrainedPath + "haarcascade_frontalface_default.xml"
detector3Path = preTrainedPath + "mmod_human_face_detector.dat"
detector4Path = preTrainedPath + "haarcascade_eye.xml"

detector5Path1 = preTrainedPath + "deploy.prototxt"
detector5Path2 = preTrainedPath + "res10_300x300_ssd_iter_140000.caffemodel"

shapePredPath = preTrainedPath + "shape_predictor_68_face_landmarks.dat"

TIMESTAMP = 0
EAR = 0
LABELED_BLINK = 1
F_VECTOR_LENGTH = 6

# Converts np array to tuple
def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

# Remove's the areas of the data
# where a face was not found by a window chunk
# amount to preserve data integrity
def remove_nans(raw):
    print("not completed")

# converts dlib rectangle to openCV rectangle
# Dealing with inclusive exclusive boundary
def dlib_rect_to_openCv(r):
    return r.top(), r.left(),r.right() - r.left(), r.bottom() - r.top()

# converts dlib rectangle to openCV rectangle
# Dealing with inclusive exclusive boundary
def openCv_rect_to_dlib(r):
    (x,y,w,h) = r
    #return dlib.rectangle(left=(x+w).item(), bottom=(y+h).item(), right=x.item(), top=y.item())
    return dlib.rectangle(right=(x+w).item(), bottom=(y+h).item(), left=x.item(), top=y.item())
   
# Identifies the apex of blinks as middle of eye-closed predictions
def get_apexes(y_predict):
    apexes = np.zeros(len(y_predict))
    i = 0 
    while(i < len(y_predict)):
        if(y_predict[i] == 1):
            j = i+1
            while(j < len(y_predict) and y_predict[j] == 1):
                j += 1
            apexes[int((i+j-1)/2)] = 1
            i = j
        i += 1 
    return apexes
   
# Smooths prediction data with time window
def smooth(y_predict,WINDOW=30):
    for i in range(len(y_predict)):
        if(y_predict[i] == 1):
            for j in range(WINDOW):
                if(i+j < len(y_predict)):
                    if(y_predict[i+j] == 1):
                        y_predict[i:i+j] = 1
   
# Scores a model
def score(model,X,y):
    y_predict = (model.predict(X) > 0.5).astype(int)
    
    print("\ttotal labels predicted: " + str(np.sum(y_predict)))
    recall = recall_score(y,y_predict)
    precision = precision_score(y,y_predict)
    accuracy = accuracy_score(y,y_predict)

    print("\trecall: " + str(recall))
    print("\tPrecision: " + str(precision))
    print("\tAccuracy: " + str(accuracy))

    return y_predict
  
# remove data that has a NULL (zero) frame in it
# should be called AFTER augmentation to prevent bleeding
# from two disparate parts of the data set
# Note that this has not had a very noticable effect
def remove_no_faces(X,Y,WINDOW = 2):
    pdb.set_trace()
    X_out = []
    y_out = []
    n = int(X.shape[1]/(WINDOW * 4))
    null_frame = "0" * n
    for x,y in zip(X,Y):
        x_str = ''.join(str(int(i)) for i in x)
        if null_frame not in x_str:
            X_out.append(x)
            y_out.append(y)
            
    pdb.set_trace()
    return np.asarray(X_out), np.asarray(y_out)
  
# augments landmark data with a rolling window
def augment_landmarks_window(X,WINDOW = 2):
    new_X = np.zeros((X.shape[0],X.shape[1] * WINDOW * 2))
    for i in range(WINDOW,len(X)-WINDOW):
        new_X[i] = np.hstack(X[i-WINDOW:i+WINDOW])
    return new_X
   
# raw is the raw windows
# extracts true blinks and true non-blinks for training
def extract_features_labels_true(raw):

	blinkMap = np.zeros(len(raw))
	true_values = []
	label = raw.shape[1] - 1

	# mark off no 0 zones
	for i in range(len(raw)):
		if raw[i][label] == 1:
			blinkMap[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1

	true_values = []
	# remove non-blinks that are within the no 0 zones
	for i in range(len(raw)):
		if raw[i][label] == 0:
			if(1 not in blinkMap[i - F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1]):
				true_values.append(raw[i])
		else:
			true_values.append(raw[i])

	return np.asarray(true_values)

# extracts features in rolling window fashion
def extract_features_labels_raw(ts):

	# normalize
	ts[:,EAR] = scipy.stats.zscore(ts[:,EAR])

	tr_vectors = []
	# extract features with step length
	for i in range(F_VECTOR_LENGTH,len(ts)-F_VECTOR_LENGTH):
			tr_vectors.append(ts[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
	X = []
	for chunk in tr_vectors:
			X.append([x[EAR] for x in chunk])
	X = np.array(X)

	y = (ts[:,LABELED_BLINK].astype(int))[F_VECTOR_LENGTH:len(ts)-F_VECTOR_LENGTH]
	return np.hstack((X,y.reshape((-1,1))))

def under_sample_balance(X,y):
	n = X.shape[1]
	to_balance = np.hstack((X,y.reshape((-1,1))))

	# balance
	non_blinks = to_balance[to_balance[:,n] == 0]
	blinks = to_balance[to_balance[:,n] == 1]
	idx = np.random.randint(len(non_blinks),size=len(blinks))
	non_blinks = non_blinks[idx]

	to_balance = np.vstack((blinks,non_blinks))
	X = to_balance[:,range(n)]
	y = to_balance[:,n]

	return X,y

# copy non blink number of blinks
def over_sample_balance(X,y):
	n = X.shape[1]
	to_balance = np.hstack((X,y.reshape((-1,1))))

	# balance
	non_blinks = to_balance[to_balance[:,n] == 0]
	blinks = to_balance[to_balance[:,n] == 1]

	balance_num = int(len(non_blinks)/len(blinks))
	to_balance = np.concatenate((blinks,non_blinks),axis=0)

	for i in range(balance_num):
		to_balance = np.concatenate((to_balance,blinks),axis=0)

	X = to_balance[:,range(n)]
	y = to_balance[:,n]

	return X,y

# pulls off the last column as labels, the rest as feature vectors
def X_y(data):
	n = data.shape[1]
	X = data[:,range(n-1)]
	y = data[:,n-1]
	return X,y