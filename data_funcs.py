import numpy as np
import scipy
import os
import cv2
import dlib

# GLOBALS
# basePath = "C:\\Users\\Paul\\Desktop\\Research\\PilotBlinkDetection\\"
basePath = os.path.dirname(os.path.realpath(__file__)) + "\\"
vidPath = basePath + "vids\\"
csvPath = basePath + "logs\\"
picPath = basePath + "pics\\"
preTrainedPath = basePath + "pre_trained_models\\"
savedModelPath = basePath + "saved_models\\"

detector2Path = preTrainedPath + "haarcascade_frontalface_default.xml"
detector3Path = preTrainedPath + "mmod_human_face_detector.dat"
detector4Path = preTrainedPath + "haarcascade_eye.xml"
shapePredPath = preTrainedPath + "shape_predictor_68_face_landmarks.dat"

TIMESTAMP = 0
EAR = 0
LABELED_BLINK = 1
F_VECTOR_LENGTH = 6


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
def openCv_rect_to_dlib(x,y,w,h):
    return dlib.rectangle(left=(x+w).item(), bottom=(y+h).item(), right=x.item(), top=y.item())

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