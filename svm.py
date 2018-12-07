# try using an svm for the series analysis
import numpy as np
import math 
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn import svm
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from numpy import nan


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

# raw is the raw windows
def extract_features_labels_true(raw, use_gradient = False):
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
def extract_features_labels_raw(ts,use_gradient = False):

	# normalize
	ts[:,EAR] = scipy.stats.zscore(ts[:,EAR])
	ts[:,GRADIENT] = scipy.stats.zscore(ts[:,GRADIENT])
	
	tr_vectors = []
	# extract features with step length
	for i in range(F_VECTOR_LENGTH,len(ts)-F_VECTOR_LENGTH):
			tr_vectors.append(ts[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1])
		
	X = []
	for chunk in tr_vectors:
		if(use_gradient):
			X.append([x[EAR] for x in chunk] + [x[GRADIENT] for x in chunk])
		else:
			X.append([x[EAR] for x in chunk])
	X = np.array(X)
	
	y = (ts[:,LABELED_BLINK].astype(int))[F_VECTOR_LENGTH:len(ts)-F_VECTOR_LENGTH]
	return np.hstack((X,y.reshape((-1,1))))
	
def under_sample_balance(extracted_data):
	X,y = X_y(extracted_data)
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
	
	return np.hstack((X,y.reshape((-1,1))))

# copy non blink number of blinks
def over_sample_balance(extracted_data):
	X,y = X_y(extracted_data)
	to_balance = np.hstack((X,y.reshape((-1,1))))

	# balance
	non_blinks = to_balance[to_balance[:,14] == 0]
	blinks = to_balance[to_balance[:,14] == 1]
	
	balance_num = int(len(non_blinks)/len(blinks))
	to_balance = np.concatenate((blinks,non_blinks),axis=0)
	
	for i in range(balance_num):
		to_balance = np.concatenate((to_balance,blinks),axis=0)
	
	X = to_balance[:,range(14)]
	y = to_balance[:,14]
	
	return np.hstack((X,y.reshape((-1,1))))

# pulls off the last column as labels, the rest as feature vectors
def X_y(data):
	n = data.shape[1]
	X = data[:,range(n-1)]
	y = data[:,n-1]
	return X,y

# cv	
def cross_val(X,y):
	pp = pprint.PrettyPrinter(indent=4)
	
	#Metric scores
	scoring = {'acc': 'accuracy',
	   'prec_macro': 'precision_macro',
	   'rec_macro': 'recall_macro'}
	   
	model_svm_linear = make_pipeline(svm.SVC(C=1, kernel='rbf',gamma='scale',class_weight="balanced"))
	
	score_lr = cross_validate(model_svm_linear, X, y, scoring=scoring, cv=5, return_train_score=True)
	pp.pprint(score_lr)
	
def score(model,X,y):
	y_predict = model.predict(X)
	print("\ttotal blinks predicted: " + str(np.sum(y_predict)))
	recall = recall_score(y,y_predict)
	precision = precision_score(y,y_predict)
	accuracy = accuracy_score(y,y_predict)

	print("\trecall: " + str(recall))
	print("\tPrecision: " + str(precision))
	print("\tAccuracy: " + str(accuracy))

	return y_predict

# STEVEN'S DATA ----------------
TIMESTAMP = 0
EAR = 0
GRADIENT = 1
LABELED_BLINK = 2
F_VECTOR_LENGTH = 3

print("loading in ears")
txt1 = csvPath + "1_EAR.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values[0:18200]
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

print("loading in labels")
txt2 = csvPath + "1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:18200]
raw = np.hstack((ears,labels))

# # MY DATA ------------------

# # print("loading in ears")
# txt1 = csvPath + "planesweater1_ears.csv"
# ears = pd.read_csv(txt1,sep=',',header=None).values
# gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
# ears = np.hstack((ears,gradient))

# # print("loading in labels")
# txt2 = csvPath + "planesweater1_labels.csv"
# labels = pd.read_csv(txt2,sep=',',header=None).values
# raw = np.hstack((ears,labels))

# remove nan entries
tr = raw[~np.isnan(raw).any(axis=1)]

# VALIDATE ON EXTRACTED DATA 
raw_features = extract_features_labels_raw(tr)
extracted_features = extract_features_labels_true(raw_features,True)
X,y = X_y(extracted_features)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1)


# RANDOM SEARCH
C = uniform(loc=0.001,scale=1000)
gamma = uniform(loc = 0.001, scale = 10)
hyperparameters = dict(C=C,gamma=gamma,kernel=['rbf','poly','linear'])
clf = RandomizedSearchCV(svm.SVC(),hyperparameters,random_state=1,n_iter=100,cv=5,verbose=1)
best_model = clf.fit(X_train,y_train)

C = best_model.best_estimator_.get_params()['C']
gamma = best_model.best_estimator_.get_params()['gamma']
#model = svm.SVC(C=C,gamma=gamma,kernel='rbf')

#print("training model")
#model.fit(X_train,y_train)
y_predict = score(clf,X_test,y_test)

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()

# print("cross validating")
# cross_val(X,y)

# VALIDATE ON RAW DATA
# raw_features = extract_features_labels_raw(tr)
# X_raw,y_raw = X_y(raw_features)
# extracted = extract_features_labels_true(raw_features,True)

# X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_raw,y_raw,test_size=0.20,random_state=49)

# X_train,y_train = X_y(extract_features_labels_true(np.hstack((X_train_raw,y_train_raw.reshape((-1,1))))))
# model = svm.SVC(gamma='scale',kernel='rbf')
# model.fit(X_train,y_train)

# y_predict = score(model,X_test_raw,y_test_raw)
# pd.Series(y_predict.flatten()/2).plot()
# pd.Series(y_test_raw.flatten()).plot()
# plt.show()

# # extracted_features = extract_features_labels_true(raw_features,False)
# # X,y = X_y(extracted_features)
# # X,y = X_y(under_sample_balance(extracted_data))
# # print("cross validating")
# # cross_val(X,y)

# # model = svm.SVC(gamma='scale')
# # model.fit(X,y)
# # extracted_data_raw = extract_features_labels_raw(test,True)
# # X,y = X_y(extracted_data_raw)
# # y_predict = score(model,X,y)

# # pd.Series(y_predict.flatten()).plot()
# # pd.Series(y.flatten()).plot()
# # plt.show()

