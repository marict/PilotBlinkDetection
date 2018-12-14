# try using an svm for the series analysis
import numpy as np
import math
import scipy.stats
import pprint
import os
import datetime
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import uniform
from sklearn import svm
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from numpy import nan

from data_funcs import *

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

# # STEVEN'S DATA ----------------

# print("loading in ears")
txt1 = csvPath + "1_EAR.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values[0:18200]

print("loading in labels")
txt2 = csvPath + "1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:18200]
raw = np.hstack((ears,labels))

# =============================================================================
# # MY DATA ------------------
#
# # print("loading in ears")
# txt1 = csvPath + "planesweater_ears.csv"
# ears = pd.read_csv(txt1,sep=',',header=None).values
#
# # print("loading in labels")
# txt2 = csvPath + "planesweater_labels.csv"
# labels = pd.read_csv(txt2,sep=',',header=None).values
# raw = np.hstack((ears,labels))
# =============================================================================

# remove nan entries
# tr = raw[~np.isnan(raw).any(axis=1)]
tr = np.nan_to_num(raw)

# VALIDATE ON EXTRACTED DATA
raw_features = extract_features_labels_raw(tr)


X,y = X_y(raw_features)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
X_train,y_train = X_y(extract_features_labels_true(np.hstack((X_train,y_train.reshape(-1,1)))))

print("model")
with open(savedModelPath + "final_linear.pkl", 'rb') as pickle_file:
    model = pickle.load(pickle_file)[0]

y_predict = score(model,X_test,y_test)

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()



# with open(savedModelPath + "final_linear.pkl", 'rb') as pickle_file:
    # model = pickle.load(pickle_file)[0]

# y_predict = score(model,X_test,y_test)

# pd.Series(y_test.flatten()).plot()
# pd.Series(y_predict.flatten()/2).plot()
# plt.show()

# RANDOM SEARCH
# C = uniform(loc=0.1,scale=1000)
# gamma = uniform(loc = 0.1, scale = 10)
# hyperparameters = dict(C=C,gamma=gamma,kernel=['rbf','poly','linear'])
# clf = RandomizedSearchCV(svm.SVC(),hyperparameters,random_state=1,n_iter=10,cv=5,verbose=1)
# best_model = clf.fit(X_train,y_train)

# C = best_model.best_estimator_.get_params()['C']
# gamma = best_model.best_estimator_.get_params()['gamma']
# #model = svm.SVC(C=C,gamma=gamma,kernel='rbf')

# #print("training model")
# #model.fit(X_train,y_train)
# y_predict = score(clf,X_test,y_test)

# pd.Series(y_test.flatten()).plot()
# pd.Series(y_predict.flatten()/2).plot()
# plt.show()

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

