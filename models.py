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
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import LSTM, Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
import functools
from keras import backend as K
import tensorflow as tf

from data_funcs import *

# converts a tensorflow metric into a keras metric
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
    
# cv
def cross_val(model,X,y):
    print("cross validating: " + str(type(model)))
    model = clone(model)
    pp = pprint.PrettyPrinter(indent=4)

    #Metric scores
    scoring = {'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'}

    score_lr = cross_validate(model, X, y, scoring=scoring, cv=3)
    
    print("\tmean fit time: " + str(np.mean(score_lr['fit_time'])))
    print("\tmean recall: " + str(np.mean(score_lr['test_rec_macro'])))
    print("\tmean precision: " + str(np.mean(score_lr['test_prec_macro'])))


def score(model,X,y,thresh=False):
    y_predict = model.predict(X)
    if(thresh):
        y_predict = y_predict > 0
        
    print("\ttotal labels predicted: " + str(np.sum(y_predict)))
    recall = recall_score(y,y_predict)
    precision = precision_score(y,y_predict)
    accuracy = accuracy_score(y,y_predict)

    print("\trecall: " + str(recall))
    print("\tPrecision: " + str(precision))
    print("\tAccuracy: " + str(accuracy))

    return y_predict
 
def prec(y,y_predict):
    pdb.set_trace()
    return precision_score(y,y_predict)
        
def model1():
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = Sequential()
    model.add(Dense(100,input_dim=136,kernel_initializer='normal',activation='relu'))
    model.add(Dense(50,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse',metrics=[precision,recall])
    return model

# MY DATA ------------------

# Training data
# print("loading in landmarks")
txt1 = csvPath + "paulsim2_landmarks.csv"
features = pd.read_csv(txt1,sep=',',header=None).values[0:13000]

# print("loading in labels")
txt2 = csvPath + "paulsim2__openclosed_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:13000]
raw = np.hstack((features,labels))

X,y = raw[:,range(raw.shape[1]-1)].astype(int), raw[:,raw.shape[1]-1].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

# ================== SVM 
# model = svm.SVC(gamma='scale',C=2)
# print("fitting")
# model.fit(X_train,y_train)
# print("finished svm")
# y_predict = score(model,X_test,y_test)
# cross_val(model,X,y)
# # svm, 93 prec
# ==================

# ================== MLP
# model = model1()
# # Standardize Data
# X = scipy.stats.zscore(X)
# model.fit(X_train,y_train,epochs=100,batch_size=5,validation_data=(X_test,y_test))
# y_predict = score(model,X_test,y_test,True)
# =================

# ================= Adaboost
forest = RandomForestClassifier(n_estimators=5,n_jobs=-1)
model = AdaBoostClassifier(forest,n_estimators=forest.n_estimators)
params = {
					'learning_rate':uniform(0.001,0.099),
					'n_estimators':randint(5,500),
}
#model = RandomizedSearchCV(model,params,scoring='precision',verbose=10,n_jobs=-1,n_iter=1)
cross_val(model,X,y)
print("fitting")
#X_train,y_train = under_sample_balance(X_train,y_train)
model.fit(X_train,y_train)
print("finished boost")
y_predict = score(model,X_test,y_test)

# # forest 5, 95 prec
# # tree, 90 prec
# ================


# ================ Random forest
# model = RandomForestClassifier(n_estimators=5,n_jobs=-1)
# print("fitting")
# model.fit(X_train,y_train)
# print("finished forest")
# y_predict = score(model,X_test,y_test)

# cross_val(model,X,y)
# ================


# pd.Series(y_test.flatten()).plot()
# pd.Series(y_predict.flatten()/2).plot()
# plt.show()





