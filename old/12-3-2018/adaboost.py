# Try adaptive boost to achieve better preformance
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, make_scorer

from data_funcs import *

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


# # STEVENS DATA --------------

print("loading in ears")
txt1 = csvPath + "1_EAR.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values[0:18200]


print("loading in labels")
txt2 = csvPath + "1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:18200]
raw = np.hstack((ears,labels))

# =============================================================================
# # MY DATA
#
# print("loading in ears")
# txt1 = csvPath + "planesweater_ears.csv"
# ears = pd.read_csv(txt1,sep=',',header=None).values
#
# print("loading in labels")
# txt2 = csvPath + "planesweater_labels0-70000.csv"
# labels = pd.read_csv(txt2,sep=',',header=None).values
# raw = np.hstack((ears,labels))
# =============================================================================

# =============================================================================
# mask = np.zeros(raw.size)
# ear = raw.shape[1] - 2
# for i in range(len(raw)):
#     if(np.isnan(ear)):
#         mask[i-F_VECTOR_LENGTH:i+F_VECTOR_LENGTH+1] = 1
# =============================================================================


# remove nan entries
# tr = raw[~np.isnan(raw).any(axis=1)]
tr = np.nan_to_num(raw)

# VALIDATE ON EXTRACTED DATA
raw_features = extract_features_labels_raw(tr)

X,y = X_y(raw_features)
# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)
# X_train,y_train = X_y(extract_features_labels_true(np.hstack((X_train,y_train.reshape(-1,1)))))
# =============================================================================

with open(savedModelPath + "final_gb.pkl", 'rb') as pickle_file:
    model = pickle.load(pickle_file)[0]

y_predict = score(model,X,y)

pd.Series(y.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()

#X_train,y_train = over_sample_balance(X_train,y_train)

# svm = svm.SVC(C=1000,gamma=0.01,kernel='rbf')
# =============================================================================
# tree = DecisionTreeClassifier(max_depth=5,class_weight='balanced')
# print("classifier")
#
# ac = AdaBoostClassifier(tree,random_state=1,n_estimators=300)
#
# =============================================================================
# scorers = {'accuracy_score': make_scorer(accuracy_score),
		   # 'precision_score': make_scorer(precision_score),
			# 'recall_score': make_scorer(recall_score)}

# param_dist_gb = {
					# 'learning_rate':uniform(0.001,0.099),
					# 'n_estimators':randint(100,2000),
# }
# n_iter_search = 20

# ac = RandomizedSearchCV(ac, param_distributions=param_dist_gb,
								   # refit='precision_score', n_iter=n_iter_search,
									# cv=5, verbose=10, scoring=scorers, n_jobs=-1)

# =============================================================================
# print("fitting")
# ac.fit(X_train,y_train)
# print("fitted, predicting:")
# y_predict = score(ac,X_test,y_test)
#
# pd.Series(y_test.flatten()).plot()
# pd.Series(y_predict.flatten()/2).plot()
# plt.show()
# =============================================================================

