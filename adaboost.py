# Try adaptive boost to achieve better preformance
import os
import numpy as np
import pandas as pd
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

# MY DATA ------------------

print("loading in data")
txt1 = csvPath + "planesweater1_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

txt2 = csvPath + "planesweater1_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw1 = np.hstack((ears,labels))

txt1 = csvPath + "planesweater3_ears.csv"
ears = pd.read_csv(txt1,sep=',',header=None).values[0:4500]
gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
ears = np.hstack((ears,gradient))

txt2 = csvPath + "planesweater3_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values[0:4500]
raw2 = np.hstack((ears,labels))

raw = np.vstack((raw1,raw2))

# # STEVENS DATA --------------

# print("loading in ears")
# txt1 = csvPath + "1_EAR.csv"
# ears = pd.read_csv(txt1,sep=',',header=None).values[0:18200]
# gradient = np.gradient(ears[:,EAR]).reshape((-1,1))
# ears = np.hstack((ears,gradient))

# print("loading in labels")
# txt2 = csvPath + "1_labels.csv"
# labels = pd.read_csv(txt2,sep=',',header=None).values[0:18200]
# raw = np.hstack((ears,labels))

# remove nan entries
tr = raw[~np.isnan(raw).any(axis=1)]

# VALIDATE ON EXTRACTED DATA 
raw_features = extract_features_labels_raw(tr)
extracted_features = extract_features_labels_true(raw_features)
X,y = X_y(extracted_features)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=1)
#X_train,y_train = over_sample_balance(X_train,y_train)

svm = svm.SVC(C=1000,gamma=0.01,kernel='rbf')
tree = DecisionTreeClassifier(max_features="auto", max_depth=2)
print("classifier")

ac = AdaBoostClassifier(tree,random_state=1)
							  
scorers = {'accuracy_score': make_scorer(accuracy_score),
		   'precision_score': make_scorer(precision_score),
			'recall_score': make_scorer(recall_score)}
			
param_dist_gb = {
					'learning_rate':uniform(0.001,0.099), 
					'n_estimators':randint(100,2000),
}
n_iter_search = 20

ac = RandomizedSearchCV(ac, param_distributions=param_dist_gb, 
								   refit='precision_score', n_iter=n_iter_search, 
									cv=5, verbose=10, scoring=scorers, n_jobs=-1)
							  
# ac = AdaBoostClassifier(tree,
							  # algorithm="SAMME.R",
							  # n_estimators=100,
							  # random_state=1)
# ac = AdaBoostClassifier(ac,
							  # algorithm="SAMME.R",
							  # n_estimators=100,
							  # random_state=3)
							  
# ac = AdaBoostClassifier(ac,
							  # algorithm="SAMME.R",
							  # n_estimators=100,
							  # random_state=1)
							
ac.fit(X_train,y_train)
print("fitted, predicting:")
y_predict = score(ac,X_test,y_test)

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()

