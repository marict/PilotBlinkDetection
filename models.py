
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
    pp = pprint.PrettyPrinter(indent=4)

    #Metric scores
    scoring = {'acc': 'accuracy',
        'prec_macro': 'precision_macro',
        'rec_macro': 'recall_macro'}

    score_lr = cross_validate(model, X, y, scoring=scoring, cv=3)
    
    print("\tmean fit time: " + str(np.mean(score_lr['fit_time'])))
    print("\tmean recall: " + str(np.mean(score_lr['test_rec_macro'])))
    print("\tmean precision: " + str(np.mean(score_lr['test_prec_macro'])))



 
def prec(y,y_predict):
    pdb.set_trace()
    return precision_score(y,y_predict)
        
def model1(X):
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    model = Sequential()
    model.add(Dense(50,input_dim=X.shape[1],activation='relu'))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=[precision,recall])
    #model.compile(optimizer='adam', loss='mse')
    return model

# MY DATA ------------------

# Training data
# print("loading in landmarks")
txt1 = csvPath + "paulsim2_landmarks.csv"
features = pd.read_csv(txt1,sep=',',header=None).values

# print("loading in labels")
txt2 = csvPath + "paulsim2_openclosed_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw = np.hstack((features,labels))
X,y = raw[:,range(raw.shape[1]-1)].astype(int), raw[:,raw.shape[1]-1].astype(int)
X = augment_landmarks_window(X)
X = scipy.stats.zscore(X,axis=0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

# ================== SVM 
model = svm.SVC(gamma='scale',C=2)
print("fitting")
model.fit(X_train,y_train)
print("finished svm")
y_predict = score(model,X_test,y_test)
# cross_val(model,X,y)
# # svm, 93 prec
# ==================

# ================== MLP
# model = model1(X)
# # Standardize Data
# X_train,y_train = under_sample_balance(X_train,y_train)
# model.fit(X_train,y_train,epochs=10,batch_size=5,validation_data=(X_test,y_test))
# y_predict = score(model,X_test,y_test)
# =================

# ================= Adaboost
# forest = RandomForestClassifier(n_estimators=10,n_jobs=-1)
# model = AdaBoostClassifier(forest,n_estimators=forest.n_estimators)
# model.fit(X_train,y_train)
# print("finished boost")
# y_predict = score(model,X_test,y_test)
#cross_val(model,X,y)
# ================


# ================ Random forest
# model = RandomForestClassifier(n_estimators=5,n_jobs=-1)
# print("fitting")
# model.fit(X_train,y_train)
# print("finished forest")
# y_predict = score(model,X_test,y_test)

# cross_val(model,X,y)
# ================

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()





