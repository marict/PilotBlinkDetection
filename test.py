from data_funcs import *

# Training data
# print("loading in landmarks")
txt1 = csvPath + "paulsim2_landmarks.csv"
features = pd.read_csv(txt1,sep=',',header=None).values

# print("loading in labels")
txt2 = csvPath + "paulsim2_openclosed_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw = np.hstack((features,labels))
X_train,y_train = raw[:,range(raw.shape[1]-1)].astype(int), raw[:,raw.shape[1]-1].astype(int)
X_train = augment_landmarks_window(X_train)

# Testing data
# print("loading in landmarks")
txt1 = csvPath + "paulsim4_landmarks.csv"
features = pd.read_csv(txt1,sep=',',header=None).values

# print("loading in labels")
txt2 = csvPath + "paulsim4_openclosed_labels.csv"
labels = pd.read_csv(txt2,sep=',',header=None).values
raw = np.hstack((features,labels))
X_test,y_test = raw[:,range(raw.shape[1]-1)].astype(int), raw[:,raw.shape[1]-1].astype(int)
X_test = augment_landmarks_window(X_test)

forest = RandomForestClassifier(n_estimators=5,n_jobs=-1)
model = AdaBoostClassifier(forest,n_estimators=forest.n_estimators)
# forest.n_estimators
print("fitting")
model.fit(X_train,y_train)
print("finished boost")
y_predict = score(model,X_test,y_test)

pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()

# Smooth prediction within a 10 frame window
smooth(y_predict)
pd.Series(y_test.flatten()).plot()
pd.Series(y_predict.flatten()/2).plot()
plt.show()

# Identify apexes of blinks
apexes = get_apexes(y_predict)
pd.Series(y_test.flatten()).plot()
pd.Series(apexes.flatten()/2).plot(color=['g'])
plt.show()
