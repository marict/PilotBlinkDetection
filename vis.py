import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.preprocessing import *

# Data visualization file

basePath = "D:\\blink-detection\\data\\"
blink = np.loadtxt(basePath + "onsim.csv",delimiter=',', skiprows=1)
blink_labels = np.loadtxt(basePath + 'onsim_labels.csv',delimiter=',')

#scaler = sk.StandardScalar()
#scalar.fit(np.atleast_2d(bl#ink.T))
# pd.Series(scalar.transform(np.atleast_2d(blink[1000:1200]).T[:,0]).plot())
# pd.Series(blink_lables[1000:12000]).plot()

ears = blink[:,2]
ears = normalize(ears[:,np.newaxis],axis=0).ravel()
blink_labels = normalize(blink_labels[:,np.newaxis],axis=0).ravel()

pd.Series(ears).plot()
pd.Series(blink_labels).plot()
plt.show()

