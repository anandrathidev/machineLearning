
# coding: utf-8

# In[1]:


import numpy as np
#from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
from sklearn import feature_selection
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis, KernelPCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.cross_decomposition import PLSCanonical

import sys
orig_stdo = sys.stdout 

from sklearn import decomposition
import matplotlib.pyplot as plt

import lightgbm as lgb

np.random.seed(42)

import pandas as pd

filepath = "/home/ubuntu/testk/"
filepath = "/home/he159490/DS/Kaggle/SantanderValue//"

data_file = filepath + "train.csv"
test_file = filepath + "test.csv"

data = None
Y = None
xdata = pd.read_csv(data_file)
list(xdata.columns)
Y = xdata["target"]
data = xdata.drop(columns=['ID', 'target'])

print(list(data.columns)[0:10])
#print(list(Y.columns))
testdata = None
Ytest = None
testID = None
try:
  testdata = pd.read_csv(test_file)
  print("Test Data Loaded...")
  #list(testdata.columns)
  testID = testdata["ID"]

except Exception as e:
  print(e)

testdata = testdata.drop(columns=['ID'])


# In[]:
YBC = np.log1p(Y)
plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(YBC))
plt.show()



#testdata=testdata.drop(colsToRemoveTest, axis=1, inplace=False)
#data=data.drop(colsToRemoveTest, axis=1, inplace=False)

# In[6]:

train_without_duplicates = data.T.drop_duplicates().T
columns_not_to_be_dropped = train_without_duplicates.columns
columns_to_be_dropped = [col for col in testdata.columns if col not in columns_not_to_be_dropped]
print(columns_to_be_dropped )
print(len(columns_to_be_dropped ))
#testdata = testdata.drop(columns_to_be_dropped, 1)
#data = data.drop(columns_to_be_dropped, 1)

# In[87]:

print("Feature scaling...")
fulldata = data.append(testdata)
rscaler = preprocessing.StandardScaler()
nrowscaler = preprocessing.Normalizer()


dataScaledLog = data
testdataScaledLog = testdata
fulldataScaledLog = fulldata

print("Done...")

# In[8]:

print(fulldataScaledLog[0:10])

# In[8]:
def StratifiedSample(data, MY,  test_size, random_state):
  ylen = MY.shape[0]
  print("Ylen = {}".format(ylen))
  nbins = np.unique(MY).shape[0]
  print("bins = {}".format(nbins))
  bins = np.linspace(0, ylen, nbins)

  y_binned = np.digitize(MY, bins)
  MY=np.log1p(MY)
  X_data, X_test, Y_data, y_test = train_test_split(data, MY,
                                                    stratify=y_binned,
                                                    test_size=test_size, random_state=random_state)
  print("X_data = {}".format(X_data.shape))
  print("X_test = {}".format(X_test.shape))
  return  X_data, X_test, Y_data, y_test


# In[8]:


#dataScaledLog=dataScaledLog.drop(["is_test"],1)
#testdataScaledLog=testdataScaledLog.drop(["is_test"],1)
dataScaledLog = np.log1p(data)
testdataScaledLog = np.log1p(testdata)
fulldataScaledLog = np.log1p(fulldata)

dataScaledLog.columns = data.columns
testdataScaledLog.columns = testdata.columns
fulldataScaledLog.columns = fulldata.columns

print(dataScaledLog[0:1])
print(testdataScaledLog[0:1])
print(fulldataScaledLog[0:1])


# In[ ]:


# In[13]:


dataScaledLog.shape
print("start ...1 ")
sys.stdout = open('out.txt', 'w')
print("start ...2")
print(dataScaledLog[0:10])
print(testdataScaledLog[0:10])
print(fulldataScaledLog[0:10])


# In[10]:


# In[ ]:
import sys
import numpy as np
from keras import backend as K
import importlib
import os

def set_keras_backend(backend):
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        os.environ['OMP_NUM_THREADS']="11"
        importlib.reload(K)
        assert K.backend() == backend
set_keras_backend(backend="theano")
        
from keras.models import Sequential
from keras.layers import LSTM, Dense, BatchNormalization
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU


# In[ ]:



data_dim = dataScaledLog.shape[1]
timesteps = 1
num_classes = 1
batch_size = 637 # dataScaledLog.shape[0]
batch_size = dataScaledLog.shape[0]
# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(batch_size, activation=None, return_sequences=True, stateful=True, unroll=False,
               batch_input_shape=(batch_size, timesteps, data_dim)))
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(LSTM(data_dim, activation=None, return_sequences=False, unroll=False))  # returns a sequence of vectors of dimension 1
model.add(LeakyReLU(alpha=0.1))
model.add(BatchNormalization())
model.add(Dense(1, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.compile(loss='mse',
              optimizer=optimizers.RMSprop(lr=0.001), #optimizers.Nadam(lr=0.009)
              metrics=['accuracy'])

# Generate dummy training data
X_train = dataScaledLog.values.reshape(dataScaledLog.shape[0], 1, dataScaledLog.shape[1])
y_train = YBC

# Generate dummy validation data
x_val = X_train
y_val = YBC

import gc 
gc.collect()

model.fit(X_train, y_train, batch_size=batch_size, epochs=11665, shuffle=False, validation_data=(x_val, y_val)) 
#model.fit(X_train, y_train,  batch_size=batch_size, epochs=3665, shuffle=False, validation_data=(x_val, y_val))


score = model.evaluate(x_val, y_val, batch_size=7)
print(score)


# In[ ]:


batch_size = 637
model.fit(X_train, y_train, batch_size=batch_size, epochs=13665, shuffle=False, validation_data=(x_val, y_val)) 


# In[ ]:


sys.stdout = orig_stdo

