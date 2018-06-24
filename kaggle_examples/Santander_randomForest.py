# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


np.random.seed(1)

import pandas as pd

data_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/train.csv/train.csv"
y_RF_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander//RF_y.csv"

data = None
Y = None
try:
  data = pd.read_csv(data_file)
  list(data.columns)
  Y = data["target"]
  data = data.drop(columns=['ID', 'target'])
except Exception as e:
  print(e)


print(list(data.columns))
#print(list(Y.columns))

#=============================================================================
# start Feature selection
#=============================================================================


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from  sklearn.feature_selection import mutual_info_regression
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA

from sklearn import decomposition
from sklearn import datasets

pca = decomposition.PCA(n_components=3200)
pca.fit(data)
X = pca.transform(data)
X.shape

#X_normalized = preprocessing.normalize(data, norm='l2')

data=X

#=============================================================================
# end Feature selection
#=============================================================================


# Instanciate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
RFregr = RandomForestRegressor(random_state=0)

# Fit to data using Maximum Likelihood Estimation of the parameters
RFregr.fit(data, Y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred = RFregr.predict(data)


from sklearn.metrics import mean_squared_error
from math import sqrt

#y_pred = np.around(y_pred, decimals=2)
ypredDF = pd.DataFrame(y_pred , columns=["YHat"])
#ypredDF["Y"] = Y

ypredDF.shape

trainrms = sqrt(mean_squared_error(Y, ypredDF['YHat']))
print("RF : trainrms {}".format(trainrms ) )
#=============================================================================
# end RF
#=============================================================================
#=============================================================================
# start XGS
#=============================================================================

import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)

# Fit to data using Maximum Likelihood Estimation of the parameters
xgb.fit(data, Y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_xgbpred = xgb.predict(data)


from sklearn.metrics import mean_squared_error
from math import sqrt

#y_pred = np.around(y_pred, decimals=2)
ypredDFXGS = pd.DataFrame(y_xgbpred, columns=["YHatXGS"])
ypredDFXGS["Y"] = Y

trainrms = sqrt(mean_squared_error(Y, y_xgbpred))
print("XGS : trainrms {}".format(trainrms ) )

yXGS_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander//XGS_y.csv"
ypredDFXGS.to_csv(yXGS_file)

#=============================================================================
# ADD RBF
#=============================================================================


# Instanciate a Gaussian Process model
kernel = C(5.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
ygp = gp.fit(data, Y)


# Make the prediction on the meshed x-axis (ask for MSE as well)
y_RBFpred = gp.predict(data)


from sklearn.metrics import mean_squared_error
from math import sqrt

#y_pred = np.around(y_pred, decimals=2)
ypredDFFinal = pd.DataFrame( {y_RBFpred, y_xgbpred, y_pred , Y}  , columns=["RBF", "xgb", "RF", "Y"])

ypredDFFinal =  pd.DataFrame(dict(RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred , Y=Y))

trainrms = sqrt(mean_squared_error(Y, y_RBFpred))
print("RBFPCA : trainrms {}".format(trainrms ) )

finalyrbf_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander//RFrbf_y.csv"

ypredDFFinal.to_csv(finalyrbf_file)

ypredDFFinal =  pd.DataFrame(dict(RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred ))


#=============================================================================
# end RBF
#=============================================================================


#=============================================================================
# start TEST
#=============================================================================


test_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/test.csv"
testdata = None
Ytest = None
testID = None
try:
  testdata = pd.read_csv(test_file)
  list(testdata.columns)
  testID = testdata["ID"]
  testdata = testdata.drop(columns=['ID'])
  testdata = pca.transform(testdata)
except Exception as e:
  print(e)

ytest_RF = RFregr.predict(testdata)
ytest_XGS = xgb.predict(testdata)
ytest_RBF = gp.predict(testdata)

ypredDFFinalDetails = pd.DataFrame(dict(RBF = ytest_RBF, XGS= ytest_XGS, RF = ytest_RF))
test_result_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/ypredDFFinalDetails.csv"
ypredDFFinalDetails.to_csv(test_result_file, index=False)

gpFinal = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gpFinal.fit(ypredDFFinal, Y)
ygpTestFinal = gpFinal.predict(ypredDFFinalDetails)

ytestpredDF = pd.DataFrame(testID, columns=["ID"])
ytestpredDF["target"] = (ytest_RF + ytest_XGS + 3*ygpTestFinal)/5
test_result_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/FinalTestResult.csv"
ytestpredDF.to_csv(test_result_file, index=False)

ytestpredDF.shape

#=============================================================================
# start TEST RBF
#=============================================================================
"""
ytest_predRFRBF = gp.predict(pd.DataFrame(ytest_pred))
ytest_predRFRBFDF = pd.DataFrame(testID, columns=["ID"])
ytest_predRFRBFDF["target"] = ytest_predRFRBF


test_result_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/RFRBFTestResult.csv"
ytest_predRFRBFDF.to_csv(test_result_file, index=False)

ytestpredDF.shape

"""


