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
print(list(Y.columns))

#=============================================================================
# start Feature selection
#=============================================================================


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing

X_normalized = preprocessing.normalize(data, norm='l2')

from  sklearn.feature_selection import mutual_info_regression

FS = mutual_info_regression(data,Y)

model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(data)
X_new.shape

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

trainrms = sqrt(mean_squared_error(ypredDF['Y'], ypredDF['YHat']))
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
ygp = gp.fit(ypredDF, Y)

ypredDFXGS = pd.DataFrame( [y_xgbpred, y_pred] , columns=["YHatXGS", "YHatRF"])

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(ypredDF, Y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_RBFpred, sigma = gp.predict(ypredDF, return_std=True)


from sklearn.metrics import mean_squared_error
from math import sqrt

#y_pred = np.around(y_pred, decimals=2)
ypredDFFinal = pd.DataFrame(y_RBFpred , columns=["YHat"])
ypredDFFinal["Y"] = Y

finalyrbf_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander//RFrbf_y.csv"

ypredDFFinal.to_csv(finalyrbf_file)


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
except Exception as e:
  print(e)

ytest_pred = RFregr.predict(testdata)
ytestpredDF = pd.DataFrame(testID, columns=["ID"])
ytestpredDF["target"] = ytest_pred
test_result_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/RFTestResult.csv"
ytestpredDF.to_csv(test_result_file, index=False)

ytestpredDF.shape

#=============================================================================
# start TEST RBF
#=============================================================================

ytest_predRFRBF = gp.predict(pd.DataFrame(ytest_pred))
ytest_predRFRBFDF = pd.DataFrame(testID, columns=["ID"])
ytest_predRFRBFDF["target"] = ytest_predRFRBF


test_result_file = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/RFRBFTestResult.csv"
ytest_predRFRBFDF.to_csv(test_result_file, index=False)

ytestpredDF.shape




