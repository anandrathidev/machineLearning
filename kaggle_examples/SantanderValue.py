# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:28:12 2018

@author: he159490
"""

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

filepath = "F:/DataScience/Kagggle/SantanderValue/"
data_file = filepath + "train.csv"
y_RF_file = filepath + "/tmp/RF_y.csv"
finalyrbf_file = filepath + "/tmp///RFrbf_y.csv"
yXGS_file = filepath + "/tmp/XGS_y.csv"

test_file = filepath + "test.csv"
test_result_file1 = filepath + "/tmp/ypredDFFinalDetails.csv"
test_result_file2 = filepath + "/tmp//FinalTestResult.csv"

test_result_file21 = filepath + "/tmp//FinalTestResultarget.csv"
test_result_file22 = filepath + "/tmp//FinalTestResultRetarget.csv"

PrepareData=True
if PrepareData:
    data = None
    Y = None
    try:
      data = pd.read_csv(data_file)
      list(data.columns)
      Y = data["target"]
      data = data.drop(columns=['ID', 'target'])
    except Exception as e:
      print(e)
    
    Y = np.log1p(Y)
    print(list(data.columns)[0:10])
    #print(list(Y.columns))

#=============================================================================
# start Feature selection
#=============================================================================

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA

from sklearn import decomposition
from sklearn import datasets


rscaler = RobustScaler()
rscaler.fit(data)
data = rscaler.transform(data)

pca = decomposition.PCA(0.94)
pca.fit(data)
data = pca.transform(data)
data.shape

#X_normalized = preprocessing.normalize(data, norm='l2')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

lreg = LinearRegression()
lreg.fit(data,Y)
y_Lpred = lreg.predict(data)

trainrms = sqrt(mean_squared_error(Y, y_Lpred))
print("LREG : trainrms {}".format(trainrms ) )


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


knnreg = KNeighborsRegressor(n_neighbors=2)
knnreg.fit(data,Y)
y_KNNpred = knnreg.predict(data)
trainrms = sqrt(mean_squared_error(Y, y_KNNpred))
print("KNN PCA : trainrms {}".format(trainrms ) )

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
print("RFPCA : trainrms {}".format(trainrms ) )
#=============================================================================
# end RF
#=============================================================================
#=============================================================================
# start XGS
#=============================================================================

import xgboost
xgb = xgboost.XGBRegressor(n_estimators=101, learning_rate=0.59, gamma=5, subsample=0.81,
                           colsample_bytree=1, max_depth=15)

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
print("XGSPCA : trainrms {}".format(trainrms ) )

ypredDFXGS.to_csv(yXGS_file)

#=============================================================================
# ADD RBF
#=============================================================================


# Instanciate a Gaussian Process model
kernel = C(7.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=None, alpha=0.0001,  n_restarts_optimizer=33, normalize_y=True)
ygp = gp.fit(data, Y)


# Make the prediction on the meshed x-axis (ask for MSE as well)
y_RBFpred = gp.predict(data)


from sklearn.metrics import mean_squared_error
from math import sqrt

#y_pred = np.around(y_pred, decimals=2)

ypredDFFinal =  pd.DataFrame(dict(RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred , Y=Y))

trainrms = sqrt(mean_squared_error(Y, y_RBFpred))
print("RBFPCA : trainrms {}".format(trainrms ) )


ypredDFFinal.to_csv(finalyrbf_file)

ypredDFFinal =  pd.DataFrame(dict(RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred ))
ypredDFFinal.shape


#=============================================================================
# end RBF
#=============================================================================


xgbFinal = xgboost.XGBRegressor(n_estimators=101, learning_rate=0.19, gamma=130, subsample=0.71,
                           colsample_bytree=1, max_depth=15)
xgbFinal.fit(ypredDFFinal,Y)

YxgbFinal = xgbFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(Y, YxgbFinal))
print("RBFFinal : trainrms {}".format(trainrms ) )


#=============================================================================
# start TEST
#=============================================================================

if PrepareData:
    testdata = None
    Ytest = None
    testID = None
    try:
      testdata = pd.read_csv(test_file)
      print("Test Data Loaded...")
      #list(testdata.columns)
      testID = testdata["ID"]
      testdata = testdata.drop(columns=['ID'])
      testdata = rscaler.transform(testdata)
      print("Test Data scaled...")
      testdata = pca.transform(testdata)
      print("Test Data pca...")
    except Exception as e:
      print(e)

print("Predict  RF...")
ytest_RF = RFregr.predict(testdata)
print("Predict  xgb...")
ytest_XGS = xgb.predict(testdata)
print("Predict  rbf...")
ytest_RBF = gp.predict(testdata)
print("Predict  KNN...")
ytest_KNNpred= knnreg.predict(testdata)

ypredDFFinalDetails = pd.DataFrame(dict(RBF = ytest_RBF, XGS= ytest_XGS, RF = ytest_RF, KNN=ytest_KNNpred, Target= (1*ytest_RBF + 10*ytest_XGS + ytest_RF + ytest_KNNpred)/13  ))
ypredDFFinalDetails.to_csv(test_result_file1, index=False)
ypredDFFinalDetails = pd.DataFrame(dict(RBF = ytest_RBF, XGS= ytest_XGS, RF = ytest_RF, KNN = ytest_KNNpred))
ypredDFFinalDetails.shape

print("Predict  xgs final ...")
ygpTestFinal = xgbFinal.predict(ypredDFFinalDetails)

print("Write test data ...")
ypredDFFinalDetails["ID"] = testID
ypredDFFinalDetails["YGP"] = ygpTestFinal

ypredDFFinalDetails["ReTarget"] = np.rint(np.exp( (ytest_RF + ytest_XGS + 3*ygpTestFinal)/5 +1))
ypredDFFinalDetails["Target"] =  np.rint(np.exp((3*ytest_RBF + 10*ytest_XGS + ytest_RF)/14  +1))

ypredDFFinalDetails.to_csv(test_result_file2, index=False)

sumbission1 = ypredDFFinalDetails[["ID"]]
sumbission1["target"] = ypredDFFinalDetails["Target"] 
sumbission1.to_csv(test_result_file21, index=False)

sumbission2 = ypredDFFinalDetails[["ID"]]
sumbission2["target"] = ypredDFFinalDetails["ReTarget"] 
sumbission2.to_csv(test_result_file22, index=False)


sumbission2 = ypredDFFinalDetails[["ID"]]
sumbission2["target"] = np.rint(np.exp( ytest_XGS -1))
sumbission2.to_csv(filepath + "/tmp/XGSTarget.csv" , index=False)

ypredDFFinalDetails.shape

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