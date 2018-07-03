# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 22:18:24 2018

@author: anandrathi
"""

import numpy as np
from scipy.stats import boxcox

from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import  MaxAbsScaler
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(42)

import pandas as pd

filepath = "F:/DataScience/Kagggle/SantanderValue/"
filepath = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/"

Y = None
xdata = pd.read_csv(filepath + "train.csv")
Y = xdata["target"]
Y = Y
YBC, ylambda = boxcox(Y)
def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))

Yt = invboxcox(YBC,ylambda)
Ydf = pd.DataFrame(dict(Y=Y,Yt=Yt))

plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(Ydf["Y"]))
plt.show()
Y=YBC
plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(Y))
plt.show()


print("Train & test split...")
#X_data, X_test, Y_data, y_test = train_test_split(dataSVD, Y, test_size=0.20, random_state=42)
#X_data, X_test, Y_data, y_test = train_test_split(dataPCA , Y, test_size=0.25, random_state=42)
X_data, X_test, Y_data, y_test = train_test_split(data, Y, test_size=0.25, random_state=42)

#=============================================================================
# end Feature selection
#=============================================================================
import time
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared
print("KernelRidge ...{}".format("" ) )

# Fit KernelRidge with parameter selection based on 5-fold cross validation
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],
              "kernel": [ExpSineSquared(l, p)
                         for l in np.logspace(-2, 2, 10)
                         for p in np.logspace(0, 2, 10)]}
kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
stime = time.time()
kr.fit(X_data, Y_data)
print("Time for KernelRidge fitting: %.3f" % (time.time() - stime))
y_kr = kr.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_kr ))
print("KernelRidge PCA : trainrms {}".format(trainrms ) )
print("Time for KernelRidge prediction: %.3f" % (time.time() - stime))
plt.figure(figsize=(8,8))
plt.scatter( y_test, y_KNNpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('RF', fontsize=12)
plt.show()


plt.figure(figsize=(8,8))
plt.scatter( y_test, y_xgbpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('XGB', fontsize=12)
plt.show()

#=============================================================================
# end RBF
#=============================================================================

#import SantanderValueTestReverse



print("xgbFinal ...{}".format("" ) )
ypredDFFinal =  pd.DataFrame(dict( KNN=y_KNNpred, XGS=y_xgbpred, RF = y_pred) )
#ypredDFFinal_Test =  pd.DataFrame(dict( KNNTest=y_KNNpredTest_Test, XGSTest= y_xgbpredTest_Test, RFTest = y_predTest_Test  ))

#ypredDFFinal=ypredDFFinal.join(ypredDFFinal_Test)
print(ypredDFFinal.shape)
xgbFinal = xgboost.XGBRegressor(n_estimators=201, learning_rate=0.19, gamma=10, subsample=0.71,
                           colsample_bytree=1, max_depth=15)
xgbFinal.fit(ypredDFFinal,y_test)
print(X_test.shape)
print(X_data.shape)
YxgbFinal = xgbFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(y_test, YxgbFinal))
print("XGSFinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(y_test, YxgbFinal, s=100,  marker="s", label='YxgbFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YxgbFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

RFregrFinal = RandomForestRegressor(n_estimators=111, random_state=0, oob_score=True)
# Fit to data using Maximum Likelihood Estimation of the parameters
RFregrFinal.fit(ypredDFFinal, y_test)
YRFFinal = RFregrFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(y_test, YRFFinal))
print("YRFFinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(y_test, YRFFinal, s=100,  marker="s", label='YRFFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YRFFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

lregFinal = LinearRegression()
lregFinal.fit(ypredDFFinal,y_test)
YLFinal  = lregFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(y_test, YLFinal ))
print("YRFFinal : trainrms {}".format(trainrms ) )

plt.figure(figsize=(11,11))
plt.scatter(y_test, YLFinal, s=100,  marker="s", label='LinFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YLFinal', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

#radregF  = RadiusNeighborsRegressor(weights='distance', radius=1.3)
#radregF.fit(ypredDFFinal,Y)
#y_radFinal = radregF.predict(ypredDFFinal)

#=============================================================================
# start TEST
#=============================================================================


print("Predict  RF...")
ytest_RF= invboxcox(RFregr.predict(testdataSVD),ylambda)
print("Predict  xgb...")
ytest_XGS = invboxcox(xgb.predict(testdataSVD),ylambda)
print("Predict  KNN...")
ytest_KNNpred= invboxcox(knnreg.predict(testdataSVD),ylambda)

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_XGS)), ytest_XGS, s=100,  marker="s", label='RF-XGS')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_XGS', fontsize=12)
plt.title("ytest_XGS Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_XGS)), ytest_KNNpred, s=100,  marker="s", label='ytest_KNNpred')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_KNNpred', fontsize=12)
plt.title("ytest_KNNpred Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_XGS)), ytest_RF, s=100,  marker="s", label='ytest_RF')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_RF', fontsize=12)
plt.title("ytest_RF Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter(ytest_RF, ytest_XGS, s=100,  marker="s", label='RF-XGS')
plt.xlabel('ytest_RF', fontsize=12)
plt.ylabel('ytest_XGS', fontsize=12)
plt.title("rf-xgs Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(ytest_XGS, ytest_KNNpred, s=100,  marker="s", label='XGS-KNN')
plt.xlabel('XGS', fontsize=12)
plt.ylabel('KNN', fontsize=12)
plt.title("xgs knn Distribution", fontsize=14)
plt.show()

#########################################################
######## Final prediction  ###############
#########################################################


ypredDFFinalDetails = pd.DataFrame(dict( KNN=ytest_KNNpred, XGS= ytest_XGS, RF = ytest_RF  ))
print("ypredDFFinalDetails = {}".format(ypredDFFinalDetails.shape))
print(ypredDFFinalDetails.columns)

print("Predict  xgs final ...")
yxgsTestFinal = invboxcox(xgbFinal.predict(ypredDFFinalDetails), ylambda)
yRFTestFinal = invboxcox(RFregrFinal.predict(ypredDFFinalDetails), ylambda)

#yradTestFinal = radregF.predict(ypredDFFinalDetails)
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(yxgsTestFinal)), yxgsTestFinal, s=100,  marker="s", label='yxgsTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('yxgsTestFinal ', fontsize=12)
plt.title("yxgsTestFinal Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter(range(0,len(yRFTestFinal)), yRFTestFinal, s=100,  marker="s", label='yRFTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('yRFTestFinal ', fontsize=12)
plt.title("yRFTestFinal Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter( yxgsTestFinal , yRFTestFinal, s=100,  marker="s", label='xgs_RFT ' )
plt.xlabel('yxgsTestFinal', fontsize=12)
plt.ylabel('yRFTestFinal ', fontsize=12)
plt.title("yRFTestFinal Distribution", fontsize=14)
plt.show()


ypredDFFinalDetailsavg = pd.DataFrame(dict( XGS_RF_KNN=( ytest_XGS + ytest_RF + ytest_KNNpred)/3  ))
ypredFinalFinal = pd.DataFrame(dict(FXGS=yxgsTestFinal, FRF=yRFTestFinal ))
ypredFinalFinal = ypredFinalFinal.join(ypredDFFinalDetailsavg).join(ypredDFFinalDetails)

print(ypredDFFinal.columns)
print(ypredDFFinalDetails.columns)
print(ypredFinalFinal.columns)

ypredFinalFinal.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)


print("Write test data ...")
ypredDFFinalDetails["ID"] = testID
ypredDFFinalDetails["target"] = (yRFTestFinal+ yxgsTestFinal) /2
ypredDFFinalDetails.to_csv(filepath + "/tmp/FinalSub.csv", index=False)


sumbission1 = ypredDFFinalDetails[["ID"]]
sumbission1["target"] = (yRFTestFinal + ytest_XGS + ytest_RF + ytest_KNNpred)/4
sumbission1.to_csv(test_result_file21, index=False)
sumbission1.to_csv(filepath + "/tmp/FinalSub.csv", index=False)

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