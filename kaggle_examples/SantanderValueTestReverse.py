# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:46:31 2018

@author: he159490
"""
# Instanciate a Gaussian Process model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

from sklearn import decomposition
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.metrics import mean_squared_error
from math import sqrt

print("KNN...{}".format("" ) )
#knnregTest = KNeighborsRegressor(n_neighbors=5)
knnregTest = KNeighborsRegressor(n_neighbors=6, weights='distance', metric='minkowski', n_jobs=5)
knnregTest.fit(X_test,y_test)
y_KNNpredTest = knnregTest.predict(X_data)
trainrms = sqrt(mean_squared_error(Y_data, y_KNNpredTest))
print("KNN PCA : testrms {}".format(trainrms ) )


print("RF ...{}".format("" ) )
RFregrTest = RandomForestRegressor(n_estimators=319, random_state=0, min_samples_leaf=3, oob_score=False, n_jobs=6)
# Fit to data using Maximum Likelihood Estimation of the parameters
RFregrTest.fit(X_test,y_test)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_predTest = RFregrTest.predict(X_data)
trainrms = sqrt(mean_squared_error(Y_data, y_predTest))
print("RFPCA : testrms  {}".format(trainrms ) )
#=============================================================================
# end RF
#=============================================================================
#=============================================================================
# start XGS
#=============================================================================

print("XGboost ...{}".format("" ) )
import xgboost
xgbTest = xgboost.XGBRegressor(n_estimators=111, learning_rate=0.59, gamma=5, subsample=0.81,
                           colsample_bytree=1, max_depth=15, n_jobs=6)

# Fit to data using Maximum Likelihood Estimation of the parameters
xgbTest.fit(X_test,y_test)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_xgbpredTest = xgbTest.predict(X_data)
ftrainrms = sqrt(mean_squared_error(Y_data, y_xgbpredTest))
print("XGSPCA : testrms  {}".format(trainrms ) )


#=============================================================================
# ADD RBF
#=============================================================================


plt.figure(figsize=(8,8))
plt.scatter( Y_data, y_predTest )
plt.xlabel('Y_data', fontsize=12)
plt.ylabel('RF', fontsize=12)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter( Y_data, y_xgbpredTest )
plt.xlabel('Y_data', fontsize=12)
plt.ylabel('XGB', fontsize=12)
plt.show()
plt.figure(figsize=(8,8))
plt.scatter( Y_data, y_KNNpredTest )
plt.xlabel('Y_data', fontsize=12)
plt.ylabel('KNN', fontsize=12)
plt.show()



