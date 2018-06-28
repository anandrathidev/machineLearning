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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

from sklearn import decomposition
import matplotlib.pyplot as plt

np.random.seed(1)

import pandas as pd

filepath = "F:/DataScience/Kagggle/SantanderValue/"
data_file = filepath + "train.csv"
y_RF_file = filepath + "/tmp/RF_y.csv"
#finalyrbf_file = filepath + "/tmp///RFrbf_y.csv"
yXGS_file = filepath + "/tmp/XGS_y.csv"

test_file = filepath + "test.csv"
test_result_file1 = filepath + "/tmp/ypredDFFinalDetails.csv"
test_result_file2 = filepath + "/tmp//FinalTestResult.csv"

test_result_file21 = filepath + "/tmp//FinalTestResultarget.csv"
test_result_file22 = filepath + "/tmp//FinalTestResultRetarget.csv"

data = None
Y = None
try:
  xdata = pd.read_csv(data_file)
  list(xdata.columns)
  Y = xdata["target"]
  data = xdata.drop(columns=['ID', 'target'])
except Exception as e:
  print(e)

#Y = np.log1p(Y)
Y = Y
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


from scipy.stats import boxcox

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
#=============================================================================
# start Feature selection
#=============================================================================


plt.plot(range(0,len(Y)),np.sort(Y))
plt.show()

#from sklearn.preprocessing import PowerTransformer

print("Feature selection...")
#FSelect = feature_selection.SelectPercentile(f_regression, percentile=30)
#FSelect.fit(data, Y)
#data = FSelect.transform(data)

testdata = testdata.drop(columns=['ID'])

fulldata = data.append(testdata)

from sklearn.cluster import MiniBatchKMeans


colsToRemove = []
for col in fulldata.columns:
  if col != 'ID' and col != 'target':
    if fulldata[col].std() == 0: 
      colsToRemove.append(col)
        
len(colsToRemove) 

# remove constant columns in the test set
testdata.drop(colsToRemove, axis=1, inplace=True)       
#testdata= FSelect.transform(testdata)
print("Test Data scaled...")

data.drop(colsToRemove, axis=1, inplace=True)

print("scale...")
rscaler = RobustScaler()
rscaler.fit(fulldata)

kmeans3300 =  MiniBatchKMeans(n_clusters=3300, init='k-means++', max_iter=20, batch_size=100, 
                                verbose=0, compute_labels=True, 
                                random_state=None, tol=0.0, 
                                max_no_improvement=10, 
                                init_size=None, 
                                n_init=600, 
                                reassignment_ratio=0.01)


print("train pca...")
pca = decomposition.PCA(0.99)
pca.fit(fulldata)

print("pca transform train...")
fulldata = pca.transform(fulldata)
print("cluster train...")
kmeans3300.fit(fulldata)

print("PCA data...")
data = rscaler.transform(data)
data = pca.transform(data)
cid = kmeans3300.transform(data)
data["cid"] = cid

print("Test Data pca...")
testdata = rscaler.transform(testdata)
cid = kmeans3300.transform(testdata)
testdata["cid"] = cid


#outliers = stats.zscore(data['_source.price']).apply(lambda x: np.abs(x) == 3)
#df_without_outliers = data[~outliers]

data.shape

#X_normalized = preprocessing.normalize(data, norm='l2')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt


print("Train & test split...")
X_data, X_test, Y_data, y_test = train_test_split(data, Y, test_size=0.26, random_state=42)
#print("Lin reg...")
lreg = LinearRegression()
lreg.fit(X_data,Y_data)
y_Lpred = lreg.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_Lpred))
print("LREG : trainrms {}".format(trainrms ) )
plt.figure(figsize=(8,8))
plt.scatter( y_test, y_Lpred)
plt.xlabel('index', fontsize=12)
plt.ylabel('Linear', fontsize=12)
plt.show()


#=============================================================================
# end Feature selection
#=============================================================================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

print("KNN ...{}".format("" ) )
knnreg = KNeighborsRegressor(n_neighbors=6, weights='distance', metric='minkowski', n_jobs=5)
knnreg.fit(data,Y)
y_KNNpred = knnreg.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_KNNpred))
print("KNN PCA : trainrms {}".format(trainrms ) )

print("RF ...{}".format("" ) )
RFregr = RandomForestRegressor(n_estimators=319, random_state=0, min_samples_leaf=3, oob_score=False, n_jobs=6)
# Fit to data using Maximum Likelihood Estimation of the parameters
RFregr.fit(X_data,Y_data)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred = RFregr.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_pred))
print("RFPCA : trainrms {}".format(trainrms ) )

plt.figure(figsize=(8,8))
plt.scatter( y_test, y_pred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('RF', fontsize=12)
plt.show()


#=============================================================================
# end RF
#=============================================================================
#=============================================================================
# start XGS
#=============================================================================

print("XGboost ...{}".format("" ) )
import xgboost

xgb = xgboost.XGBRegressor(n_estimators=121, learning_rate=0.59, gamma=5, subsample=0.81,
                           colsample_bytree=1, max_depth=17,  n_jobs=5)

# Fit to data using Maximum Likelihood Estimation of the parameters
xgb.fit(X_data,Y_data)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_xgbpred = xgb.predict(X_test)
ftrainrms = sqrt(mean_squared_error(y_test, y_xgbpred))
print("XGSPCA : trainrms {}".format(trainrms ) )


#=============================================================================
# ADD RBF
#=============================================================================


# Instanciate a Gaussian Process model
#kernel = C(7.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gp = GaussianProcessRegressor(kernel=None, alpha=0.0001,  n_restarts_optimizer=33, normalize_y=True)
#ygp = gp.fit(X_data,Y_data)
# Make the prediction on the meshed x-axis (ask for MSE as well)
#y_RBFpred = gp.predict(X_test)
#trainrms = sqrt(mean_squared_error(y_test, y_RBFpred))
#print("RBFPCA : trainrms {}".format(trainrms ) )



plt.figure(figsize=(8,8))
plt.scatter( y_test, y_xgbpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('XGB', fontsize=12)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter( y_test, y_KNNpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('KNN', fontsize=12)
plt.show()


plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()

#=============================================================================
# end RBF
#=============================================================================

import SantanderValueTestReverse

print("Run Test Models on Test  ...{}".format("" ) )
y_xgbpredTest_Test = xgbTest.predict(X_test)
y_predTest_Test = RFregrTest.predict(X_test)
y_KNNpredTest_Test = knnregTest.predict(X_test)



print("xgbFinal ...{}".format("" ) )
ypredDFFinal =  pd.DataFrame(dict( KNN=y_KNNpred, XGS=y_xgbpred, RF = y_pred) )
ypredDFFinal_Test =  pd.DataFrame(dict( KNNTest=y_KNNpredTest_Test, XGSTest= y_xgbpredTest_Test, RFTest = y_predTest_Test  ))

ypredDFFinal=ypredDFFinal.join(ypredDFFinal_Test)
print(ypredDFFinal.shape)
xgbFinal = xgboost.XGBRegressor(n_estimators=201, learning_rate=0.19, gamma=130, subsample=0.71,
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

print("Predict  RF test model...")
ytest_RFTest = invboxcox(RFregrTest.predict(testdata),ylambda)
print("Predict  xgb test model...")
ytest_XGSTest = invboxcox(xgbTest.predict(testdata),ylambda)
print("Predict  KNN test model...")
ytest_KNNpredTest= invboxcox(knnregTest.predict(testdata),ylambda)


print("Predict  RF...")
ytest_RF= invboxcox(RFregr.predict(testdata),ylambda)
print("Predict  xgb...")
ytest_XGS = invboxcox(xgbTest.predict(testdata),ylambda)
print("Predict  KNN...")
ytest_KNNpred= invboxcox(knnreg.predict(testdata),ylambda)


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

ypredDFFinalDetails = pd.DataFrame(dict( KNN=ytest_KNNpred, XGS= ytest_XGS, RF = ytest_RF  ))
print("ypredDFFinalDetails = {}".format(ypredDFFinalDetails.shape))
ypredDFFinalDetailsTest = pd.DataFrame(dict( KNNTest=ytest_KNNpredTest, XGSTest= ytest_XGSTest, RFTest = ytest_RFTest  ))
print("ypredDFFinalDetails = {}".format(ypredDFFinalDetails.shape))
ypredDFFinalDetails=ypredDFFinalDetails.join(ypredDFFinalDetailsTest)
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

ypredDFFinalDetailsavg = pd.DataFrame(dict( XGS_RF_KNN=( ytest_XGS + ytest_RF + ytest_KNNpred)/3  ))
ypredFinalFinal = pd.DataFrame(dict(FXGS=yxgsTestFinal, FRF=yRFTestFinal ))
ypredFinalFinal = ypredFinalFinal.join(ypredDFFinalDetailsavg).join(ypredDFFinalDetails)

print(ypredDFFinal.columns)
print(ypredDFFinalDetails.columns)
print(ypredFinalFinal.columns)

ypredFinalFinal.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)


print("Write test data ...")
ypredDFFinalDetails["ID"] = testID

ypredDFFinalDetails["target"] = (ytest_KNNpred + ytest_RF + ytest_XGS )/3

ypredDFFinalDetails.to_csv(test_result_file2, index=False)

sumbission1 = ypredDFFinalDetails[["ID"]]
sumbission1["target"] = (ytest_KNNpred + ytest_RF + ytest_XGS )/3
sumbission1.to_csv(test_result_file21, index=False)

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