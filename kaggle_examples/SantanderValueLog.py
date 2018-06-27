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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn import feature_selection

from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import numpy as np

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
print("train pca...")
pca = decomposition.PCA(0.995)
pca.fit(fulldata)

print("PCA data...")
data = rscaler.transform(data)
data = pca.transform(data)

print("Test Data pca...")
testdata = rscaler.transform(testdata)
testdata = pca.transform(testdata)

#outliers = stats.zscore(data['_source.price']).apply(lambda x: np.abs(x) == 3)
#df_without_outliers = data[~outliers]

print("PCA...")
data.shape

#X_normalized = preprocessing.normalize(data, norm='l2')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

#print("Lin reg...")
lreg = LinearRegression()
lreg.fit(data,Y)
y_Lpred = lreg.predict(data)
trainrms = sqrt(mean_squared_error(Y, y_Lpred))
print("LREG : trainrms {}".format(trainrms ) )
plt.figure(figsize=(8,8))
plt.scatter( Y, y_Lpred)
plt.show()

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

print("KNN ...{}".format("" ) )
knnreg = KNeighborsRegressor(n_neighbors=1)
knnreg.fit(data,Y)
y_KNNpred = knnreg.predict(data)
trainrms = sqrt(mean_squared_error(Y, y_KNNpred))
print("KNN PCA : trainrms {}".format(trainrms ) )


print("Rad ...{}".format("" ) )
from sklearn.neighbors import RadiusNeighborsRegressor
radreg  = RadiusNeighborsRegressor(weights='distance', radius=10.3)
radreg.fit(data, Y) 
y_radpred = radreg.predict(data)
trainrms = sqrt(mean_squared_error(Y, y_radpred))
print("Rad PCA : trainrms {}".format(trainrms ) )

#=============================================================================
# end Feature selection
#=============================================================================


# Instanciate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
print("RF ...{}".format("" ) )
RFregr = RandomForestRegressor(n_estimators=111, random_state=0, oob_score=True)

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

print("XGboost ...{}".format("" ) )
import xgboost
xgb = xgboost.XGBRegressor(n_estimators=111, learning_rate=0.59, gamma=5, subsample=0.81,
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

ypredDFFinal =  pd.DataFrame(dict(KNN = knnreg, RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred, Y=Y))

trainrms = sqrt(mean_squared_error(Y, y_RBFpred))
print("RBFPCA : trainrms {}".format(trainrms ) )

ypredDFFinal.to_csv(finalyrbf_file)


plt.figure(figsize=(8,8))
plt.scatter( Y, y_pred )
plt.show()
plt.figure(figsize=(8,8))
plt.scatter( Y, y_RBFpred )
plt.show()
plt.figure(figsize=(8,8))
plt.scatter( Y, y_xgbpred )
plt.show()
plt.figure(figsize=(8,8))
plt.scatter( Y, y_KNNpred )
plt.show()


plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()

#=============================================================================
# end RBF
#=============================================================================

print("xgbFinal ...{}".format("" ) )
ypredDFFinal =  pd.DataFrame(dict( KNN=y_KNNpred, RBF = y_RBFpred, XGS= y_xgbpred, RF = y_pred))
ypredDFFinal.shape



xgbFinal = xgboost.XGBRegressor(n_estimators=201, learning_rate=0.19, gamma=130, subsample=0.71,
                           colsample_bytree=1, max_depth=15)
xgbFinal.fit(ypredDFFinal,Y)
YxgbFinal = xgbFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(Y, YxgbFinal))
print("XGSFinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(Y, YxgbFinal, s=100,  marker="s", label='YxgbFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YxgbFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

gpFinal = GaussianProcessRegressor(kernel=None, alpha=0.0001,  n_restarts_optimizer=33, normalize_y=True)
gpFinal.fit(ypredDFFinal, Y)
YGPFinal = gpFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(Y, YGPFinal))
print("YGPinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(Y, YGPFinal, s=100,  marker="s", label='YGPFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YGPFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

RFregrFinal = RandomForestRegressor(n_estimators=311, random_state=0, oob_score=True)
# Fit to data using Maximum Likelihood Estimation of the parameters
RFregrFinal.fit(ypredDFFinal, Y)
YRFFinal = RFregrFinal.predict(ypredDFFinal)
trainrms = sqrt(mean_squared_error(Y, YRFFinal))
print("YRFFinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(Y, YRFFinal, s=100,  marker="s", label='YRFFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YRFFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()

lregFinal = LinearRegression()
lregFinal.fit(data,Y)
YLFinal  = lregFinal.predict(data)
trainrms = sqrt(mean_squared_error(Y, YLFinal ))
print("YRFFinal : trainrms {}".format(trainrms ) )

plt.figure(figsize=(11,11))
plt.scatter(Y, YLFinal, s=100,  marker="s", label='LinFinal')
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
ytest_RF = invboxcox(RFregr.predict(testdata),ylambda)

print("Predict  xgb...")
ytest_XGS = invboxcox(xgb.predict(testdata),ylambda)

print("Predict  rbf...")
ytest_RBF = invboxcox(gp.predict(testdata),ylambda)

print("Predict  KNN...")
ytest_KNNpred= invboxcox(knnreg.predict(testdata),ylambda)

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_XGS)), ytest_XGS, s=100,  marker="s", label='RF-XGS')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_XGS', fontsize=12)
plt.title("ytest_XGS Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_XGS)), ytest_RBF, s=100,  marker="s", label='ytest_RBF')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_RBF', fontsize=12)
plt.title("ytest_RBF Distribution", fontsize=14)
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
plt.scatter(ytest_RF, ytest_RBF, s=100,  marker="s", label='RF-RBF')
plt.scatter(ytest_XGS, ytest_RBF, s=100,  marker="s", label='XGS-RBF')
plt.scatter(ytest_XGS, ytest_KNNpred, s=100,  marker="s", label='XGS-KNN')
plt.scatter(ytest_RBF, ytest_KNNpred, s=100,  marker="s", label='ytest_RBF, ytest_KNNpred ')
plt.scatter(ytest_RF, ytest_KNNpred, s=100,  marker="s", label='ytest_RF, ytest_KNNpred ')
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Distribution", fontsize=14)
plt.show()



ypredDFFinalDetails = pd.DataFrame(dict(XGS= ytest_XGS, RF = ytest_RF, KNN=ytest_KNNpred, RBF=ytest_RBF, Target= ( 2*ytest_XGS + ytest_RF + 2*ytest_KNNpred)/5  ))
ypredDFFinalDetails.to_csv(test_result_file1, index=False)
ypredDFFinalDetails = pd.DataFrame(dict( KNN=ytest_KNNpred, RBF = ytest_RBF, XGS= ytest_XGS, RF = ytest_RF  ))

ypredDFFinalDetails.shape

print("Predict  xgs final ...")
yxgsTestFinal = invboxcox(xgbFinal.predict(ypredDFFinalDetails), ylambda)
ygpTestFinal = invboxcox(gpFinal.predict(ypredDFFinalDetails), ylambda) 
yRFTestFinal = invboxcox(RFregrFinal.predict(ypredDFFinalDetails), ylambda) 


#yradTestFinal = radregF.predict(ypredDFFinalDetails)
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(yxgsTestFinal)), yxgsTestFinal, s=100,  marker="s", label='yxgsTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('yxgsTestFinal ', fontsize=12)
plt.title("yxgsTestFinal Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ygpTestFinal)), ygpTestFinal, s=100,  marker="s", label='ygpTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('ygpTestFinal ', fontsize=12)
plt.title("ygpTestFinal Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(yRFTestFinal)), yRFTestFinal, s=100,  marker="s", label='yRFTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('yRFTestFinal ', fontsize=12)
plt.title("yRFTestFinal Distribution", fontsize=14)
plt.show()


ypredFinalFinal = pd.DataFrame(dict(FXGS=yxgsTestFinal, FRF=yRFTestFinal, FGP= ygpTestFinal ))
ypredFinalFinal.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)

print("Write test data ...")
ypredDFFinalDetails["ID"] = testID
ypredDFFinalDetails["YGP"] = ygpTestFinal

ypredDFFinalDetails["ReTarget"] = (ytest_RBF + ytest_KNNpred + ytest_XGS + 2*ygpTestFinal)/5
ypredDFFinalDetails["Target"] =  (ytest_RF + ytest_KNNpred + ytest_XGS)/3  

ypredDFFinalDetails.to_csv(test_result_file2, index=False)

sumbission1 = ypredDFFinalDetails[["ID"]]
sumbission1["target"] = (ytest_RF + ytest_KNNpred + ytest_XGS)/3  
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