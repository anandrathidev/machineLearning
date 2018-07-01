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

####################### Save Transformations ################################
fulldataScaledSVD  = pd.read_csv( filepath  + "//fulldataScaledSVD81.csv" ,   delimiter=","  , header=None).values
print(" loaded fulldataScaledSVD ")
dataSVD = pd.read_csv( filepath  + "//dataSVD81.csv" ,  delimiter=","  , header=None).values
print(" loaded dataSVD ")
testdataSVD  = pd.read_csv( filepath  + "//testdataSVD81.csv" , delimiter=","  , header=None).values
print(" loaded  testdataSVD ")

fulldataScaledPCA  = pd.read_csv( filepath  + "//fulldataScaledPCA97.csv" , delimiter="," , header=None).values
print(" loaded  fulldataScaledPCA ")
dataPCA = pd.read_csv( filepath  + "//dataPCA97.csv" ,  delimiter=",", header=None).values
print(" loaded  dataPCA ")
testdataPCA  = pd.read_csv( filepath  + "//testdataPCA97.csv" ,  delimiter=","   , header=None).values
print(" loaded  testdataPCA ")

fulldataScaled = pd.read_csv( filepath  + "//fulldataScaled.csv" ,  delimiter=","   ,  header=None).values
print(" loaded  fulldataScaled ")
dataScaled = pd.read_csv( filepath  + "//dataScaled.csv" ,  delimiter=","   , header=None).values
print(" loaded  dataScaled ")
testdataScaled  = pd.read_csv( filepath  + "//testdataScaled.csv" ,  delimiter=","   , header=None).values
print(" loaded  testdataScaled  ")

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

print("train pca...")
pca = decomposition.PCA(0.975)
pca = pca.fit(fulldataScaledPCA)
fulldataScaledPCA2 = pca.transform(fulldataScaledPCA)
print("pca transform train...")
dataPCA2 = pca.transform(dataPCA)
print("Test Data pca...")
testdataPCA2= pca.transform(testdataPCA)

#X_normalized = preprocessing.normalize(data, norm='l2')

print("Train & test split...")
#X_data, X_test, Y_data, y_test = train_test_split(dataSVD, Y, test_size=0.20, random_state=42)
#X_data, X_test, Y_data, y_test = train_test_split(dataPCA , Y, test_size=0.25, random_state=42)
X_data, X_test, Y_data, y_test = train_test_split(dataScaled, Y, test_size=0.25, random_state=42)

#=============================================================================
# end Feature selection
#=============================================================================
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

print("KNN ...{}".format("" ) )
knnreg = KNeighborsRegressor(n_neighbors=11, weights='distance',  n_jobs=5)
knnreg.fit(X_data,Y_data)
y_KNNpred = knnreg.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_KNNpred))
print("KNN PCA : trainrms {}".format(trainrms ) )
plt.figure(figsize=(8,8))
plt.scatter( y_test, y_KNNpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('RF', fontsize=12)
plt.show()

print("RF ...{}".format("" ) )
n_components=int(X_data.shape[1])
RFregr = RandomForestRegressor(n_estimators=319, random_state=42, max_features=int(0.33*n_components),
                               min_samples_leaf=2, min_samples_split=21, oob_score=False, n_jobs=6)
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

from sklearn.svm import NuSVR
NuSVRreg = NuSVR(C=65.0, nu=.99)
params = {'C':[ 45, 55,65 ], 'nu' : [ 1/i for i in range(1,10) ]  }
NuSVRreg = GridSearchCV(NuSVRreg, params)
NuSVRreg.fit(X_data,Y_data)
# Make the prediction on the meshed x-axis (ask for MSE as well)
y_NuSVRreg = NuSVRreg.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_NuSVRreg))
print("NuSVRreg : trainrms {}".format(trainrms ) )
plt.figure(figsize=(8,8))
plt.scatter( y_test, y_NuSVRreg)
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
params = {'min_child_weight':[4,5], 'gamma':[ 0.3,0.6],  'subsample':[0.6, 1.1 ],
'colsample_bytree':[i/10.0 for i in range(6,11)], 'max_depth': [21,31,41]}
xgb = xgboost.XGBRegressor(n_estimators=111, learning_rate=0.61, n_jobs=6)
xgbgrid = GridSearchCV(xgb, params)
xgbgrid.fit(X_data,Y_data)
y_xgbpred = xgbgrid.best_estimator_.predict(X_test)
ftrainrms = sqrt(mean_squared_error(y_test, y_xgbpred))
print("XGSPCA : trainrms {}".format(trainrms ) )

plt.figure(figsize=(8,8))
plt.scatter( y_test, y_xgbpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('XG', fontsize=12)
plt.show()


#=============================================================================
# ADD RBF
#=============================================================================


#kernel = gaussian_process.kernels.RBF(length_scale=100.0, length_scale_bounds= (1e-07, 10000.0))
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
kernel  = ConstantKernel(20.0) * RBF(length_scale=55.0, length_scale_bounds= (1, 70.0))
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, normalize_y=False, )
gp.fit(X_data,Y_data)
#Make the prediction on the meshed x-axis (ask for MSE as well)
y_RBFpred = gp.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_RBFpred))
print("RBFPCA : trainrms {}".format(trainrms ) )
plt.figure(figsize=(8,8))
plt.scatter( y_test, y_RBFpred )
plt.xlabel('ytest', fontsize=12)
plt.ylabel('RB', fontsize=12)
plt.show()



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