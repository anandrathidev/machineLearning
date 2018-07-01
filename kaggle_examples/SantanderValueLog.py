# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 11:43:27 2018

@author: anandrathi
"""


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import  MaxAbsScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

from sklearn import decomposition
import matplotlib.pyplot as plt

np.random.seed(42)

import pandas as pd

filepath = "F:/DataScience/Kagggle/SantanderValue/"
filepath = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/"

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

plt.figure(figsize=(8,8))
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

###########  Scale ###########
print("scale...")
rscaler = RobustScaler()
rscaler = MaxAbsScaler()

rscaler.fit(fulldata)
dataScaled = rscaler.transform(data)
sum(dataScaled)
testdataScaled = rscaler.transform(testdata)
sum(testdataScaled )
fulldataScaled = rscaler.transform(fulldata)
sum(fulldataScaled)

fulldataScaledLog =  np.log1p(fulldataScaled)
dataScaledLog =  np.log1p(dataScaled)
testdataScaledLog =  np.log1p(testdataScaled)

###########  PCA ###########
print("train pca...")
pca = decomposition.PCA(0.975)
pca = pca.fit(fulldataScaled)

fulldataScaledPCA = pca.transform(fulldataScaled)
print("pca transform train...")
dataPCA = pca.transform(dataScaled)
print("Test Data pca...")
testdataPCA = pca.transform(testdataScaled)

###########  SVD  ###########
from sklearn.decomposition import TruncatedSVD
import math
n_components = int( 5 * math.sqrt(fulldataScaled.shape[1]) )
n_components =  int( 0.81 * fulldataScaledLog.shape[1] )
svd = TruncatedSVD(n_components , n_iter=13, random_state=42)
svd.fit(fulldataScaledLog)
print("%d: Percentage explained: %s\n" % (n_components, svd.explained_variance_ratio_.sum()))
fulldataScaledSVD  = svd.transform( fulldataScaledLog)
print( " fulldataScaledSVD  {}".format(fulldataScaledSVD .shape))
print("svd transform train...")
dataSVD = svd.transform(dataScaledLog)
print("Test Data SVD...")
testdataSVD = svd.transform(testdataScaledLog)

print("fullData pca...")
fulldata = pca.transform(fulldata)

####################### Save Transformations ################################

np.savetxt( filepath  + "//fulldataScaledSVD81.csv" , fulldataScaledSVD , delimiter=",")
np.savetxt( filepath  + "//dataSVD81.csv" , dataSVD , delimiter=",")
np.savetxt( filepath  + "//testdataSVD81.csv" , testdataSVD , delimiter=",")

np.savetxt( filepath  + "//fulldataScaledPCA97.csv" , fulldataScaledPCA , delimiter=",")
np.savetxt( filepath  + "//dataPCA97.csv" , dataPCA , delimiter=",")
np.savetxt( filepath  + "//testdataPCA97.csv" , testdataPCA , delimiter=",")

np.savetxt( filepath  + "//fulldataScaled.csv" , fulldataScaled, delimiter=",")
np.savetxt( filepath  + "//dataScaled.csv" , dataScaled, delimiter=",")
np.savetxt( filepath  + "//testdataScaled.csv" , testdataScaled , delimiter=",")

"""
from sklearn.cluster import KMeans
kmeans3300 =  KMeans(n_clusters=711, init='k-means++', max_iter=7,

#outliers = stats.zscore(data['_source.price']).apply(lambda x: np.abs(x) == 3)
#df_without_outliers = data[~outliers]
                                verbose=0,
                                random_state=None, tol=0.0,
                                n_init=900, n_jobs=7,
                                precompute_distances=True)
print("cluster train...")
kmeans3300.fit(fulldataScaledPCA)
cidFullSVD = kmeans3300.transform(fulldataScaledPCA)

cidTrain = kmeans3300.transform(dataPCA)
np.savetxt( filepath  + "//KmeansTrain.csv" , cidTrain, delimiter=",")
dataPCAKmeans = np.append(dataPCA, cidTrain, axis=1)

print("Test Data pca...")
cidTest = kmeans3300.transform(testdataPCA)
testdataPCAKmeans = np.append(testdataPCA, cidTest, axis=1)
np.savetxt( filepath  + "//KmeansTest.csv" , cidTrain, delimiter=",")
"""
