
# coding: utf-8

# In[85]:


# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:07:42 2018

@author: anandrathi
"""

# coding: utf-8

# In[5]:


import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import  MaxAbsScaler
from sklearn import preprocessing
from sklearn.decomposition import SparsePCA
from sklearn import feature_selection
from sklearn.model_selection import train_test_split

from sklearn import decomposition
import matplotlib.pyplot as plt

import lightgbm as lgb

np.random.seed(42)

import pandas as pd

filepath = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/"
filepath = "F:/DataScience/Kagggle/SantanderValue/"
filepath = "/home/he159490/DS/Kaggle/SantanderValue//"

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

# In[]:
from scipy.stats import boxcox
Y = xdata["target"]
YBC,ylambda= boxcox(Y)
#YBC= boxcox(Y)
plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(YBC))
plt.show()
print("ylambda =  {}".format(ylambda))
# In[6]:

print("Feature selection...")
#FSelect = feature_selection.SelectPercentile(f_regression, percentile=30)
#FSelect.fit(data, Y)
#data = FSelect.transform(data)
colsToRemove = []
for col in data.columns:
  if col != 'ID' and col != 'target':
    if data[col].std() == 0  or testdata[col].std() == 0:
      pass
    if testdata[col].std() == 0:
      colsToRemove.append(col)

    #if abs(data[col].std() - testdata[col].std())*100/testdata[col].std()  > 89.0:
    #  colsToRemove.append(col)

print("Drop cols with std ==0 {} {}".format(len(colsToRemove), colsToRemove))
# remove constant columns in the test set
testdata.drop(colsToRemove, axis=1, inplace=True)
data.drop(colsToRemove, axis=1, inplace=True)


# In[86]:


# In[6]:

Y = xdata["target"]
YBC,ylambda= boxcox(Y)
ylambda=None

if ylambda is None:
   YBC=Y 
else:
   YBC= boxcox(Y,ylambda)

def invboxcox(y,ld):
   if ld is None:
      return y 
   if ld == 0:
      return(np.expm1(y))
   else:
      return(np.expm1(np.log1p(ld*y+1)/ld))
    
print(YBC[0:10])    


# In[87]:


print("Feature scaling...")
fulldata = data.append(testdata)
#rscaler = RobustScaler()
rscaler = preprocessing.Normalizer()
rscaler.fit( fulldata )
dataScaled = rscaler.transform( data )
testdataScaled = rscaler.transform( testdata )
fulldataScaled = rscaler.transform( fulldata )
#dataScaled = ( data )
#testdataScaled = ( testdata )
#fulldataScaled = ( fulldata )


# In[88]:


"""
colsToRemove = []
for col in data.columns:
  if col != 'ID' and col != 'target':
    if dataScaled[col].std() == 0  or testdataScaled[col].std() == 0:
      colsToRemove.append(col)
    if dataScaled[col].std() == 0:
      pass  
      #colsToRemove.append(col)

    #if abs(data[col].std() - testdata[col].std())*100/testdata[col].std()  > 89.0:
    #  colsToRemove.append(col)

colsToRemove = []
print("Drop cols with std ==0 {} {}".format(len(colsToRemove), colsToRemove))
"""


# In[89]:


# remove constant columns in the test set
testdataScaled = testdata.drop(colsToRemove, axis=1)
dataScaled = data.drop(colsToRemove, axis=1)


# In[ ]:


# In[6]:
print("Feature normalisation...")
fulldataBC = pd.DataFrame(fulldataScaled)
dataBC = pd.DataFrame(dataScaled)
testdataBC = pd.DataFrame(testdataScaled)
BCLAMBDA = {}
"""
for col in range(data.shape[1]):
    xBC,Xlambda= boxcox( fulldataBC[col] +1 )
    Xlambda=0
    BCLAMBDA[col] =  Xlambda
    fulldataBC[col] = xBC
    xBC =  boxcox(dataBC[col] +1 , Xlambda)
    dataBC[col] = xBC
    testdataBC[col] = np.array(boxcox(testdataBC[col]+1,Xlambda))
    xBC =  boxcox(testdataBC[col] +1 , Xlambda)
    testdataBC[col] = xBC
"""

#print("BOCOX {} {}".format(len(BCLAMBDA), BCLAMBDA))

# In[7]:

datastd = pd.DataFrame(data.std(), columns=["std"])
datastd[datastd["std"]==0]


# In[7]:
###########  Scale ###########
print("scale...")
#dataScaled = rscaler.transform(dataBC)
#sum(dataScaled)
print("Test Data scaled...")
#testdataScaled = rscaler.transform(testdataBC)
print("Full Data scaled...")
#fulldataScaled = rscaler.transform(fulldataBC)
## Ignore scaled data for now

# In[7]:

#fulldataScaledLog =  np.log1p(fulldataScaled)
#dataScaledLog =  np.log1p(dataScaled)
#testdataScaledLog =  np.log1p(testdataScaled)
dataScaledLog = (dataBC)
testdataScaledLog = (testdataBC)
fulldataScaledLog = (fulldataBC)
print("Done...")
# In[8]:

print(fulldataScaledLog[0:10])


# In[101]:


###########  PCA ###########
print("train pca...")
pca = decomposition.PCA(n_components=0.85, 
                        copy=True, whiten=False, svd_solver='full', 
                        tol=0.0 )
pca = pca.fit(fulldataScaledLog)

fulldataScaledPCA = pca.transform(fulldataScaledLog)
print("pca transform train... ")
dataPCA = pca.transform(dataScaledLog)
print("Test Data pca... ")
testdataPCA = pca.transform(testdataScaledLog)


# In[102]:


# In[10]:

#Y=YBC
X_data, X_test, Y_data, y_test = train_test_split(dataPCA, YBC, test_size=0.19, random_state=1)
print(X_data.shape)
dataScaledLogPD = pd.DataFrame(dataScaledLog,columns=data.columns)

# In[14]:


"""
diffcollist=[]
for c in XTrainPD.columns:
    if (abs(np.std(XTrainPD[c]) - np.std(XTestLogPD[c])) / np.std(XTestLogPD[c])) * 100.0 > 50.0:
        diffcollist.append(c)
#print(diffcollist)
"""
pass


# In[15]:


def ExploreData(start,end,size=4):
    import matplotlib.pyplot as plt
    for c in data.columns[start:end] :
        if (abs(np.std(fulldata[c]) - np.std(data[c])) / np.std(fulldata[c])) * 100.0 > 51.0:
            fig, ax = plt.subplots(1,3)
            #print("Columns Name {}".format(c))
            #out = pd.cut(data[c], bins=5, include_lowest=True)
            #out = pd.cut(fulldata[c], bins=5, include_lowest=True)
            #out.value_counts(sort=False).plot.bar(rot=0, color="r", figsize=(size,size))
            ax[0].hist(np.log1p(dataScaledLogPD[c]), 100, normed=1, facecolor='blue', alpha=0.75)
            ax[0].set_title('TrainScale {}'.format(c))
            ax[1].plot(np.sort(data[c]), 'go-', label='train', linewidth=2)
            ax[1].set_title('Train  {}'.format(c))
            ax[2].plot(np.sort(fulldata[c]), 'rs', label='full', linewidth=1)
            ax[2].set_title('Full  {}'.format(c))
            plt.show()

# In[16]:


import numpy as np
from scipy.stats import boxcox

from sklearn import decomposition
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(42)
import pandas as pd


# In[ ]:




print(X_data[0:2])

""""lgbmx1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=21,
                        max_depth=150,
                        min_data_in_leaf=3,
                        learning_rate= 0.051,
                        feature_fraction= 0.90,
                        bagging_fraction= 0.8,
                        bagging_freq= 5,
                        verbose=0,
                        num_threads=6,
                        n_estimators=321)
lgbmx1.fit(X_data, Y_data,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=511)
y_lgbmx1 = lgbmx1.predict(X_test)
trainrms = sqrt(mean_squared_error(invboxcox(y_test,ylambda), invboxcox(y_lgbmx1,ylambda)))
print("lgbmreg trainrms {}".format(  trainrms ) )
"""

from catboost import CatBoostRegressor
catgbmx1=CatBoostRegressor(iterations=50, depth=15, learning_rate=0.01, loss_function='RMSE', thread_count=11)
catgbmx1.fit(X_data, Y_data, eval_set=(X_test, y_test),plot=True)


# In[ ]:


trainrms = sqrt(mean_squared_error(invboxcox(y_test,ylambda), invboxcox(y_lgbmx1,ylambda)))
print("lgbmreg trainrms {}".format(  trainrms ) )

print('Plot feature importances...')
#print( list(lgbmx1.feature_importances_ ))
NameImp  = list(zip( X_data.columns , list(lgbmx1.feature_importances_ ) ))
print(NameImp)
import operator
print( list(NameImp.sort(key=operator.itemgetter(1))))

ax = lgb.plot_importance(lgbmx1, max_num_features=10)
plt.show()


# In[95]:


# In[75]:
from sklearn import utils

X_Cdata, X_Ctest, Y_Cdata, Y_Ctest = train_test_split(dataPCA, YBC, test_size=0.20, random_state=42)
lab_enc = preprocessing.LabelEncoder()


# In[94]:


print(np.unique(YBC).shape)


# In[96]:


lab_enc.fit(YBC)
Y_FULL_encoded = lab_enc.transform(YBC)
Y_data_encoded = lab_enc.transform(Y_Cdata)
Y_test_encoded = lab_enc.transform(Y_Ctest)
Y_data_encoded.shape

print(utils.multiclass.type_of_target(Y_data_encoded))

lgbmClass1 = lgb.LGBMClassifier(n_estimators=40, 
                               num_threads=10,
                                objective='multiclassova' )

lgbmClass1.fit(data, Y_FULL_encoded,
        eval_set=[(X_Ctest, Y_test_encoded)],
        early_stopping_rounds=111)

Y_lgbmClass1_encodedPredict = lgbmClass1.predict(X_Ctest)
trainrms = sqrt(mean_squared_error(
   lab_enc.inverse_transform( Y_test_encoded  ),
   lab_enc.inverse_transform( Y_lgbmClass1_encodedPredict)
                                   ))
print("lgbmClass1  trainrms {}".format(  trainrms ) )


# In[ ]:


# In[82]:

lgbm2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        min_data_in_leaf=2,
                        learning_rate= 0.055,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.65,
                        bagging_freq= 7,
                        verbose= 0,
                        n_estimators=1511,
                        num_threads=4,
                        boosting="dart")

lgbm1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        min_data_in_leaf=2,
                        learning_rate= 0.049,
                        feature_fraction= 0.9,
                        bagging_fraction= 0.8,
                        bagging_freq= 2,
                        verbose= 0,
                        num_threads=4,
                        n_estimators=2511)

lgbm2.fit(dataPCA, YBC, eval_set=[(dataBC, YBC)], eval_metric='l2',  early_stopping_rounds=511)
lgbm1.fit(dataPCA, YBC, eval_set=[(dataBC, YBC)], eval_metric='l1',  early_stopping_rounds=511)


# In[ ]:


# In[89]:

print("Predict  lgbm1...")
ytest_lgbm1= invboxcox(lgbm1.predict(testdataPCA),ylambda)
print("Predict  lgbm2...")
ytest_lgbm2= invboxcox(lgbm2.predict(testdataPCA),ylambda)

print("Predict  lgbmx1...")
ytest_lgbmx1= invboxcox(lgbmx1.predict(testdataPCA),ylambda)


# In[ ]:


print("Predict  lgbmClass1...")
ytest_lgbmClass1= lgbmClass1.predict(testdata)


# In[ ]:


ytest_lgbmInvClass1 = lab_enc.inverse_transform( ytest_lgbmClass1  )
print(ytest_lgbmInvClass1[0:10])

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm1)), ytest_lgbm1, s=100,  marker="s", label='ytest_lgbm1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbm1 ', fontsize=12)
plt.title("ytest_lgbm1 Distribution", fontsize=14)
plt.show()
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm2)), ytest_lgbmInvClass1, s=100,  marker="s", label='ytest_lgbmClass1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbmClass1 ', fontsize=12)
plt.title("ytest_lgbmClass1 Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_lgbmInvClass1, s=100,  marker="s", label='lgbm1_RFT ' )
plt.xlabel('ytest_lgbm1', fontsize=12)
plt.ylabel('ytest_lgbmClass1 ', fontsize=12)
plt.title("ytest_lgbmClass1 lgbm1 Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_lgbmx1, s=100,  marker="s", label='lgbm1_lgbmx1' )
plt.xlabel('ytest_lgbm1', fontsize=12)
plt.ylabel('ytest_lgbmx1 ', fontsize=12)
plt.title("RF lgbm2 Distribution", fontsize=14)
plt.show()


# In[ ]:


# In[91]:

ypredavg = pd.DataFrame(dict(
    XLGB1=ytest_lgbmx1,
    LGB1=ytest_lgbm1,
    LGB2=ytest_lgbm2,
    CLASS=ytest_lgbmInvClass1,
    AVG= (ytest_lgbm1 + ytest_lgbm2   )/2  ) )

print(ypredavg.columns)
ypredavg.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)
print(ypredavg.mean())
print(ypredavg.std())


# In[ ]:


# In[93]:
ySubmit = pd.DataFrame(dict( ID=testID,target=(ypredavg["AVG"] )))
ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)
print(ySubmit.shape)

