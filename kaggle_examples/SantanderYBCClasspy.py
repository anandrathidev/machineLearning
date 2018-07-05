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

filepath = "/home/he159490/DS/Kaggle/SantanderValue//"
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
YBC,ylambda= boxcox(Y)
#YBC= boxcox(Y)
plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(YBC))
plt.show()
print("ylambda =  {}".format(ylambda))
def invboxcox(y,ld):
   if ld == 0:
      return(np.expm1(y))
   else:
      return(np.expm1(np.log1p(ld*y+1)/ld))

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

# In[6]:

print("Feature scaling...")
fulldata = data.append(testdata)
rscaler = RobustScaler()
rscaler.fit( fulldata )
dataScaled = rscaler.transform( data )
testdataScaled = rscaler.transform( testdata )
fulldataScaled = rscaler.transform( fulldata )


# In[6]:
print("Feature normalisation...")
fulldataBC = pd.DataFrame(fulldataScaled)
dataBC = pd.DataFrame(dataScaled)
testdataBC = pd.DataFrame(testdataScaled)
BCLAMBDA = {}
for col in range(data.shape[1]):
    xBC,Xlambda= boxcox( fulldataBC[col] +1 )
    BCLAMBDA[col] =  Xlambda
    fulldataBC[col] = xBC
    xBC =  boxcox(dataBC[col] +1 , Xlambda)
    dataBC[col] = xBC
    testdataBC[col] = np.array(boxcox(testdataBC[col]+1,Xlambda))
    xBC =  boxcox(testdataBC[col] +1 , Xlambda)
    testdataBC[col] = xBC

print("BOCOX {} {}".format(len(BCLAMBDA), BCLAMBDA))



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

###########  PCA ###########
print("train pca...")
pca = decomposition.PCA(0.999)
pca = pca.fit(fulldataScaledLog)
fulldataScaledPCA = pca.transform(fulldataScaledLog)
print("pca transform train... ")
dataPCA = pca.transform(dataScaledLog)
print("Test Data pca... ")
testdataPCA = pca.transform(testdataScaledLog)


# In[10]:

Y=YBC
X_data, X_test, Y_data, y_test = train_test_split(dataPCA, Y, test_size=0.15, random_state=42)
print(X_data.shape)
dataScaledLogPD = pd.DataFrame(dataScaledLog,columns=data.columns)
#XTrainPD = pd.DataFrame(X_data,columns=data.columns)
#XTestLogPD = pd.DataFrame(X_test,columns=data.columns)

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
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn.metrics import mean_squared_error
from math import sqrt

np.random.seed(42)
import pandas as pd



# In[87]:


lgbmx1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        max_depth=2000,
                        min_data_in_leaf=2,
                        learning_rate= 0.059,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.8,
                        bagging_freq= 2,
                        verbose= 0,
                        num_threads=6,
                        n_estimators=321)
lgbmx1.fit(X_data, Y_data,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1511)
y_lgbmx1 = lgbmx1.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_lgbmx1))
print("lgbmreg trainrms {}".format(  trainrms ) )


# In[75]:
from sklearn import utils

X_Cdata, X_Ctest, Y_Cdata, Y_Ctest = train_test_split(data, Y, test_size=0.20, random_state=42)
lab_enc = preprocessing.LabelEncoder()

lab_enc.fit(Y)
Y_FULL_encoded = lab_enc.transform(Y)
Y_data_encoded = lab_enc.transform(Y_Cdata)
Y_test_encoded = lab_enc.transform(Y_Ctest)
Y_data_encoded.shape

print(utils.multiclass.type_of_target(Y_data_encoded))

lgbmClass1 = lgb.LGBMClassifier(n_estimators=171, objective='multiclassova' )

lgbmClass1.fit(data, Y_FULL_encoded,
        eval_set=[(X_Ctest, Y_test_encoded)],
        early_stopping_rounds=111)

Y_lgbmClass1_encodedPredict = lgbmClass1.predict(X_Ctest)
trainrms = sqrt(mean_squared_error(
   lab_enc.inverse_transform( Y_test_encoded  ),
   lab_enc.inverse_transform( Y_lgbmClass1_encodedPredict)
                                   ))
print("lgbmClass1  trainrms {}".format(  trainrms ) )




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
                        num_threads=3,
                        boosting="dart")

lgbm1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        min_data_in_leaf=2,
                        learning_rate= 0.049,
                        feature_fraction= 0.9,
                        bagging_fraction= 0.8,
                        bagging_freq= 2,
                        verbose= 0,
                        n_estimators=2511)

lgbm2.fit(dataPCA, Y, eval_set=[(X_test, y_test)], eval_metric='l2',  early_stopping_rounds=511)
lgbm1.fit(dataPCA, Y, eval_set=[(X_test, y_test)], eval_metric='l1',  early_stopping_rounds=511)


# In[89]:


print("Predict  lgbm1...")
ytest_lgbm1= invboxcox(lgbm1.predict(testdataPCA),ylambda)
print("Predict  lgbm2...")
ytest_lgbm2= invboxcox(lgbm2.predict(testdataPCA),ylambda)

ytest_lgbmx1= invboxcox(lgbmx1.predict(testdataPCA),ylambda)

print("Predict  lgbmx2...")
ytest_lgbmClass1= lgbmClass1.predict(testdata)


plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm1)), ytest_lgbm1, s=100,  marker="s", label='ytest_lgbm1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbm1 ', fontsize=12)
plt.title("ytest_lgbm1 Distribution", fontsize=14)
plt.show()
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm2)), ytest_lgbmClass1, s=100,  marker="s", label='ytest_lgbmClass1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbmClass1 ', fontsize=12)
plt.title("ytest_lgbmClass1 Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_lgbmClass1, s=100,  marker="s", label='lgbm1_RFT ' )
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


# In[91]:


ypredavg = pd.DataFrame(dict(
    XLGB1=ytest_lgbmx1,
    LGB1=ytest_lgbm1,
    LGB2=ytest_lgbm2,
    CLASS=ytest_lgbmClass1,
    AVG= (ytest_lgbmClass1 + ytest_lgbm1 + ytest_lgbm2 + ytest_lgbmx1  )/4  ) )

print(ypredavg.columns)
ypredavg.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)
print(ypredavg.mean())
print(ypredavg.std())


# In[93]:


ySubmit = pd.DataFrame(dict( ID=testID,target=(ypredavg["AVG"] )))
ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)
print(ySubmit.shape)


# In[94]:




# In[ ]:


#ySubmit = pd.DataFrame(dict( ID=testID,target=ypredFinalFinal["lgbm"]  ))
#ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)
#print("write submit....")

