
# coding: utf-8

# In[5]:


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

filepath = "/home/he159490/DS/Kaggle/SantanderValue//"
#filepath = "F:/DataScience/Kagggle/SantanderValue/"
#filepath = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/"

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

Y = np.log1p(Y)
YBC, ylambda = boxcox(Y)
#YBC=Y
def invboxcox(y,ld):
   if ld == 0:
      return(np.exp(y))
   else:
      return(np.exp(np.log(ld*y+1)/ld))
def invboxcox(y,ld):
     return(np.expm1(y))
Yt = invboxcox(YBC,ylambda)
Ydf = pd.DataFrame(dict(Y=Y,Yt=Yt))

plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(YBC))
plt.show()
#Y=YBC
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

fulldata = data.append(testdata)


# In[6]:


datastd = pd.DataFrame(data.std(), columns=["std"])
datastd[datastd["std"]==0]


# In[7]:


###########  Scale ###########
print("scale...")
rscaler = RobustScaler()
rscaler = MaxAbsScaler()

rscaler.fit(fulldata)
dataScaled = rscaler.transform(data)
sum(dataScaled)
testdataScaled = rscaler.transform(testdata)
print("Test Data scaled...")
sum(testdataScaled )
fulldataScaled = rscaler.transform(fulldata)
sum(fulldataScaled)

fulldataScaledLog =  np.log1p(fulldataScaled)
dataScaledLog =  np.log1p(dataScaled)
testdataScaledLog =  np.log1p(testdataScaled)



print("Done...")
    


# In[8]:


###########  PCA ###########
print("train pca...")
pca = decomposition.PCA(0.999)
pca = pca.fit(fulldataScaledLog)

fulldataScaledPCA = pca.transform(fulldataScaledLog)
print("pca transform train...")
dataPCA = pca.transform(dataScaledLog)
print("Test Data pca...")
testdataPCA = pca.transform(testdataScaledLog)


# In[10]:


X_data, X_test, Y_data, y_test = train_test_split(dataPCA, Y, test_size=0.09, random_state=42)
print(X_data.shape)
dataScaledLogPD = pd.DataFrame(dataScaledLog,columns=data.columns)
#XTrainPD = pd.DataFrame(X_data,columns=data.columns)
#XTestLogPD = pd.DataFrame(X_test,columns=data.columns)


# In[11]:


def ExploreDataScaled(start,end,size=4):
    import matplotlib.pyplot as plt
    i=0
    collist=[]
    for c in XTrainPD.columns[start:end] :
        collist.append(c)
        if (abs(np.std(XTrainPD[c]) - np.std(XTestLogPD[c])) / np.std(XTestLogPD[c])) * 100.0 > 50.0:
            i=i+1
            fig, ax = plt.subplots(1,2)
            ax[0].hist(np.log1p(XTrainPD[c]), 100, normed=1, facecolor='blue', alpha=0.55)
            ax[0].set_title('TrainScale {}'.format(c))
            ax[1].hist(np.log1p(XTestLogPD[c]), 100, normed=1, facecolor='blue', alpha=0.55)
            ax[1].set_title('XTestLogPD {}'.format(c))
            plt.show()
    return collist 


# In[12]:


#diffcollist = ExploreDataScaled(start=10,end=100)


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


# In[17]:



def DoKNN(X_data, X_test, Y_data, y_test, n_neighbors, txt, printplot=False):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    knnreg = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance',  n_jobs=8)
    knnreg.fit(X_data,Y_data)
    y_KNNpred = knnreg.predict(X_test)
    trainrms = sqrt(mean_squared_error(y_test, y_KNNpred))
    print("KNN {} : neighbors {} :trainrms {}".format(txt, n_neighbors, trainrms ) )
    if printplot:
        plt.figure(figsize=(8,8))
        plt.scatter( y_test, y_KNNpred )
        plt.xlabel('ytest', fontsize=12)
        plt.ylabel( "KNN {} : neighbors {} : trainrms {}".format(txt, n_neighbors, trainrms ) , fontsize=12)
        plt.show()
    return trainrms,y_KNNpred

trainrms,yKNN = DoKNN(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, n_neighbors=7 , txt="log")


# In[18]:


def DoRF(X_data, X_test, Y_data, y_test, txt, printplot=False):
    n_components=int(X_data.shape[1])
    params = {'n_estimators' : [  215], 
              "min_samples_leaf": [1,2] ,
              "min_samples_split" : [2] }
    reg = RandomForestRegressor(min_samples_split=2,n_estimators=115,n_jobs=11)
    grid = GridSearchCV(reg, params, n_jobs=9)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    reg.fit(X_data,Y_data)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred = reg.predict(X_test)
    trainrms = sqrt(mean_squared_error(y_test, y_pred))
    #print("RF {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) )
    print("RF {} : trainrms {}".format(txt, trainrms ) )
    if printplot:
        plt.figure(figsize=(8,8))
        plt.scatter( y_test, y_pred)
        plt.xlabel('ytest', fontsize=12)
        #plt.ylabel( "RF {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) , fontsize=12)
        plt.show()
    #return trainrms,y_pred,grid.best_params_
    return trainrms,y_pred

#rf_rms,yRF,RFParams = DoRF(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")
#rf_rms,yRF  = DoRF(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")

print("rf_rms : {}".format(rf_rms))
# In[19]:


def DoXGS(X_data, X_test, Y_data, y_test, txt, printplot=False):
    import xgboost
    reg = xgboost.XGBRegressor(n_estimators=111, learning_rate=0.61, n_jobs=11)
    params = {'min_child_weight':[4], 
              'gamma':[ 0.3],  
              'subsample':[1.0 ], 
              'colsample_bytree':[0.8], 
              'max_depth': [101]}
    grid = GridSearchCV(reg, params, n_jobs=10)
    reg.fit(X_data,Y_data)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred = reg.predict(X_test)
    trainrms = sqrt(mean_squared_error(y_test, y_pred))
    #print("xgboost {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) )
    if printplot:
        plt.figure(figsize=(8,8))
        plt.scatter( y_test, y_pred)
        plt.xlabel('ytest', fontsize=12)
        #plt.ylabel( "xgboost {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) , fontsize=12)
        plt.show()
    #return trainrms,y_pred,grid.best_params_
    return trainrms,y_pred

#xgs_rms,yXGB =DoXGS(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")
#xgs_rms,yXGB,XGSParams=DoXGS(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")
#print("xgboost {} : grid.best_params {} : trainrms {}".format("log", XGSParams, xgs_rms ) )


# In[20]:


#xgs_rms,yXGB,XGSParams=DoXGS(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")
#print("xgboost {} : grid.best_params {} : trainrms {}".format("log", XGSParams, xgs_rms ) )


# In[21]:


#print(xgs_rms)
#print("xgboost {} : grid.best_params {}  ".format( xgs_rms,XGSParams ) )
#print(yXGB.shape)


# In[22]:


def DoNUF(X_data, X_test, Y_data, y_test, txt, printplot=False):
    from sklearn.svm import NuSVR
    reg = NuSVR(C=65.0, nu=.99 )
    params = {'C':[ 65 ], 'nu' : [.99]  }
    grid = GridSearchCV(reg, params, n_jobs=9)
    grid.fit(X_data,Y_data)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred = grid.predict(X_test)
    trainrms = sqrt(mean_squared_error(y_test, y_pred))
    print("NuSVRreg {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) )
    if printplot:
        plt.figure(figsize=(8,8))
        plt.scatter( y_test, y_pred)
        plt.xlabel('ytest', fontsize=12)
        plt.ylabel( "NuSVRreg {} : grid.best_params {} : trainrms {}".format(txt, grid.best_params_, trainrms ) , fontsize=12)
        plt.show()
    return trainrms,y_pred

#nu_rms,yNU=DoNUF(X_data=X_data, X_test=X_test, Y_data=Y_data, y_test=y_test, txt="log")


# In[57]:


#print("nu_rms  :  {}  ".format( nu_rms ) )


# In[87]:


import lightgbm as lgb

lgbmx1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=11,
                        learning_rate= 0.049,
                        feature_fraction= 0.9,
                        bagging_fraction= 0.8,
                        bagging_freq= 2,
                        verbose= 0,
                        n_estimators=121)
lgbmx1.fit(X_data, Y_data,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=1511)
y_lgbm1 = lgbmx1.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_lgbm1))
print("lgbmreg trainrms {}".format(  trainrms ) )


# In[75]:


import lightgbm as lgb
lgbmx2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=21,
                        learning_rate= 0.056,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.55,
                        bagging_freq= 7,
                        verbose= 0,
                        n_estimators=131,
                        num_threads=3,
                         boosting="dart")
lgbmx2.fit(X_data, Y_data,
        eval_set=[(X_test, y_test)],
        eval_metric='l2_root',
        early_stopping_rounds=111)
y_lgbmx2 = lgbmx2.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, y_lgbm2))
print("lgbmreg  dart l2_root trainrms {}".format(  trainrms ) )


# In[81]:


#lgbmreg trainrms 1.510527327858087
###########       1.510527327858087
import lightgbm as lgb
lgbmRFx = lgb.LGBMRegressor(objective='regression',
                        num_leaves=11,
                        learning_rate= 0.056,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.55,
                        bagging_freq= 7,
                        verbose= 0,
                        n_estimators=111,
                        num_threads=3,
                         boosting="rf")
lgbmRFx.fit(X_data, Y_data,
        eval_set=[(X_test, y_test)],
        eval_metric='l2_root',
        early_stopping_rounds=111)
yRF = lgbmRFx.predict(X_test)
trainrms = sqrt(mean_squared_error(y_test, yRF))
print("lgbmreg trainrms {}".format(  trainrms ) )


# In[82]:


print("Re arget DF ...{}".format("" ) )
XpredDFFinal =  pd.DataFrame(dict( RF = yRF, lgbm1=y_lgbm1, lgbm2=y_lgbm2 ) )
print(XpredDFFinal.shape)


# In[83]:


print("xgbFinal ...{}".format("" ) )
lgbmFinal = lgb.LGBMRegressor(objective='regression',
                        num_leaves=9,
                        learning_rate= 0.05,
                        feature_fraction= 0.8,
                        bagging_fraction= 0.8,
                        bagging_freq= 5,
                        verbose= 0,
                        n_estimators=311)
lgbmFinal.fit(XpredDFFinal, y_test,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        early_stopping_rounds=21)
y_lgbm = lgbm.predict(X_test)

print(X_test.shape)
print(X_data.shape)

YxgbFinal = lgbmFinal.predict(XpredDFFinal)
trainrms = sqrt(mean_squared_error(y_test, YxgbFinal))
print("XGSFinal : trainrms {}".format(trainrms ) )
plt.figure(figsize=(11,11))
plt.scatter(y_test, YxgbFinal, s=100,  marker="s", label='YxgbFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('YxgbFinal ', fontsize=12)
plt.title("Y", fontsize=14)
plt.show()
    


# In[84]:


lgbmRF = lgb.LGBMRegressor(objective='regression',
                        num_leaves=11,
                        learning_rate= 0.056,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.55,
                        bagging_freq= 7,
                        verbose= 0,
                        n_estimators=2511,
                        num_threads=3,
                         boosting="rf")

lgbm2 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=21,
                        learning_rate= 0.056,
                        feature_fraction= 0.91,
                        bagging_fraction= 0.55,
                        bagging_freq= 7,
                        verbose= 0,
                        n_estimators=2511,
                        num_threads=3,
                        boosting="dart")

lgbmx1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=11,
                        learning_rate= 0.049,
                        feature_fraction= 0.9,
                        bagging_fraction= 0.8,
                        bagging_freq= 2,
                        verbose= 0,
                        n_estimators=2511)

lgbmRF.fit(dataPCA, Y, eval_set=[(X_test, y_test)], eval_metric='l2',  early_stopping_rounds=511)
lgbm2.fit(dataPCA, Y, eval_set=[(X_test, y_test)], eval_metric='l2',  early_stopping_rounds=511)
lgbm1.fit(dataPCA, Y, eval_set=[(X_test, y_test)], eval_metric='l1',  early_stopping_rounds=511)


# In[89]:


print("Predict  RF...")
ytest_RF= invboxcox(lgbmRF.predict(testdataPCA),ylambda)
print("Predict  lgbm1...")
ytest_lgbm1= invboxcox(lgbm1.predict(testdataPCA),ylambda)
print("Predict  lgbm2...")
ytest_lgbm2= invboxcox(lgbm2.predict(testdataPCA),ylambda)

print("Predict  RFx...")
ytest_RFx= invboxcox(lgbmRFx.predict(testdataPCA),ylambda)
print("Predict  lgbmx1...")
ytest_lgbmx1= invboxcox(lgbmx1.predict(testdataPCA),ylambda)
print("Predict  lgbmx2...")
ytest_lgbmx2= invboxcox(lgbmx2.predict(testdataPCA),ylambda)


plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm1)), ytest_lgbm1, s=100,  marker="s", label='yxgsTestFinal')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbm1 ', fontsize=12)
plt.title("ytest_lgbm1 Distribution", fontsize=14)
plt.show()
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm2)), ytest_lgbm2, s=100,  marker="s", label='ytest_lgbm1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbm2 ', fontsize=12)
plt.title("ytest_lgbm2 Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_RF)), ytest_RF, s=100,  marker="s", label='ytest_lgbm2')
plt.xlabel('index', fontsize=12)
plt.ylabel('yRFTestFinal ', fontsize=12)
plt.title("yRFTestFinal Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_RF, s=100,  marker="s", label='lgbm1_RFT ' )
plt.xlabel('ytest_lgbm1', fontsize=12)
plt.ylabel('ytest_RF ', fontsize=12)
plt.title("RF lgbm1 Distribution", fontsize=14)
plt.show()

plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_lgbmx2, s=100,  marker="s", label='lgbm2_RFT ' )
plt.xlabel('ytest_lgbm1', fontsize=12)
plt.ylabel('ytest_lgbmx2 ', fontsize=12)
plt.title("RF lgbm2 Distribution", fontsize=14)
plt.show()


# In[91]:


ypredavg = pd.DataFrame(dict(
    XLGB1=ytest_lgbmx1, 
    XLGB2=ytest_lgbmx2, 
    XRF=ytest_RF,
    LGB1=ytest_lgbm1, 
    LGB2=ytest_lgbm2, 
    RF=ytest_RFx,
    AVG= (ytest_RF + ytest_lgbm1 + ytest_lgbm2 + ytest_RFx + ytest_lgbmx1 + ytest_lgbmx2  )/6  ) )

print(ypredavg.columns)
ypredavg.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)
print(ypredavg.mean())
print(ypredavg.std())


# In[93]:


ySubmit = pd.DataFrame(dict( ID=testID,target=(ypredavg["AVG"] )))
print(ySubmit.shape)


# In[94]:


ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)


# In[ ]:


#ySubmit = pd.DataFrame(dict( ID=testID,target=ypredFinalFinal["lgbm"]  ))
#ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)
#print("write submit....")

