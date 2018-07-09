
# coding: utf-8

# In[2]:


"""
Created on Sun Jul  8 14:49:33 2018

@author: anandrathi
"""


import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import  MaxAbsScaler
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.grid_search import GridSearchCV

from sklearn import decomposition
import matplotlib.pyplot as plt

import lightgbm as lgb

np.random.seed(42)

import pandas as pd

filepath = "/home/he159490/DS/Kaggle/SantanderValue//"

data_file = filepath + "train.csv"
test_file = filepath + "test.csv"

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
YBC = np.log1p(Y)
plt.figure(figsize=(8,8))
plt.plot(range(0,len(Y)),np.sort(YBC))
plt.show()

# In[6]:

print("Feature selection...")
colsToRemoveTest  = []
colsToRemoveTrain = []
for col in data.columns:
  if col != 'ID' and col != 'target':
    if data[col].std() ==0:
      colsToRemoveTrain.append(col)
    if testdata[col].std() ==0:
      colsToRemoveTest.append(col)

print("colsToRemoveTrain ==0 {} {}".format(len(colsToRemoveTrain), colsToRemoveTrain))
print("colsToRemoveTest ==0 {} {}".format(len(colsToRemoveTest), colsToRemoveTest))
# remove constant columns in the test set0

# In[6]:

testdata=testdata.drop(colsToRemoveTest, axis=1, inplace=False)
data=data.drop(colsToRemoveTest, axis=1, inplace=False)

# In[6]:

train_without_duplicates = data.T.drop_duplicates().T
columns_not_to_be_dropped = train_without_duplicates.columns
columns_to_be_dropped = [col for col in testdata.columns if col not in columns_not_to_be_dropped]
print(columns_to_be_dropped )
print(len(columns_to_be_dropped ))
testdata = testdata.drop(columns_to_be_dropped, 1)
data = data.drop(columns_to_be_dropped, 1)


# In[87]:

print("Feature scaling...")
fulldata = data.append(testdata)
rscaler = preprocessing.StandardScaler()

dataLogScaled = np.log1p(data)
testdataLogScaled  = np.log1p(testdata)
fulldataLogScaled = np.log1p(fulldata)

rscaler.fit( fulldata )
dataScaled = rscaler.transform( dataLogScaled )
testdataScaled = rscaler.transform( testdataLogScaled   )
fulldataScaled = rscaler.transform( fulldataLogScaled )

dataScaledLog = (dataScaled)
testdataScaledLog = (testdataScaled)
fulldataScaledLog = (fulldataScaled)
print("Done...")

# In[8]:

print(fulldataScaledLog[0:10])

# In[8]:
def StratifiedSample(data, MY,  test_size, random_state):
  ylen = MY.shape[0]
  print("Ylen = {}".format(ylen))
  nbins = np.unique(MY).shape[0]
  print("bins = {}".format(nbins))
  bins = np.linspace(0, ylen, nbins)

  y_binned = np.digitize(MY, bins)
  MY=np.log1p(MY)
  X_data, X_test, Y_data, y_test = train_test_split(data, MY,
                                                    stratify=y_binned,
                                                    test_size=test_size, random_state=random_state)
  print("X_data = {}".format(X_data.shape))
  print("X_test = {}".format(X_test.shape))
  return  X_data, X_test, Y_data, y_test

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def remove_features_using_importance(x_train, y_train, x_test, num_of_features = 1000):
    def rmsle(actual, predicted):
        return np.sqrt(np.mean(np.power(np.log1p(actual)-np.log1p(predicted), 2)))

    print("Split train and test")
    x1, x2, y1, y2 = StratifiedSample(data=x_train, MY=y_train, test_size = 0.20, random_state = 42)
    model = RandomForestRegressor(n_jobs = -1, random_state = 7)
    model.fit(x1, y1)
    print(rmsle(np.expm1(y2), np.expm1(model.predict(x2))))

    print("Get columns by feature importances")
    col_df = pd.DataFrame({'importance': model.feature_importances_, 'feature': x_train.columns})
    col_df_sorted = col_df.sort_values(by = ['importance'], ascending = [False])
    columns = col_df_sorted[:num_of_features]['feature'].values

    x_train = x_train[columns]
    x_test = x_test[columns]

    return x_train, x_test

num_of_features = int(data.shape[1] * 0.79  )+1
x_trainTOPRF, x_testTOPRF = remove_features_using_importance(x_train=data, y_train=YBC, x_test=testdata , num_of_features = num_of_features )

print("x_trainTOPRF {}".format(x_trainTOPRF.shape) )
print("x_testTOPRF {}".format(x_testTOPRF.shape) )


# In[3]:


# In[101]:

def generate_adversarial_validation_set(train, test , topp=0.75):
    x_test = test.drop(["is_test", "target"], 1)
    tratio = int(topp * train.shape[0])+1
    train, val = train.iloc[tratio :], train.iloc[:train.shape[0]-tratio ]
    train = train.drop(["is_test", "predicted_probs"], 1)
    val = val.drop(["is_test", "predicted_probs"], 1)
    x_train, y_train = train.drop("target", 1), train.target
    x_val, y_val = val.drop("target", 1), val.target

    return x_train, y_train, x_val, y_val, x_test

def get_training_set_with_test_set_similarity_predictions(X_train, Y_train, X_test):

    print("Add target column")
    X_train['target'] = Y_train
    X_test['target'] = 0

    X_train["is_test"] = 0
    X_test["is_test"] = 1
    assert(np.all(data.columns == testdata.columns))

    print("Concat train and test data")
    total = pd.concat([X_train, X_test])
    total = total.fillna(0)

    x = total.drop(["is_test", "target"], axis = 1)
    y = total.is_test

    print("Start cross-validating")
    n_estimators = 100
    classifier = RandomForestClassifier(n_estimators = n_estimators, n_jobs = -2)
    predictions = np.zeros(y.shape)

    stratified_kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    bauc= None
    bpredicted_probabilities=None
    for fold_index, (train_indices, test_indices) in enumerate(stratified_kfold.split(x, y)):
        print("Fold - " + str(fold_index))

        x_train = x.iloc[train_indices]
        y_train = y.iloc[train_indices]
        x_test = x.iloc[test_indices]
        y_test = y.iloc[test_indices]

        classifier.fit(x_train, y_train)
        
        predicted_probabilities = classifier.predict_proba(x_test)[:, 1]

        auc = roc_auc_score(y_test, predicted_probabilities)
        print("AUC Score - " + str(auc) + "%")
        if bauc is None:
            bauc=auc
            bpredicted_probabilities=predicted_probabilities
        elif auc> bauc:
            bac=auc
            bpredicted_probabilities=predicted_probabilities
            
        predictions[test_indices] = bpredicted_probabilities
    total['predicted_probs'] = predictions

    print("Generating training set")
    total = total[total.is_test == 0]

    print("Sorting according to predictions")
    train_set_with_predictions_for_test_set_similarity = total.sort_values(["predicted_probs"], ascending = False)

    return train_set_with_predictions_for_test_set_similarity, X_test

#x_trainTOPRF, x_testTOPRF
train_set_with_predictions_for_test_set_similarity, X_test =  get_training_set_with_test_set_similarity_predictions(X_train=x_trainTOPRF.copy(),
                                                                                                                    Y_train=YBC,
                                                                                                                    X_test=x_testTOPRF.copy())
x_train, y_train, x_val, y_val, x_test = generate_adversarial_validation_set(train=train_set_with_predictions_for_test_set_similarity, test=X_test )

# In[101]:

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.grid_search import GridSearchCV

def use_decomposed_features_as_new_df(train,
                                      test,
                                      total,
                                      n_components,
                                      use_pca = False,
                                      use_tsvd = False,
                                      use_ica = False,
                                      use_fa = False,
                                      use_grp = False,
                                      use_srp = False):
    N_COMP = n_components
    ntrain = len(train)

    print("\nStart decomposition process...")

    if use_pca:
        print("PCA")
        pca = PCA(n_components = N_COMP, random_state = 42)
        pca_results = pca.fit_transform(total)
        pca_results_train = pca_results[:ntrain]
        pca_results_test = pca_results[ntrain:]

    if use_tsvd:
        print("tSVD")
        tsvd = TruncatedSVD(n_components = N_COMP, random_state=42)
        tsvd_results = tsvd.fit_transform(total)
        tsvd_results_train = tsvd_results[:ntrain]
        tsvd_results_test = tsvd_results[ntrain:]

    if use_ica:
        print("ICA")
        ica = FastICA(n_components = N_COMP, random_state=42)
        ica_results = ica.fit_transform(total)
        ica_results_train = ica_results[:ntrain]
        ica_results_test = ica_results[ntrain:]

    if use_fa:
        print("FA")
        fa = FactorAnalysis(n_components = N_COMP, random_state=42)
        fa_results = fa.fit_transform(total)
        fa_results_train = fa_results[:ntrain]
        fa_results_test = fa_results[ntrain:]

    if use_grp:
        print("GRP")
        grp = GaussianRandomProjection(n_components = N_COMP, eps=0.1, random_state=42)
        grp_results = grp.fit_transform(total)
        grp_results_train = grp_results[:ntrain]
        grp_results_test = grp_results[ntrain:]

    if use_srp:
        print("SRP")
        srp = SparseRandomProjection(n_components = N_COMP, dense_output=True, random_state=42)
        srp_results = srp.fit_transform(total)
        srp_results_train = srp_results[:ntrain]
        srp_results_test = srp_results[ntrain:]

    print("Append decomposition components together...")
    train_decomposed = np.concatenate([srp_results_train, grp_results_train, ica_results_train, pca_results_train, tsvd_results_train], axis=1)
    test_decomposed = np.concatenate([srp_results_test, grp_results_test, ica_results_test, pca_results_test, tsvd_results_test], axis=1)

    train_with_only_decomposed_features = pd.DataFrame(train_decomposed)
    test_with_only_decomposed_features = pd.DataFrame(test_decomposed)

    for agg_col in ['sum', 'var', 'mean', 'median', 'std', 'weight_count', 'count_non_0', 'num_different', 'max', 'min']:
        train_with_only_decomposed_features[col] = train[col]
        test_with_only_decomposed_features[col] = test[col]

    # Remove any NA
    train_with_only_decomposed_features = train_with_only_decomposed_features.fillna(0)
    test_with_only_decomposed_features = test_with_only_decomposed_features.fillna(0)

    return train_with_only_decomposed_features, test_with_only_decomposed_features
print("train pca...")
pca = decomposition.PCA(n_components=0.96,
                        copy=True, whiten=True, svd_solver='full',
                        tol=0.0 )

"""
pca = pca.fit(fulldataScaledLog)

fulldataScaledPCA = pca.transform(fulldataScaledLog)
print("pca transform train... ")
dataPCA = pca.transform(dataScaledLog)
print("Test Data pca... ")
testdataPCA = pca.transform(testdataScaledLog)
"""
dataPCA =x_train
testdataPCA =x_test
Y_data=y_train

# In[10]:
YBC=Y



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


# In[4]:


from catboost import CatBoostRegressor
catgbmx1=CatBoostRegressor(iterations=1550,
                           depth=10,
                           border_count=126,
                           learning_rate=0.021, loss_function='RMSE',
                           thread_count=6)
catgbmx1.fit(x_train, y_train, eval_set=(x_val, y_val),plot=False, use_best_model=True)


# In[5]:


y_catgbmx1 = catgbmx1.predict(x_val)
trainrms = sqrt(mean_squared_error( np.expm1(y_val), np.expm1(y_catgbmx1)))
print("catgbmx1 trainrms {}".format(  trainrms ) )


# In[4]:


# In[6]:


# In[82]:


lgbm1 = lgb.LGBMRegressor(objective='regression',
                        num_leaves=31,
                        min_data_in_leaf=2,
                        learning_rate= 0.02,
                        feature_fraction= 0.9,
                        bagging_fraction= 0.8,
                        bagging_freq= 4,
                        verbose= 0,
                        num_threads=4,
                        n_estimators=1811)

lgbm1.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='l1',  early_stopping_rounds=511)


# In[7]:


# In[ ]:

y_lgbmx1 = lgbm1.predict(x_val)
trainrms = sqrt(mean_squared_error( np.expm1(y_val ), np.expm1(y_lgbmx1 ) ))
print("y_lgbmx1  trainrms {}".format(  trainrms ) )


# In[6]:


# In[8]:


# In[89]:
print("Predict  lgbm1...")
ytest_lgbm1= np.expm1(lgbm1.predict(x_test ))

# In[89]:
print("Predict  catgbmx1...")
ytest_catgbmx1= np.expm1(catgbmx1.predict(x_test ))


# In[ ]:

plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_lgbm1)), ytest_lgbm1, s=100,  marker="s", label='ytest_lgbm1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_lgbm1 ', fontsize=12)
plt.title("ytest_lgbm1 Distribution", fontsize=14)
plt.show()
plt.figure(figsize=(11,11))
plt.scatter(range(0,len(ytest_catgbmx1)), ytest_catgbmx1, s=100,  marker="s", label='ytest_lgbmClass1')
plt.xlabel('index', fontsize=12)
plt.ylabel('ytest_catgbmx1 ', fontsize=12)
plt.title("ytest_catgbmx1 Distribution", fontsize=14)
plt.show()


plt.figure(figsize=(11,11))
plt.scatter( ytest_lgbm1 , ytest_catgbmx1, s=100,  marker="s", label='lgbm1_RFT ' )
plt.xlabel('ytest_lgbm1', fontsize=12)
plt.ylabel('ytest_catgbmx1 ', fontsize=12)
plt.title("ytest_lgytest_catgbmx1 bmClass1 lgbm1 Distribution", fontsize=14)
plt.show()


# In[91]:


# In[10]:


ypredavg = pd.DataFrame(dict(
    XCATGB1=ytest_catgbmx1,
    LGB1=ytest_lgbm1,
    AVG= (ytest_lgbm1 + ytest_catgbmx1 )/2  ) )

print(ypredavg.columns)
ypredavg.to_csv(filepath + "/tmp/FinalFinal.csv", index=False)
print(ypredavg.mean())
print(ypredavg.std())


# In[ ]:


# In[93]:
ySubmit = pd.DataFrame(dict( ID=testID,target=(ypredavg["AVG"] )))
ySubmit.to_csv(filepath + "/tmp/submit.csv", index=False)
print(ySubmit.shape)

