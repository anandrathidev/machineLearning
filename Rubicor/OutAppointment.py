
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[2]:


import logging
import numpy as np
import pandas as pd
#from optparse import OptionParser
import os
from time import time
import matplotlib.pyplot as plt

#from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
# Display progress logs on stdout
    


# In[3]:


xpath = "/home/he159490/DS/OUT/"
AppointCSV = "AppointmentsJanToJuly.csv"


# In[4]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelBinarizer


# In[5]:


def multiclass_roc_auc_score(truth, pred, metricstype, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)

    truth = lb.transform(truth)
    pred = lb.transform(pred)
    if metricstype==None:
      metricstype=roc_auc_score
    return metricstype(truth, pred, average=average)

def print_evaluation_scores_multi_class(y_val, predicted):
    print( "accuracy={}".format( accuracy_score(y_val, predicted)))
    print( "")
    print( "roc_auc_score={}".format( multiclass_roc_auc_score(y_val, predicted, metricstype=None, average = "weighted")))
    print( "")
    print( "average_precision_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=average_precision_score, average="macro") ))
    print( "")

    print( "macro average_precision_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=average_precision_score, average="macro")))
    print( "micro average_precision_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=average_precision_score, average="micro")))
    print( "weighted average_precision_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=average_precision_score, average="weighted")))

    print( "")

    print( "macro recall_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=recall_score, average = "macro")))
    print( "micro recall_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=recall_score, average= "micro")))
    print( "weighted recall_score={}".format( multiclass_roc_auc_score(y_val, predicted,  metricstype=recall_score, average = "weighted")))
    print( "")

    print( "macro f1_score={}".format( f1_score(y_val, predicted, average = "macro")))
    print( "micro f1_score={}".format( f1_score(y_val, predicted, average = "micro")))
    print( "weighted f1_score={}".format( f1_score(y_val, predicted, average = "weighted")))


def print_evaluation_scores(y_val, predicted):
    pos_label=list(set(y_val))
    print( "accuracy={}".format( accuracy_score(y_val, predicted)))
    print( "")
    #print( "roc_auc_score={}".format( roc_auc_score(y_val, predicted)))
    print( "")
    print( "average_precision_score={}".format(  metrics.f1_score(y_test, predicted, pos_label=list(set(y_test)))  ))
    print( "")

    print( "macro average_precision_score={}".format( average_precision_score(y_val, predicted, average = "macro")))
    print( "micro average_precision_score={}".format( average_precision_score(y_val, predicted, average = "micro")))

    print( "")

    print( "macro recall_score={}".format( recall_score(y_val, predicted, average = "macro")))
    print( "micro recall_score={}".format( recall_score(y_val, predicted, average = "micro")))
    print( "")

    print( "macro f1_score={}".format( f1_score(y_val, predicted, average = "macro")))
    print( "micro f1_score={}".format( f1_score(y_val, predicted, average = "micro")))


    #print( "f1_score={}".format( f1_score(y_val, predicted)))


# In[6]:


print(xpath + AppointCSV)


# In[7]:


AppointdfO = pd.read_csv(xpath + AppointCSV)
#AppointdfO.head(2)


# In[8]:


AppointdfO.shape


# In[9]:


print(AppointdfO['Hosp Code'].value_counts())
FS  = AppointdfO['Hosp Code']=='FS' 
FH = AppointdfO['Hosp Code']=='FH' 
RPH = AppointdfO['Hosp Code']=='RPH' 
print(AppointdfO['Hosp Code'][FS | FH])


# In[10]:


#AppointdfOFSFH = AppointdfO[FH | FS]
#AppointdfOFSFH = AppointdfO[FH | FS]
#AppointdfOFSFH = AppointdfO[RPH]
AppointdfOFSFH = AppointdfO
print(AppointdfOFSFH['Hosp Code'].value_counts())


# In[11]:


print(AppointdfOFSFH['Date Of Appointment'].min())
print(AppointdfOFSFH['Date Of Appointment'].max())


# In[12]:


AppointdfOFSFH['Date Of Appointment'] = pd.to_datetime(AppointdfOFSFH['Date Of Appointment'])
print(AppointdfOFSFH['Date Of Appointment'].min())
print(AppointdfOFSFH['Date Of Appointment'].max())
from dateutil.relativedelta import relativedelta


# In[13]:


from dateutil import parser
dtJul = parser.parse("28 July 2018")
dtJan = parser.parse("01 Jan 2018")
print(dtJul)
print(dtJan)


# In[14]:


AppointdfOFSFH=AppointdfOFSFH.sort_values('Date Of Appointment').reset_index()
print(AppointdfOFSFH['Date Of Appointment'].min())
print(AppointdfOFSFH['Date Of Appointment'].max())
from dateutil.relativedelta import relativedelta
AppointdfOFSFH = AppointdfOFSFH.loc[AppointdfOFSFH['Date Of Appointment'].between(dtJan, dtJul)]
print(AppointdfOFSFH['Date Of Appointment'].min())
print(AppointdfOFSFH['Date Of Appointment'].max())


# In[15]:


print(AppointdfOFSFH.columns)


# In[16]:


print(AppointdfOFSFH.shape[0])


# In[17]:


start = AppointdfOFSFH.shape[0]-10000
end = start+5
print("start {} end {}".format(start,end))
AppointdfOFSFH.loc[start:end]


# In[18]:


print(list(AppointdfOFSFH.columns))


# In[19]:


AppointdfOFSFH.tail()


# In[20]:


print(AppointdfOFSFH[AppointdfOFSFH['Source Appointment Reason'] == '[No Value]'].count())
print(AppointdfOFSFH['Source Appointment Reason'].value_counts())


# In[21]:


mask = AppointdfOFSFH['Source Appointment Reason'] == '[No Value]'
AppointdfOFSFH['Source Appointment Reason'].loc[mask]  = 'NoVal'


# In[22]:


print(AppointdfOFSFH['Source Appointment Reason'].value_counts())


# In[29]:


def prepareData(Appointdf):
    print("Subset data")
    Appointdf= Appointdf[[
     'Hosp Code',
     'Postcode',  
     'Appointment Sequence',
     'Date Of Appointment',
     'Date Of Appointment Booked',
     'Clinic Category Code',
     'Time of Slot',
     'Indigenous Status at Apppointment',
     'Account Type',
     'Is Interpreter Required',
     'Age',
     'DayOfWeekShort',
     #'MonthNameShort',
     'Gender',
     #'PriorAttendanceRate',
     'Source Appointment Reason',
     'Patient Attended', ## Y 
     ]]

    print("")
    print("diff Appoint_Book_Length...")
    print("............................................")
    Appointdf["Date Of Appointment"] =  pd.to_datetime(Appointdf["Date Of Appointment"])
    Appointdf['Date Of Appointment Booked'] =  pd.to_datetime(Appointdf['Date Of Appointment Booked'])
    Appointdf["Appoint_Book_Length"] = Appointdf["Date Of Appointment"] -  Appointdf['Date Of Appointment Booked']
    Appointdf["Appoint_Book_Length"] = Appointdf["Appoint_Book_Length"].dt.total_seconds()
    Appointdf = Appointdf.drop(columns=["Date Of Appointment", 'Date Of Appointment Booked', ])

    print("")
    print("............................................")

    #print("drop Source Appointment Status== [No Value]")
    #Appointdf['Count of Patient Attended Appointments'].unique()
    print(list(Appointdf.columns))
    
    #Appointdf['Appointment Reason'].unique()
    #Appointdf.groupby(by=['Appointment Reason']).count()
    #Appointdf['Source Appointment Reason'].unique()
    #Appointdf.groupby(by=['Source Appointment Reason']).count()
    
    print("")
    print("Convert Time of Slot in Hour of day...")
    print("............................................")
    Appointdf['Time of Slot'] =   pd.to_datetime(Appointdf['Time of Slot'] )
    Appointdf['Time of Slot'] .head()
    Appointdf['Time of Slot']  = Appointdf['Time of Slot'].dt.hour


    print("")
    print("Writing to Appointdf_Clean.csv")
    print("                                   ...")
    #Appointdf.to_csv(xpath + "/Appointdf_Clean.csv")
    
    ###################################################################################################### 
    ################################## missing vals treatment        ####################################
    ###################################################################################################### 
    
    Appointdf.isnull().sum()     
    #Appointdf['Duration Of Appointment'].fillna(Appointdf['Duration of Appointment'].mean(), inplace=True) 
    
    Appointdf.isnull().sum()     

    #Appointdf['Appoint_Book_Length'].median()
    #temp = Appointdf['Appoint_Book_Length']
    
    print("")
    print("fill na Appoint_Book_Length  .........")
    print("............................................")
    #Appointdf['Appoint_Book_Length'].fillna(Appointdf['Appoint_Book_Length'].mean(), inplace=True) 
    Appointdf['Appoint_Book_Length'] = Appointdf['Appoint_Book_Length'].replace(np.nan, Appointdf['Appoint_Book_Length'].mean())

    if 'Postcode' in Appointdf.columns:
        novalmask = Appointdf.Postcode.str.contains('No Value', regex=False, na=False)
        Appointdf.loc[novalmask, 'Postcode'] = 1
        Appointdf.Postcode = Appointdf.Postcode.astype('int')

    print("............................................")
    print("............................................")
    print("Null count: {}".format( Appointdf.columns[Appointdf.isna().any()].tolist() ))
    print("............................................")
    print(Appointdf.isna().any())
    Appointdf['Appointment Sequence']=Appointdf['Appointment Sequence'].fillna(Appointdf['Appointment Sequence'].mean())
    if 'PriorAttendanceRate' in Appointdf.columns:
        Appointdf['PriorAttendanceRate']=Appointdf['PriorAttendanceRate'].fillna(Appointdf['PriorAttendanceRate'].mean())
    # Look for NA values 
    #Appointdf=Appointdf.dropna()
    for i, v in Appointdf.isna().any().items():
        if v:
            print('index: ', i, 'value: ', v)    
    Y = Appointdf.pop('Patient Attended')
    print(Appointdf.isna().any())
    return Appointdf,Y


# In[ ]:


AppointdfOFSFH=AppointdfOFSFH.dropna()
Appointdf,Y= prepareData(Appointdf=AppointdfOFSFH)
print(list(Appointdf.columns))
Appointdf.head(1).to_csv(xpath + "/AppointCSV.csv")


# In[ ]:



def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

def fix_columns( d, columns ):  

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print( "extra columns:", extra_cols)

    d = d[ columns ]
    return d
        
def Dummify(Appointdf, cat_vars):

    for var in cat_vars:   
        print("Dummyfing data:" + var)
        print("............................................")
        cat_list='var'+'_'+var
        u_var = list(set(Appointdf[var].unique()))
        print("var {} labels {}".format(var,u_var) )
        cat_list = pd.get_dummies(Appointdf[var], prefix=var, columns=u_var, dummy_na=True, sparse=False)
        cat_list.columns = [c.strip().replace("/","_").replace("(","_").replace(")","_").replace(" ","_").replace("[","_").replace("]","_").replace(".","_").replace(r"\\t","_").replace(r"\\n","_") for c in cat_list.columns.values.tolist()]
        data1=Appointdf.join(cat_list)
        print("data1 shape {} var {}".format(data1.shape,var))
        print("Appointdf shape {} var{}".format(Appointdf.shape,var))
        del cat_list
        Appointdf=data1
        del data1
   
    print("drop dummified columns")
    print("............................................")
    print("Appointdf shape {}".format(Appointdf.shape))
    Appointdf.drop(columns=cat_vars, inplace =True )
    print("droped dummified columns")
    print("Appointdf shape {}".format(Appointdf.shape))
    print((Appointdf.dtypes))
    tylist = [i for i in Appointdf.dtypes]
    for itylist in tylist:
      #print(str(itylist))
      if itylist == 'object':
         print(itylist)
    Appointdf.columns = [c.strip().replace("/","_").replace("(","_").replace(")","_").replace(" ","_").replace("[","_").replace("]","_").replace(".","_").replace(r"\\t","_").replace(r"\\n","_") for c in Appointdf.columns.values.tolist()]
    
    return Appointdf

cat_vars=['Hosp Code',
          'Postcode',  
          #'Health Service Code',
          'Clinic Category Code',
          'Indigenous Status at Apppointment',
          'Account Type',
          'DayOfWeekShort',
          #'MonthNameShort',
          'Gender', 
          'Source Appointment Reason']


# In[ ]:


print("Dummyfing data")
print("............................................")
AppointdfNonDummy = Appointdf



# In[ ]:


print(AppointdfNonDummy.dtypes)
print(AppointdfNonDummy.Postcode.unique())
print(AppointdfNonDummy.dtypes)
print(AppointdfNonDummy.Postcode.unique())


# In[ ]:


print(len((AppointdfNonDummy.columns)))
print(len(set(AppointdfNonDummy.columns)))

Appointdf = Dummify(Appointdf=AppointdfNonDummy, cat_vars=cat_vars)
print(len((Appointdf.columns)))
print(len(set(Appointdf.columns)))


# In[ ]:


print(sum(Appointdf.isnull().sum()))
print(Appointdf.head())


# In[ ]:


print(len((Appointdf.columns)))
print(len(set(Appointdf.columns)))


# In[ ]:


#for c in sorted(Appointdf.columns): 
#    print(c)
    
#print((set(Appointdf.columns)))


# In[ ]:


#for c in sorted(set(Appointdf.columns)): 
#    print(c)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Appointdf[Appointdf.columns] = scaler.fit_transform(Appointdf)

X_train, X_test, y_train, y_test = train_test_split( Appointdf,  Y,  test_size=0.30,  random_state=61) 


# In[ ]:


print("X_train: : {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Null: {}".format(Appointdf.isnull().sum()))


# In[ ]:


#for c in X_train.columns:
#   print(c)


# In[ ]:



clf = None
clf = RandomForestClassifier(min_samples_split=30, min_samples_leaf=3,n_estimators=350, n_jobs=12)

#del clf
#clf = RandomForestClassifier(n_jobs=2, n_estimators=7,class_weight ='balanced')
print("............................................")
print("............................................")
print("Train Random forest")
print("............................................")
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
scoring = {'roc_auc': 'roc_auc',
            'rec_micro': make_scorer(recall_score, average=None)}

scoring = ('roc_auc', 'recall')

scores = cross_validate(clf, Appointdf,  Y, scoring=None, cv=5, return_train_score=False)
clf.fit(Appointdf,  Y)
print(scores)    


# In[ ]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
gp_opt = GaussianProcessClassifier(kernel=100.0 * RBF())

#scores = cross_validate(gp_opt, Appointdf,  Y, scoring=None, cv=1, return_train_score=False)
#gp_opt.fit(Appointdf, Y)
#print(scores)    


# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score

logreg = linear_model.LogisticRegression(C=2, penalty='l2')
scoring = ('roc_auc', 'recall')

scores = cross_validate(logreg, Appointdf, Y, scoring=None, cv=2, return_train_score=False)
logreg.fit(Appointdf,  Y)
print(scores)    


# In[ ]:


from lightgbm import LGBMClassifier
clsb = LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.051,n_estimators=151)
#print('Feature importances:', list(gbm.feature_importances_))
print("............................................")
print("............................................")
print("Train LGBMClassifier ")
print("............................................")

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
scoring = {'prec_micro': 'precision_micro',
            'rec_micro': make_scorer(recall_score, average='micro')}
scoring = ('roc_auc', 'recall')
scores = cross_validate(clsb, Appointdf,  Y, scoring=scoring, cv=2, return_train_score=False)
clsb.fit(Appointdf,  Y)
print(scores)    


# In[ ]:


def FeatureImportance(forest,X, fsize=None):
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_],
               axis=0)
  indices = np.argsort(importances)[::-1]
  #indices = indices[:10]
  
  # Print the feature ranking
  print("Feature ranking:")
  # Plot the feature importances of the forest
  plt.figure()
  plt.title("Feature importances")
  if fsize==None:
    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]]  , importances[indices[f]]))
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    #plt.xticks(range(X.shape[1]), indices)
    plt.xticks(range(X.shape[1]), X.columns[indices])
    plt.xlim([-1, X.shape[1]])
  else:
    for f in range(fsize):
        print("%d. feature %s (%f)" % (f + 1, X.columns[indices[f]]  , importances[indices[f]]))
    plt.bar(range(fsize), importances[indices[:fsize]],
           color="r", yerr=std[indices[:fsize]], align="center")
    plt.xticks(range(fsize), X.columns[indices[:fsize]] , rotation=80)
    plt.xlim([-1, fsize])
  plt.show()

FeatureImportance(forest=clf, X=X_train, fsize=150)

clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
               axis=0)
indices = np.argsort(clf.feature_importances_)[::-1]
X_train.shape[1] 
X_train.columns[indices[:10]]


plt.bar(range(10), clf.feature_importances_[indices[:10]],
         color="r", yerr=std[indices[:10]], align="center")


# In[ ]:


# Predict 
predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)
#predxgs = clsb.predict(X_test)
#predxgs_prob = clsb.predict_proba(X_test)
print("evaluate ....")
print_evaluation_scores_multi_class(y_val=y_test, predicted=predictions)
#print_evaluation_scores_multi_class(y_val=y_test, predicted=predxgs)
print("confusion matrix ....")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confMatrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True)
#confMatrixXB = pd.crosstab(y_test, predxgs, rownames=['True'], colnames=['Predicted'], margins=True)

print("confMatrix {}".format(confMatrix))
#print("confMatrixXB {}".format( confMatrixXB))

#confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
#print("confMatrixPercent {}".format( confMatrixPercent))


# In[ ]:


print(X_train.shape)
print(X_test.shape)


# In[ ]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(y_test, predictions, labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))


# In[ ]:


#testData = pd.read_csv(xpath + AppointCSV)
#print(testData.shape)


# In[ ]:


#testData.head(2)


# In[ ]:


#testData[testData["class_0"].notnull()].head(2) 


# In[ ]:


#testData = testData[testData["class_0"].notnull()] 
#print("test Data size {}".format(testData.shape))


# In[ ]:


#testDatap,Yt= prepareData(Appointdf=testData)


# In[ ]:


#testDataDummy = Dummify(Appointdf=testDatap, cat_vars=cat_vars)


# In[ ]:


#print(testDataDummy.shape )
#print(X_train.shape)


# In[ ]:


XtrainEmpty = X_train[0:0]


# In[ ]:


#print(XtrainEmpty.shape)
#print(XtrainEmpty.dtypes)


# In[ ]:



#resultdf = pd.concat([XtrainEmpty, testDataDummy], sort=False)
#resultdf = testDataDummy[list(XtrainEmpty.columns)]


# In[ ]:


#print(resultdf.shape)


# In[ ]:


#print(resultdf.columns[resultdf.isna().any()].tolist())
#NullCols = resultdf.columns[resultdf.isna().any()].tolist()
#for c in NullCols:
#    resultdf[c]=0
    


# In[ ]:


#print(resultdf.columns[resultdf.isna().any()].tolist())


# In[ ]:


## Predict 
#print("Predict ....")
#predictions_2weeks = clf.predict(resultdf)
#predictions_prob_2weeks = clf.predict_proba(resultdf)
#predxgs_2weeks = clsb.predict(resultdf)
#predxgs_prob_2weeks = clsb.predict_proba(resultdf)


# In[ ]:


#print("evaluate ....")
#print_evaluation_scores_multi_class(y_val=Yt, predicted=predictions_2weeks)
#print_evaluation_scores_multi_class(y_val=Yt, predicted=predxgs_2weeks)


# In[ ]:


#print("confusion matrix ....")
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#confMatrix = pd.crosstab(Yt, predictions_2weeks, rownames=['True'], colnames=['Predicted'], margins=True)
#confMatrixXB = pd.crosstab(Yt, predxgs_2weeks, rownames=['True'], colnames=['Predicted'], margins=True)

#print(confMatrix)
#print(confMatrixXB)

#confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
#print(confMatrixPercent)


# In[ ]:


#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#unique_label = np.unique(y_test)
#print(pd.DataFrame(confusion_matrix(Yt, predictions_2weeks, labels=unique_label), 
#                   index=['true:{:}'.format(x) for x in unique_label], 
#                   columns=['pred:{:}'.format(x) for x in unique_label]))


# In[ ]:


#predictionsDF = pd.DataFrame(predictions_prob_2weeks, columns=["WillMiss", "WillNotMiss"])
#print(predictionsDF.head(1))
#print(predictionsDF.shape)


# In[ ]:


#resultDF_PredProb = predictionsDF.join(testData)


# In[ ]:


#list(resultDF_PredProb.columns)
#list(resultdf.columns)


# In[ ]:


#  resultDF = resultDF_PredProb[[
#     'class_0',
#     'class_1',
#     'WillMiss',
#     'WillNotMiss',
#     'Hosp Code',
#     'Appointment ID',
#     'Account Number']]
    


# In[ ]:


futureDataO = pd.read_csv(xpath + "/AllAppointments.csv")
#print(futureDataO['Hosp Code'].value_counts())
FS  = futureDataO['Hosp Code']=='FS' 
FH = futureDataO['Hosp Code']=='FH' 
RPH = futureDataO['Hosp Code']=='RPH' 
#print(futureDataO['Hosp Code'][FS | FH])
#futureData = futureDataO[RPH]
futureData = futureDataO
#futureData=futureData.dropna()
print(futureData.shape)
print(futureData.head(2))


# In[ ]:


import datetime
futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
print("test Data size {}".format(futureData.shape))
futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
futureData = futureData[futureData['Date Of Appointment'] >= datetime.date(year=2018,month=7,day=3) ]
print("test after date  Data size {}".format(futureData.shape))
FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]


# In[ ]:


FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
print("FutureIDDF size {}".format(FutureIDDF.shape))
print("FutureIDDF cols {}".format(FutureIDDF.columns))


# In[ ]:


futureDatap,Yf= prepareData(Appointdf=futureData)
futureDatapDummy = Dummify(Appointdf=futureDatap, cat_vars=cat_vars)
print("futureDatapDummy {}".format(futureDatapDummy.shape))


# In[ ]:


XtrainEmpty = X_train[0:0]

print(XtrainEmpty.shape)
print(len(XtrainEmpty.columns))
print(len(set(XtrainEmpty.columns)))
print((XtrainEmpty.columns))


# In[ ]:


print("concat columns futureDatapDummy {}".format(futureDatapDummy.shape))
frames=[XtrainEmpty, futureDatapDummy]
XtrainEmpty_cols = set(XtrainEmpty.columns) 
print("XtrainEmpty_cols {}" .format(len(XtrainEmpty_cols)))
futureDatapDummy_cols = set(futureDatapDummy.columns)
comm_cols = XtrainEmpty_cols.intersection(futureDatapDummy_cols)
print(len(comm_cols))


# In[ ]:


#futureDatapDummy = pd.concat([df[common_cols] for df in frames], ignore_index=True, sort=False)
print(futureDatapDummy.shape)
futureDatapDummy = futureDatapDummy[list(comm_cols)]
futureDatapDummy = pd.concat([XtrainEmpty,futureDatapDummy])
print(futureDatapDummy.shape)


# In[ ]:


print(len(XtrainEmpty_cols))
print(len(comm_cols))
print(len(set(XtrainEmpty_cols - comm_cols)))

for ac in list(set(XtrainEmpty_cols - comm_cols)):
    futureDatapDummy[ac]=0
print(futureDatapDummy.shape)


# In[ ]:


print("after concat futureDatapDummy {}".format(futureDatapDummy.shape))
print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
NullCols = futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist()
for c in NullCols:
    futureDatapDummy[c]=0
print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
futureDatapDummy.head(1)


# In[ ]:


predictions_Future = clf.predict(futureDatapDummy)
predictions_prob_Future = clf.predict_proba(futureDatapDummy)
predxgs_Future = clsb.predict(futureDatapDummy)
predxgs_prob_Future = clsb.predict_proba(futureDatapDummy)
print(clf.classes_)


# In[ ]:



predictions_Future = logreg.predict(futureDatapDummy)
predictions_prob_Future = logreg.predict_proba(futureDatapDummy)
predxgs_Future = logreg.predict(futureDatapDummy)
predxgs_prob_Future = logreg.predict_proba(futureDatapDummy)

print(logreg.classes_)


# In[ ]:



print(predxgs_Future)
print(predictions_prob_Future)

print(predxgs_Future.shape)
print(FutureIDDF.shape)


# In[ ]:


predictionsFutureDF = pd.DataFrame(predictions_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
predictionsFutureDFB = pd.DataFrame(predxgs_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
print(FutureIDDF.shape)
print(predictions_Future.shape)

FutureIDDF = FutureIDDF.assign(Attend=predictions_Future)  
FutureIDDF = FutureIDDF.assign(AttendB=predxgs_Future)  
FutureIDDF = FutureIDDF.assign(MissProb=predictionsFutureDF["WillMissProb"].values)
FutureIDDF = FutureIDDF.assign(MissProbB=predictionsFutureDFB["WillMissProb"].values)


# In[ ]:


#FutureDF_PredProb = predictionsFutureDF.join(FutureIDDF)
print(FutureIDDF.shape)
print(FutureIDDF.columns)
print(predictionsFutureDF.shape)
print(predictionsFutureDFB.shape)


# In[ ]:


FutureIDDF.to_csv(xpath + "/PredictionJuly2018.csv",index=True, header=True, na_rep="NA")
predictionsFutureDF.to_csv(xpath + "/ProbsJuly2018.csv",index=True, header=True, na_rep="NA")
predictionsFutureDFB.to_csv(xpath + "/ProbsJuly2018B.csv",index=True, header=True, na_rep="NA")


# In[ ]:


print(FutureIDDF.isnull().sum())
print(FutureIDDF.isna().sum())
print(FutureIDDF[FutureIDDF['Date Of Appointment']==pd.Timestamp('')].shape)


# In[ ]:


FutureIDDF.head(5)


# In[ ]:


print(FutureIDDF.dtypes)


# In[ ]:


outcomerph = pd.read_csv(xpath + "OutcomeRPHJuly2018.csv")


# In[ ]:


outcomerph.columns
outcomerph=outcomerph.rename(columns = {'(No column name)':'SourceStatus'})


# In[ ]:


pred_out = pd.merge(futureData, outcomerph, left_on=['Appointment ID'], right_on=['AppointmentID'] )


# In[ ]:


pred_outResult =  pd.merge(pred_out, FutureIDDF, left_on=['Appointment ID'], right_on=['Appointment ID'] ) 

print(pred_outResult.head())


# In[ ]:


pd.crosstab(pred_outResult['Attend'], pred_outResult['SourceStatus'])


# In[ ]:


pred_outResult.to_csv(xpath + "/pred_outResultJuly2018B.csv",index=True, header=True, na_rep="NA")

