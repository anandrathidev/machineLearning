# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:21:54 2018

@author: he21061
"""

import logging
import numpy as np
import pandas as pd
import sqlite3
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

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
xpath = "C:/temp/DataScience/Out/"
AppointCSV = "Outpatients Appointments.csv"

Appointdf = pd.read_csv(xpath + AppointCSV)

###################################################################################################### 
###################################################################################################### 
################################## HELPER FUNCTIONS  #################################################
###################################################################################################### 
###################################################################################################### 

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelBinarizer

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

###################################################################################################### 
###################################################################################################### 
################################## DATA UNDERSTANDING & CLEANING ####################################
###################################################################################################### 
###################################################################################################### 

Appointdf.head(10)
print(list(Appointdf.columns))



print("Subset data")
Appointdf = Appointdf[['Hosp Code',
 'Appointment Sequence',
 'Health Service Code',
 'Datetime of Appointment',
 'Date Of Appointment Booked',
 'Clinic Category',
 'Time of Slot',
 'Duration of Appointment',
 'Time Of Arrival',
 'Source Appointment Reason',
 'Appointment Reason',
 'Indigenous Status at Apppointment',
 'Postcode at Appointment',
 'Account Type',
 'Count of Patient Attended Appointments',
 'Count of Referred to Inpatients',
 'Is Interpreter Required',
 'Age',
 'DayOfWeekShort',
 'MonthNameShort',
 '24Hour',
 'Postcode',
 'Gender',
 'CountryOfBirth',
 'MinTemp',
 'MaxTemp',
 'rainfall',
  'Source Appointment Status' ## Y 
 ]]

Appointdf["Datetime of Appointment"] =  pd.to_datetime(Appointdf["Datetime of Appointment"])
Appointdf['Date Of Appointment Booked'] =  pd.to_datetime(Appointdf['Date Of Appointment Booked'])
Appointdf["Appoint_Book_Length"] = Appointdf["Datetime of Appointment"] -  Appointdf['Date Of Appointment Booked']
Appointdf["Appoint_Book_Length"] = Appointdf["Appoint_Book_Length"].dt.total_seconds()

Appointdf = Appointdf.drop(columns=["Datetime of Appointment", 'Date Of Appointment Booked', ])

print("drop Source Appointment Status== [No Value]")
Appointdf['Source Appointment Status'].unique()
Appointdf.groupby(by=['Source Appointment Status']).count()
Appointdf = Appointdf[Appointdf['Source Appointment Status']!='[No Value]']
Appointdf['Source Appointment Status'].unique()
Appointdf.groupby(by=['Source Appointment Status']).count()
print(list(Appointdf.columns))

Appointdf['Appointment Reason'].unique()
Appointdf.groupby(by=['Appointment Reason']).count()
Appointdf['Source Appointment Reason'].unique()
Appointdf.groupby(by=['Source Appointment Reason']).count()

print("drop 'Appointment Reason', 'Source Appointment Reason'")
Appointdf = Appointdf.drop(columns=['Appointment Reason', 'Source Appointment Reason', ])
print(list(Appointdf.columns))
print((Appointdf.dtypes))

print("drop Time Of Arrival")
Appointdf.pop('Time Of Arrival' )
print(list(Appointdf.columns))
print((Appointdf.dtypes))

Appointdf['Time of Slot'] =   pd.to_datetime(Appointdf['Time of Slot'] )
Appointdf['Time of Slot'] .head()
Appointdf['Time of Slot']  = Appointdf['Time of Slot'].dt.hour


Appointdf['Postcode_change'] = Appointdf['Postcode at Appointment']  != Appointdf['Postcode'] 
print(list(Appointdf.columns))
print((Appointdf.dtypes))
Appointdf.drop(columns=['Postcode at Appointment'], inplace =True)
#Appointdf.pop('Postcode at Appointment')
#Appointdf.drop('Postcode at Appointment', inplace =True)
print("Writing to Appointdf_Clean.csv")
#Appointdf.to_csv(xpath + "/Appointdf_Clean.csv")
Appointdf.isnull().sum()     
Appointdf['Duration of Appointment'].fillna(Appointdf['Duration of Appointment'].mean(), inplace=True) 

Appointdf.isnull().sum()     

Appointdf['Appoint_Diff'].median()

temp = Appointdf['Appoint_Diff']

Appointdf['Appoint_Diff'].fillna(Appointdf['Appoint_Diff'].mean(), inplace=True) 

Appointdf.isnull().sum()


###################################################################################################### 
################################## missing vals treatment        ####################################
###################################################################################################### 

print("Dummyfing data")

cat_vars=['Hosp Code',
          'Health Service Code','Clinic Category',
          'Indigenous Status at Apppointment',
          'Account Type',
          'DayOfWeekShort',
          'MonthNameShort',
          'Postcode',
          'Gender', 
          'CountryOfBirth']

for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(Appointdf[var], prefix=var)
    data1=Appointdf.join(cat_list)
    Appointdf=data1

del data1
  
print("drop dummified columns")
Appointdf.drop(columns=cat_vars, inplace =True )
print(list(Appointdf.columns))
print((Appointdf.dtypes))
tylist = [i for i in Appointdf.dtypes]
for itylist in tylist:
  #print(str(itylist))
  if itylist == 'object':
     print(itylist)
     
###################################################################################################### 
###################################################################################################### 
################################## DATA SPLIT                    ####################################
###################################################################################################### 
###################################################################################################### 

Appointdf.isnull().sum()

#Appointdf.to_csv(xpath + "/Appointdf_Clean.csv")

Y = Appointdf.pop('Source Appointment Status')
X_train, X_test, y_train, y_test = train_test_split( Appointdf,
                                                    Y, 
                                                    test_size=0.33, 
                                                    random_state=42, 
                                                    stratify=Y)

del Appointdf

###################################################################################################### 
###################################################################################################### 
################################## TRAIN CLASSIFIER               ####################################
###################################################################################################### 
###################################################################################################### 

clf = RandomForestClassifier(n_jobs=2)
print("Train Random forest")
clf.fit(X_train, y_train)

###################################################################################################### 
###################################################################################################### 
################################## Important Features             ####################################
###################################################################################################### 
###################################################################################################### 

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
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    #plt.xticks(range(X.shape[1]), indices)
    plt.xticks(range(X.shape[1]), X.columns[indices])
    plt.xlim([-1, X.shape[1]])
  else:
    for f in range(fsize):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    plt.bar(range(fsize), importances[indices[:fsize]],
           color="r", yerr=std[indices[:fsize]], align="center")
    plt.xticks(range(fsize), X.columns[indices[:fsize]] , rotation=80)
    plt.xlim([-1, fsize])
  plt.show()

FeatureImportance(forest=clf, X=X_train, fsize=10)

clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
               axis=0)
indices = np.argsort(clf.feature_importances_)[::-1]
X_train.shape[1] 
X_train.columns[indices[:10]]

plt.bar(range(10), clf.feature_importances_[indices[:10]],
         color="r", yerr=std[indices[:10]], align="center")

###################################################################################################### 
###################################################################################################### 
################################## EVALUATE & PREDICT             ####################################
###################################################################################################### 
###################################################################################################### 

print("Predict ....")
predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)
print("evaluate ....")
print_evaluation_scores_multi_class(y_val=y_test, predicted=predictions)

