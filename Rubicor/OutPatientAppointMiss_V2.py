# -*- coding: utf-8 -*-
"""
Created on Wed May 16 09:21:54 2018

@author: he21061
"""

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




xpath = "C:/temp/DataScience/Out/"
AppointCSV = "Outpatients Appointments.csv"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


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



def prepareData(Appointdf):
    print("Subset data")
    Appointdf= Appointdf[[
            'Hosp Code',
     'Appointment Sequence',
     'Health Service Code',
     'Datetime of Appointment',
     'Date Of Appointment Booked',
     'Clinic Category Code',
     'Time of Slot',
     'Duration of Appointment',
     'Indigenous Status at Apppointment',
     'Account Type',
     'Count of Referred to Inpatients',
     'Is Interpreter Required',
     'Age',
     'DayOfWeekShort',
     'MonthNameShort',
     'Gender',
     'Source Appointment Reason',
     'Count of Patient Attended Appointments', ## Y 
     ]]
    
    print("")
    print("diff Appoint_Book_Length...")
    print("............................................")
    Appointdf["Datetime of Appointment"] =  pd.to_datetime(Appointdf["Datetime of Appointment"])
    Appointdf['Date Of Appointment Booked'] =  pd.to_datetime(Appointdf['Date Of Appointment Booked'])
    Appointdf["Appoint_Book_Length"] = Appointdf["Datetime of Appointment"] -  Appointdf['Date Of Appointment Booked']
    Appointdf["Appoint_Book_Length"] = Appointdf["Appoint_Book_Length"].dt.total_seconds()
    Appointdf = Appointdf.drop(columns=["Datetime of Appointment", 'Date Of Appointment Booked', ])

    print("")
    print("filter Appoint_Book_Length < 0...")
    print("............................................")
    #Appointdf =  Appointdf[  Appointdf["Appoint_Book_Length"]>=0]

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
    Appointdf['Duration of Appointment'].fillna(Appointdf['Duration of Appointment'].mean(), inplace=True) 
    
    Appointdf.isnull().sum()     

    #Appointdf['Appoint_Book_Length'].median()
    #temp = Appointdf['Appoint_Book_Length']
    
    print("")
    print("fill na Appoint_Book_Length  .........")
    print("............................................")
    #Appointdf['Appoint_Book_Length'].fillna(Appointdf['Appoint_Book_Length'].mean(), inplace=True) 
    Appointdf['Appoint_Book_Length'] = Appointdf['Appoint_Book_Length'].replace(np.nan, Appointdf['Appoint_Book_Length'].mean())


    print("............................................")
    print("............................................")
    print("Null count: {}".format( Appointdf.columns[Appointdf.isna().any()].tolist() ))
    print("............................................")
    
    #Appointdf['Duration of Appointment'].fillna(Appointdf['Duration of Appointment'].mean(), inplace=True) 
    Appointdf['Duration of Appointment'] = Appointdf['Duration of Appointment'].replace(np.nan, Appointdf['Duration of Appointment'].mean())
    Appointdf['Duration of Appointment'].unique()
    Appointdf['Duration of Appointment'].dtype
    Y = Appointdf.pop('Count of Patient Attended Appointments')
    return Appointdf,Y

Appointdf = pd.read_csv(xpath + AppointCSV)
Appointdf.head(10)
print(list(Appointdf.columns))
Appointdf,Y= prepareData(Appointdf=Appointdf)
print(list(Appointdf.columns))
Appointdf.head(1).to_csv(xpath + "/AppointCSV.csv")

###################################################################################################### 
################################## Dummy fing data               ####################################
###################################################################################################### 

print("Dummyfing data")
print("............................................")

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
        cat_list = pd.get_dummies(Appointdf[var], prefix=var, sparse=True)
        data1=Appointdf.join(cat_list)
        del cat_list
        Appointdf=data1
        del data1
   
    print("drop dummified columns")
    print("............................................")
    Appointdf.drop(columns=cat_vars, inplace =True )
    print(list(Appointdf.columns))
    print((Appointdf.dtypes))
    tylist = [i for i in Appointdf.dtypes]
    for itylist in tylist:
      #print(str(itylist))
      if itylist == 'object':
         print(itylist)
    return Appointdf

cat_vars=['Hosp Code',
          'Health Service Code',
          'Clinic Category Code',
          'Indigenous Status at Apppointment',
          'Account Type',
          'DayOfWeekShort',
          'MonthNameShort',
          'Gender', 
          'Source Appointment Reason']

Appointdf = Dummify(Appointdf=Appointdf, cat_vars=cat_vars)
list(Appointdf.columns)
bettercolumns = ['Appointment Sequence',
 'Time of Slot',
 'Duration of Appointment',
 'Count of Referred to Inpatients',
 'Is Interpreter Required',
 'Age',
 'Appoint_Book_Length',
 'Hosp Code_AKHS',
 'Hosp Code_BHS',
 'Hosp Code_FH',
 'Hosp Code_FS',
 'Hosp Code_KEMH',
 'Hosp Code_KHS',
 'Hosp Code_KW',
 'Hosp Code_MC',
 'Hosp Code_OPH',
 'Hosp Code_PJ',
 'Hosp Code_PMH',
 'Hosp Code_RKHS',
 'Hosp Code_RPH',
 'Hosp Code_SCGH',
 'Health Service Code_CAHS',
 'Health Service Code_EMHS',
 'Health Service Code_NMHS',
 'Health Service Code_SMHS',
 'Clinic Category Code_ACA',
 'Clinic Category Code_ACAT',
 'Clinic Category Code_ADO',
 'Clinic Category Code_AMAC',
 'Clinic Category Code_ANA',
 'Clinic Category Code_ANAE',
 'Clinic Category Code_ANAS',
 'Clinic Category Code_ANT',
 'Clinic Category Code_ANTE',
 'Clinic Category Code_ANTH',
 'Clinic Category Code_APY',
 'Clinic Category Code_ASTH',
 'Clinic Category Code_AUD',
 'Clinic Category Code_BFWA',
 'Clinic Category Code_BRE',
 'Clinic Category Code_BRFE',
 'Clinic Category Code_BRST',
 'Clinic Category Code_BUR',
 'Clinic Category Code_CAR',
 'Clinic Category Code_CARD',
 'Clinic Category Code_CARN',
 'Clinic Category Code_CCT',
 'Clinic Category Code_CHEM',
 'Clinic Category Code_CHI',
 'Clinic Category Code_CHP',
 'Clinic Category Code_CMDN',
 'Clinic Category Code_CMDS',
 'Clinic Category Code_CMN',
 'Clinic Category Code_COLP',
 'Clinic Category Code_CON',
 'Clinic Category Code_CONR',
 'Clinic Category Code_CONT',
 'Clinic Category Code_COT',
 'Clinic Category Code_CPSY',
 'Clinic Category Code_CTEC',
 'Clinic Category Code_CTS',
 'Clinic Category Code_DAAS',
 'Clinic Category Code_DACR',
 'Clinic Category Code_DAYH',
 'Clinic Category Code_DEDU',
 'Clinic Category Code_DEN',
 'Clinic Category Code_DENT',
 'Clinic Category Code_DER',
 'Clinic Category Code_DERM',
 'Clinic Category Code_DEVA',
 'Clinic Category Code_DIA',
 'Clinic Category Code_DIAB',
 'Clinic Category Code_DIAL',
 'Clinic Category Code_DIAN',
 'Clinic Category Code_DIE',
 'Clinic Category Code_DIET',
 'Clinic Category Code_ECD',
 'Clinic Category Code_EDTC',
 'Clinic Category Code_EMER',
 'Clinic Category Code_END',
 'Clinic Category Code_ENDO',
 'Clinic Category Code_ENT',
 'Clinic Category Code_ENTH',
 'Clinic Category Code_ETEC',
 'Clinic Category Code_FALL',
 'Clinic Category Code_FBCA',
 'Clinic Category Code_FERT',
 'Clinic Category Code_FPCL',
 'Clinic Category Code_FRAC',
 'Clinic Category Code_GADO',
 'Clinic Category Code_GAS',
 'Clinic Category Code_GAST',
 'Clinic Category Code_GER',
 'Clinic Category Code_GERI',
 'Clinic Category Code_GES',
 'Clinic Category Code_GMED',
 'Clinic Category Code_GNET',
 'Clinic Category Code_GPM',
 'Clinic Category Code_GSUR',
 'Clinic Category Code_GYN',
 'Clinic Category Code_GYNA',
 'Clinic Category Code_HAE',
 'Clinic Category Code_HAEM',
 'Clinic Category Code_HAN',
 'Clinic Category Code_HDTU',
 'Clinic Category Code_HEPA',
 'Clinic Category Code_HITH',
 'Clinic Category Code_HLK',
 'Clinic Category Code_HOPC',
 'Clinic Category Code_HYP',
 'Clinic Category Code_HYPE',
 'Clinic Category Code_ICS',
 'Clinic Category Code_IMM',
 'Clinic Category Code_IMML',
 'Clinic Category Code_IMMN',
 'Clinic Category Code_IMMU',
 'Clinic Category Code_INDI',
 'Clinic Category Code_INF',
 'Clinic Category Code_INFX',
 'Clinic Category Code_INTD',
 'Clinic Category Code_KPHY',
 'Clinic Category Code_KPOD',
 'Clinic Category Code_LYM',
 'Clinic Category Code_MATY',
 'Clinic Category Code_MCON',
 'Clinic Category Code_MDIE',
 'Clinic Category Code_MEDI',
 'Clinic Category Code_MENO',
 'Clinic Category Code_MET',
 'Clinic Category Code_MMH',
 'Clinic Category Code_MOCT',
 'Clinic Category Code_MPG',
 'Clinic Category Code_MPHY',
 'Clinic Category Code_MPOD',
 'Clinic Category Code_MSOC',
 'Clinic Category Code_MSPE',
 'Clinic Category Code_MTOC',
 'Clinic Category Code_NASR',
 'Clinic Category Code_NCMD',
 'Clinic Category Code_NEO',
 'Clinic Category Code_NEON',
 'Clinic Category Code_NES',
 'Clinic Category Code_NEU',
 'Clinic Category Code_NEUR',
 'Clinic Category Code_NEUS',
 'Clinic Category Code_NGEN',
 'Clinic Category Code_NIIS',
 'Clinic Category Code_NMPC',
 'Clinic Category Code_NONX',
 'Clinic Category Code_NRSG',
 'Clinic Category Code_NTEC',
 'Clinic Category Code_NUC',
 'Clinic Category Code_NUCL',
 'Clinic Category Code_OBS',
 'Clinic Category Code_OBST',
 'Clinic Category Code_OBSX',
 'Clinic Category Code_OCC',
 'Clinic Category Code_OCCT',
 'Clinic Category Code_OCTH',
 'Clinic Category Code_ONBR',
 'Clinic Category Code_ONC',
 'Clinic Category Code_ONCL',
 'Clinic Category Code_ONCO',
 'Clinic Category Code_ONPY',
 'Clinic Category Code_OPH',
 'Clinic Category Code_OPHT',
 'Clinic Category Code_OPT',
 'Clinic Category Code_ORA',
 'Clinic Category Code_ORP',
 'Clinic Category Code_ORT',
 'Clinic Category Code_ORTH',
 'Clinic Category Code_ORTT',
 'Clinic Category Code_OTC',
 'Clinic Category Code_PAAS',
 'Clinic Category Code_PAC',
 'Clinic Category Code_PAE',
 'Clinic Category Code_PAED',
 'Clinic Category Code_PAI',
 'Clinic Category Code_PAIN',
 'Clinic Category Code_PAL',
 'Clinic Category Code_PALL',
 'Clinic Category Code_PAS',
 'Clinic Category Code_PCOL',
 'Clinic Category Code_PCSR',
 'Clinic Category Code_PDIE',
 'Clinic Category Code_PELV',
 'Clinic Category Code_PHCY',
 'Clinic Category Code_PHTH',
 'Clinic Category Code_PHY',
 'Clinic Category Code_PHYS',
 'Clinic Category Code_PHYX',
 'Clinic Category Code_PLA',
 'Clinic Category Code_PLAS',
 'Clinic Category Code_PLST',
 'Clinic Category Code_PNLS',
 'Clinic Category Code_POD',
 'Clinic Category Code_PODI',
 'Clinic Category Code_PODM',
 'Clinic Category Code_PODT',
 'Clinic Category Code_PODX',
 'Clinic Category Code_PRE',
 'Clinic Category Code_PREA',
 'Clinic Category Code_PSG',
 'Clinic Category Code_PSGE',
 'Clinic Category Code_PSY',
 'Clinic Category Code_PSYC',
 'Clinic Category Code_PYO',
 'Clinic Category Code_RAC',
 'Clinic Category Code_RAD',
 'Clinic Category Code_RADL',
 'Clinic Category Code_RADT',
 'Clinic Category Code_RAO',
 'Clinic Category Code_REH',
 'Clinic Category Code_REM',
 'Clinic Category Code_RENA',
 'Clinic Category Code_RES',
 'Clinic Category Code_RESN',
 'Clinic Category Code_RESP',
 'Clinic Category Code_RET',
 'Clinic Category Code_RHAB',
 'Clinic Category Code_RHE',
 'Clinic Category Code_RHEU',
 'Clinic Category Code_RIT',
 'Clinic Category Code_RITH',
 'Clinic Category Code_SAM',
 'Clinic Category Code_SBRE',
 'Clinic Category Code_SCOL',
 'Clinic Category Code_SHCL',
 'Clinic Category Code_SOCI',
 'Clinic Category Code_SOCW',
 'Clinic Category Code_SOW',
 'Clinic Category Code_SOWK',
 'Clinic Category Code_SPAC',
 'Clinic Category Code_SPCH',
 'Clinic Category Code_SPEE',
 'Clinic Category Code_SPP',
 'Clinic Category Code_SPS',
 'Clinic Category Code_STM',
 'Clinic Category Code_STOM',
 'Clinic Category Code_SURG',
 'Clinic Category Code_SWOR',
 'Clinic Category Code_THOR',
 'Clinic Category Code_ULCN',
 'Clinic Category Code_URO',
 'Clinic Category Code_UROL',
 'Clinic Category Code_VAS',
 'Clinic Category Code_VASC',
 'Clinic Category Code_VTEC',
 'Clinic Category Code_WOU',
 'Clinic Category Code_YCS',
 'Clinic Category Code_Unknown',
 'Indigenous Status at Apppointment_Aboriginal but not Torres Strait Islander origin',
 'Indigenous Status at Apppointment_Both Aboriginal and Torres Strait Islander origin',
 'Indigenous Status at Apppointment_Neither Aboriginal nor Torres Strait Islander origin',
 'Indigenous Status at Apppointment_Not stated_inadequately described',
 'Indigenous Status at Apppointment_Torres Strait Islander but not Aboriginal origin',
 'Indigenous Status at Apppointment_Unknown',
 'Account Type_AUSTRALIAN DEFENCE FORCES',
 'Account Type_Bulk Billed',
 'Account Type_COMPENSABLE OTHER',
 'Account Type_DEPT. OF VETERAN AFFAIRS',
 'Account Type_EASTERN STATES MVIT',
 'Account Type_FOREIGN DEFENCE FORCES',
 'Account Type_INELIGIBLE OVERSEAS VISITOR',
 'Account Type_NHT Private',
 'Account Type_NHT Public',
 'Account Type_NHT Veteran Affairs',
 'Account Type_OVERSEAS STUDENT',
 'Account Type_PRIVATE INSURED',
 'Account Type_PRIVATE UNINSURED',
 'Account Type_PUBLIC',
 'Account Type_SHIPPING',
 'Account Type_UNACCOMPANIED_UNKNOWN',
 'Account Type_WA   SGIC_MVIT',
 'Account Type_WORKERS COMPENSATION',
 'Account Type_NoValue',
 'DayOfWeekShort_Fri',
 'DayOfWeekShort_Mon',
 'DayOfWeekShort_Sat',
 'DayOfWeekShort_Sun',
 'DayOfWeekShort_Thu',
 'DayOfWeekShort_Tue',
 'DayOfWeekShort_Wed',
 'MonthNameShort_Apr',
 'MonthNameShort_Aug',
 'MonthNameShort_Dec',
 'MonthNameShort_Feb',
 'MonthNameShort_Jan',
 'MonthNameShort_Jul',
 'MonthNameShort_Jun',
 'MonthNameShort_Mar',
 'MonthNameShort_May',
 'MonthNameShort_Nov',
 'MonthNameShort_Oct',
 'MonthNameShort_Sep',
 'Gender_Female',
 'Gender_Intersex or indeterminate',
 'Gender_Male',
 'Gender_Not stated_inadequately described',
 'Source Appointment Reason_Advice',
 'Source Appointment Reason_Antenatal Classes',
 'Source Appointment Reason_Anticoagulation Therapy',
 'Source Appointment Reason_Appointment',
 'Source Appointment Reason_Assessment',
 'Source Appointment Reason_Blood and specimen collection',
 'Source Appointment Reason_Cancelled admission for occasion of service',
 'Source Appointment Reason_Care planning',
 'Source Appointment Reason_Chart Review',
 'Source Appointment Reason_Contracted Services',
 'Source Appointment Reason_Counselling',
 'Source Appointment Reason_Diagnostic scan',
 'Source Appointment Reason_Drain Management',
 'Source Appointment Reason_Dressing',
 'Source Appointment Reason_Education',
 'Source Appointment Reason_Education_Support',
 'Source Appointment Reason_Follow Up',
 'Source Appointment Reason_Follow Up Acute',
 'Source Appointment Reason_Home Visit',
 'Source Appointment Reason_I_V Medication',
 'Source Appointment Reason_Injection',
 'Source Appointment Reason_Inpatient',
 'Source Appointment Reason_New Patient',
 'Source Appointment Reason_New Patient Splint',
 'Source Appointment Reason_Post inpatient/procedure review',
 'Source Appointment Reason_Preoperative education',
 'Source Appointment Reason_Prescription chart request',
 'Source Appointment Reason_Private Referral',
 'Source Appointment Reason_Pump refill',
 'Source Appointment Reason_Research trial',
 'Source Appointment Reason_Review',
 'Source Appointment Reason_S_C Medication',
 'Source Appointment Reason_Special Plastic Dressings',
 'Source Appointment Reason_Special Procedures',
 'Source Appointment Reason_Special Test',
 'Source Appointment Reason_Stoma Care',
 'Source Appointment Reason_Telephone Consult',
 'Source Appointment Reason_Telephone call',
 'Source Appointment Reason_Video conferencing',
 'Source Appointment Reason_Wound Management',
 'Source Appointment Reason_Wound care',
 'Source Appointment Reason_NoValue']

Appointdf.columns = bettercolumns


###################################################################################################### 
###################################################################################################### 
################################## prepare Test  data       ####################################
###################################################################################################### 
###################################################################################################### 

testCSV = "Future Outpatients Appointments.csv"
testdf = pd.read_csv(xpath + testCSV)
testdf.head(10)
print(list(testdf.columns))
print(list(Appointdf.columns))
testdfX,test_Y= prepareData(Appointdf=testdf)
print(list(testdf.columns))
testdfX = Dummify(Appointdf=testdfX, cat_vars=cat_vars)

testdfX = fix_columns( d=testdfX, columns=Appointdf.columns )
testdfX.columns = bettercolumns

X_test = testdfX
y_test = test_Y

###################################################################################################### 
###################################################################################################### 
################################## DATA SPLIT                    ####################################
###################################################################################################### 
###################################################################################################### 
if False:
    print("............................................")
    print("............................................")
    print("Null count After dummy : {}".format( Appointdf.columns[Appointdf.isna().any()].tolist() ))
    print("............................................")
    #Appointdf['Duration of Appointment'].fillna(Appointdf['Duration of Appointment'].mean(), inplace=True) 
    
    
    #Appointdf.to_csv(xpath + "/Appointdf_Clean.csv")
    
    print("............................................")
    print("............................................")
    print("Split train & test .........................")
    print("............................................")
    
    X_train, X_test, y_train, y_test = train_test_split( Appointdf,
                                                        Y, 
                                                        test_size=0.21, 
                                                        random_state=61, 
                                                        stratify=Y)
    
    del Appointdf
    del Y

###################################################################################################### 
###################################################################################################### 
################################## TRAIN CLASSIFIER               ####################################
###################################################################################################### 
###################################################################################################### 
#param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
#num_round = 3
#bst = clsb.fit(X_train,y_train)

#from xgboost import XGBClassifier
import xgboost as xgb

clf = None
clf = RandomForestClassifier(n_jobs=2)
clsb = xgb.XGBClassifier( nthread=3)

from sklearn import linear_model
logreg = linear_model.LogisticRegression(C=10)

#del clf
#clf = RandomForestClassifier(n_jobs=2, n_estimators=7,class_weight ='balanced')
print("............................................")
print("............................................")
print("Train Random forest")
print("............................................")
X_train, y_train = Appointdf, Y
clf.fit(X_train, y_train)
clsb.fit(X_train, y_train)

import pickle
from sklearn.externals import joblib

if clf == None:
    clf = joblib.load(xpath + "/RandimForest_7E_CWbalanced.joblib")
else:
    joblib.dump(clf,xpath + "/RandimForest.joblib")
    joblib.dump(clsb,xpath + "/XGSBoost.joblib")

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

print("............................................")
print("............................................")
print("Rank features ")
print("............................................")
FeatureImportance(forest=clf, X=X_train, fsize=200)

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
################################## PLOT LOGISTIC             ####################################
###################################################################################################### 
###################################################################################################### 



###################################################################################################### 
###################################################################################################### 
################################## EVALUATE & PREDICT             ####################################
###################################################################################################### 
###################################################################################################### 


print("RF Predict ....")
X_test=X_train
y_test=Y

list(X_train.columns)

predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)

predictionsDF = pd.DataFrame(predictions_prob, columns=["class_0", "class_1"])

resultDF = predictionsDF.join(testdf)
list(resultDF.columns)

resultDF = resultDF[['class_0',
 'class_1',
 'Hosp Code',
 'Appointment ID',
 'Account Number']]

resultDF.to_csv(xpath + "/Prediction.csv")

resultDF = pd.read_csv(xpath + "/Prediction.csv")
resultDF['ApptDate'] = pd.to_datetime(resultDF['Date Of Appointment']) - pd.to_timedelta(7, unit='d')
resultDF['qty'] = 1

PreGroupby_df = resultDF.groupby(['class_0', 'Hospital', 'Clinic Category', pd.Grouper(key='ApptDate', freq='D')]).agg({'qty': 'sum'}).reset_index()
SumGroup_df = resultDF.groupby([ 'Hospital', 'Clinic Category', pd.Grouper(key='ApptDate', freq='D')]).agg({'qty': 'sum'}).add_suffix('_Sum').reset_index()
resultGDF = pd.merge(PreGroupby_df, SumGroup_df)
resultGDF["Percent"] = 100.0*resultGDF["qty"] / resultGDF["qty_Sum"]

resultGDF.to_csv(xpath + "/GroupedPrediction.csv")

percent = resultGDF.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
percent.reset_index()


resultGDF.head(10)
resultGFullDF.head(10)

print("evaluate ....")
print_evaluation_scores_multi_class(y_val=y_test, predicted=predictions)
print("confusion matrix ....")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confMatrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True)
confMatrixXB = pd.crosstab(y_test, predxgs, rownames=['True'], colnames=['Predicted'], margins=True)

print(confMatrix)
print(confMatrixXB)

confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
print(confMatrixPercent)

