
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[14]:



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
    

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelBinarizer



# In[3]:
def prepareData(Appointdf):
    print("Subset data")
    Appointdf= Appointdf[[
        'Hosp Code', 
        'SourceAppointmentTypeCode',
        'AppointmentSequence',
       'Date Of Appointment', 
        'Date Of Appointment Booked',
       #'Clinic Category Code', 
        'ClinicType', 
        #'SpecialtyCode', 
        'Time of Slot',
       'Appointment Reason Code', 
        'Age', 
        'Indigenous Status at Apppointment',
       'AccountTypeCode', 
       'IsFirstNonCancelledAppointment',
       'Is Interpreter Required', 
        'DayOfWeekShort', 
        'MonthNameShort',
       'Postcode', 
        'Gender', 
        'MinTemp', 
        'MaxTemp', 
        'rainfall', 
        'AHSCode',
       'RemotenessCode', 
        'SEIFAStatePercentile', 
        'T2CODE', 
       'PriorAttendanceRate',
        'Patient Attended' ##Y
     ]]

    print("")
    print("diff Appoint_Book_Length...")
    print("............................................")
    Appointdf["Date Of Appointment"] =  pd.to_datetime(Appointdf["Date Of Appointment"])
    Appointdf['Date Of Appointment Booked'] =  pd.to_datetime(Appointdf['Date Of Appointment Booked'])
    Appointdf["Appoint_Book_Length"] = Appointdf["Date Of Appointment"] -  Appointdf['Date Of Appointment Booked']
    print(Appointdf['Date Of Appointment'].min())
    print(Appointdf['Date Of Appointment'].max())
    from dateutil.relativedelta import relativedelta
    
    Appointdf["Appoint_Book_Length"] = Appointdf["Appoint_Book_Length"].dt.total_seconds()/ 3600 
    Appointdf.loc[(Appointdf["Appoint_Book_Length"] < -1 ) , ["Appoint_Book_Length"] ] = -1
    
    Appointdf = Appointdf.drop(columns=["Date Of Appointment", 'Date Of Appointment Booked', ])

    mask = Appointdf['Appointment Reason Code'] == '[No Value]'
    Appointdf['Appointment Reason Code'].loc[mask]  = 'NoVal'
    mask = Appointdf['AccountTypeCode'] == '[No Value]'
    Appointdf['AccountTypeCode'].loc[mask]  = 'NoVal'
    Appointdf= Appointdf.replace({r'No Value': 'NoVal'}, regex=True)
    NoValmask = Appointdf.applymap(lambda x:  r'No Value' in str(x))
    
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
        novalmask = Appointdf.Postcode.astype(str).str.contains('No Value', regex=False, na=False)
        Appointdf.loc[novalmask, 'Postcode'] = 1
        Appointdf.Postcode = Appointdf.Postcode.astype('int')
    if 'AccountTypeCode' in Appointdf.columns:
        novalmask = Appointdf['AccountTypeCode'].astype(str).str.contains('No Value', regex=False, na=False)
        Appointdf.loc[novalmask, 'AccountTypeCode'] = 'NoVal'

    if 'Appointment Reason Code' in Appointdf.columns:
        novalmask = Appointdf['Appointment Reason Code'].str.contains('No Value', regex=False, na=False)
        Appointdf.loc[novalmask, 'Appointment Reason Code'] = 'NoVal'
        
        
    print("............................................")
    print("............................................")
    print("Null count: {}".format( Appointdf.columns[Appointdf.isna().any()].tolist() ))
    print("............................................")
    print(Appointdf.isna().any())
    Appointdf['AppointmentSequence']=Appointdf['AppointmentSequence'].fillna(Appointdf['AppointmentSequence'].mean())
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



from  sklearn.preprocessing import LabelEncoder

class MultiColumnLabelEncoder(LabelEncoder):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns
        self.all_encoders_ = {}
        self.all_classes_ = {}

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            #self.all_classes_ = np.ndarray(shape=self.columns.shape,dtype=object)
            #self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                print("column {}".format( column ) )
                # fit LabelEncoder to get `classes_` for the column
                le = LabelEncoder()
                
                le = le.fit(dframe.loc[:, column].fillna('NA_VAL').values)
                # append the `classes_` to our ndarray container
                self.all_classes_[column] = np.array(le.classes_.tolist(), dtype=object)
                
                # append this column's encoder
                self.all_encoders_[column] = le
                #print("dframe.loc[:, column].values {}".format(dframe.loc[:, column].values ) )
                print("self.all_encoders_[column]classes_.tolist() {}".format(self.all_encoders_[column].classes_.tolist()) )
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            #self.all_classes_ = np.ndarray(shape=self.columns.shape,dtype=object)
            self.all_classes_ = {}
            for idx, column in enumerate(self.columns):
                le = LabelEncoder()
                le = le.fit(dframe.loc[:, column].fillna('NA_VAL').values)
                self.all_classes_[column] = np.array(le.classes_.tolist(),dtype=object)
                self.all_encoders_[column] = le
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            print("self.columns {}".format(self.columns))
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            #self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
            self.all_classes_ = {}
            #self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
            self.all_encoders_ = {}
            #self.all_labels_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelEncoder()
                # fit and transform labels in the column
                dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].fillna('NA_VAL'))
                # append the `classes_` to our ndarray container
                self.all_classes_[column] = np.array(le.classes_.tolist(),dtype=object)
                self.all_encoders_[column] = le
                self.all_labels_[column] = le
        else:
            # no columns specified; assume all are to be encoded
            self.columns = dframe.iloc[:, :].columns
            #self.all_classes_ = np.ndarray(shape=self.columns.shape,dtype=object)
            self.all_classes_ = {}
            for idx, column in enumerate(self.columns):
                print("idx {}, column {}".format(idx, column))
                le = LabelEncoder()
                dframe.loc[:, column] = le.fit_transform(
                        dframe.loc[:, column].fillna('NA_VAL').values)
                self.all_classes_[column] = (column, 
                                          np.array(le.classes_.tolist(),
                                                  dtype=object))
                self.all_encoders_[column] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        print("self.all_encoders_ {}".format(self.all_encoders_))
        if self.columns is not None:
            print("self.columns {}".format(self.columns))
            for idx, column in enumerate(self.columns):
                print("idx {}, column {}".format(idx, column))
                #print(self.all_encoders_[column].transform(dframe.loc[:, column].values))
                dframe.loc[:, column] = self.all_encoders_[column].transform(dframe.loc[:, column].fillna('NA_VAL').values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                print("idx {}, column {}".format(idx, column))
                dframe.loc[:, column] = self.all_encoders_[column].transform(dframe.loc[:, column].fillna('NA_VAL').values)
        #return dframe.loc[:, ].values
        return dframe

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[column]                    .inverse_transform(dframe.loc[:, column].values)
        else:
            self.columns = dframe.iloc[:, :].columns
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[column]                    .inverse_transform(dframe.loc[:, column].values)
        return dframe
    
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




from  sklearn.preprocessing import LabelBinarizer

class MultiColumnLabelBin(LabelBinarizer):
    """
    Wraps sklearn LabelEncoder functionality for use on multiple columns of a
    pandas dataframe.

    """
    def __init__(self, columns=None):
        self.columns = columns
        self.all_encoders_ = {}
        self.all_classes_ = {}

    def fit(self, dframe):
        """
        Fit label encoder to pandas columns.

        Access individual column classes via indexig `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`
        """
        # if columns are provided, iterate through and get `classes_`
        
        if self.columns is not None:
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            #self.all_classes_ = np.ndarray(shape=self.columns.shape,dtype=object)
            #self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                print("column {}".format( column ) )
                # fit LabelEncoder to get `classes_` for the column
                le = LabelBinarizer()
                le = le.fit(dframe.loc[:, column].astype(str).fillna('NA_VAL').values)
                # append the `classes_` to our ndarray container
                self.all_classes_[column] = le.classes_
                print(le)
                # append this column's encoder
                self.all_encoders_[column] = le
                #print("dframe.loc[:, column].values {}".format(dframe.loc[:, column].values ) )
                print("self.all_encoders_[column]classes_.tolist() {}".format(self.all_encoders_[column].classes_.tolist()) )
                
        return self

    def fit_transform(self, dframe):
        """
        Fit label encoder and return encoded labels.

        Access individual column classes via indexing
        `self.all_classes_`

        Access individual column encoders via indexing
        `self.all_encoders_`

        Access individual column encoded labels via indexing
        `self.all_labels_`
        """
        # if columns are provided, iterate through and get `classes_`
        if self.columns is not None:
            print("self.columns {}".format(self.columns))
            # ndarray to hold LabelEncoder().classes_ for each
            # column; should match the shape of specified `columns`
            #self.all_classes_ = np.ndarray(shape=self.columns.shape, dtype=object)
            self.all_classes_ = {}
            #self.all_encoders_ = np.ndarray(shape=self.columns.shape, dtype=object)
            self.all_encoders_ = {}
            #self.all_labels_ = np.ndarray(shape=self.columns.shape, dtype=object)
            for idx, column in enumerate(self.columns):
                # instantiate LabelEncoder
                le = LabelBinarizer()
                # fit and transform labels in the column
                dframe.loc[:, column] = le.fit_transform(dframe.loc[:, column].astype(str).fillna('NA_VAL'))
                # append the `classes_` to our ndarray container
                self.all_classes_[column] = le.classes_
                self.all_encoders_[column] = le
                self.all_labels_[column] = le
        return dframe

    def transform(self, dframe):
        """
        Transform labels to normalized encoding.
        """
        print("self.all_encoders_ {}".format(self.all_encoders_))
        if self.columns is not None:
            print("self.columns {}".format(self.columns))
            dfOneHotList = [dframe]
            colList = []
            for idx, column in enumerate(self.columns):
                print("idx {}, column {}".format(idx, column))
                #print(self.all_encoders_[column].transform(dframe.loc[:, column].values))
                lb_results = self.all_encoders_[column].transform(dframe.loc[:, column].astype(str).fillna('NA_VAL').values)
                if len(mcle.all_classes_[column]) == 2:
                    dfOneHot = pd.DataFrame(lb_results, columns=[column] )
                else:
                    xcolumns = [ column + "_" + c.strip().replace("/","_").replace("(","_").replace(")","_").replace(" ","_").replace("[","_").replace("]","_").replace(".","_").replace(r"\\t","_").replace(r"\\n","_") for c in self.all_classes_[column].tolist()]
                    print("xcolumns {}".format(xcolumns))
                    dfOneHot = pd.DataFrame(lb_results, columns=xcolumns )
                    
                dfOneHotList.append(dfOneHot)
                colList.append(column)
            dframe = pd.concat(dfOneHotList, axis=1)
            #self.columns.tolist()
            dframe.drop(columns=colList, inplace=True)
        return dframe

    def inverse_transform(self, dframe):
        """
        Transform labels back to original encoding.
        """
        if self.columns is not None:
            for idx, column in enumerate(self.columns):
                dframe.loc[:, column] = self.all_encoders_[column]                    .inverse_transform(dframe.loc[:, column].values)
        return dframe
    


    

xpath = "/home/he159490/DS/OUT/"
xpath = "/media/DataDrive/"
AppointCSV = "Appointments_10Aug2015_17Aug2018.csv"


# In[4]:



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



cat_vars=[
        'Hosp Code', 
        'SourceAppointmentTypeCode',
        #'Clinic Category Code', 
        'ClinicType', 
        #'SpecialtyCode', 
        'Time of Slot',
        'Appointment Reason Code', 
        'Indigenous Status at Apppointment',
       'AccountTypeCode', 
       #'IsFirstNonCancelledAppointment',
       #'Is Interpreter Required', 
        'DayOfWeekShort', 
        'MonthNameShort',
       'Postcode', 
        'Gender', 
        'AHSCode',
       'RemotenessCode', 
        'T2CODE']



def PredictFuture(futureFile):
    futureDataO = pd.read_csv(futureFile)
    
    #print(futureDataO['Hosp Code'].value_counts())
    #print(futureDataO['Hosp Code'][FS | FH])
    #futureData = futureDataO[RPH]
    futureData = futureDataO
    #futureData=futureData.dropna()
    print(futureData.shape)
    print(futureData.head(2))
    futureData.dropna(inplace=True)
    import datetime
    futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
    print("test Data size {}".format(futureData.shape))
    futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
    #futureData = futureData[futureData['Date Of Appointment'] >= datetime.date(year=2018,month=7,day=21) ]
    print("test after date  Data size {}".format(futureData.shape))
    FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
    
    FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
    print("FutureIDDF size {}".format(FutureIDDF.shape))
    print("FutureIDDF cols {}".format(FutureIDDF.columns))
    
    futureDatap,Yf=prepareData(Appointdf=futureData)
    #futureDatapDummy = Dummify(Appointdf=futureDatap, cat_vars=cat_vars)
    futureDatapDummy = mcle.transform(futureDatap)
    print("futureDatapDummy {}".format(futureDatapDummy.shape))
    
    futureDatapDummy['AppointmentSequence'].fillna((futureDatapDummy['AppointmentSequence'].mean()), inplace=True)
    print('AppointmentSequence')
    futureDatapDummy['Age'].fillna( futureDatapDummy['Age'].mean() , inplace=True)
    print('Age')
    futureDatapDummy['IsFirstNonCancelledAppointment'].fillna( 0 , inplace=True)
    print('IsFirstNonCancelledAppointment')
    futureDatapDummy['Is Interpreter Required'].fillna(0, inplace=True)
    print('Is Interpreter Required')
    futureDatapDummy['MinTemp'].fillna((futureDatapDummy['MinTemp'].mean()), inplace=True)
    print('MinTemp')
    futureDatapDummy['MaxTemp'].fillna((futureDatapDummy['MaxTemp'].mean()), inplace=True)
    print('MaxTemp')
    futureDatapDummy['rainfall'].fillna( 0, inplace=True)
    print('rainfall')
    futureDatapDummy['SEIFAStatePercentile'].fillna( (futureDatapDummy['SEIFAStatePercentile'].mean()), inplace=True)
    print('SEIFAStatePercentile')
    futureDatapDummy['PriorAttendanceRate'].fillna((futureDatapDummy['PriorAttendanceRate'].mean()), inplace=True)
    print('PriorAttendanceRate')
    futureDatapDummy['Appoint_Book_Length'].fillna( (futureDatapDummy['Appoint_Book_Length'].mean() ), inplace=True)
    print('Appoint_Book_Length')
    
    print("Null: {}".format(futureDatapDummy.isnull().sum(axis=0)))
    
    futureDatapDummy.dropna(inplace=True)
    
    print("after concat futureDatapDummy {}".format(futureDatapDummy.shape))
    print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
    #NullCols = futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist()
    #for c in NullCols:
    #    futureDatapDummy[c]=0
    print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
    futureDatapDummy.head(1)
    
    predictions_Future = clf.predict(futureDatapDummy)
    predictions_prob_Future = clf.predict_proba(futureDatapDummy)
    #predxgs_Future = clsb.predict(futureDatapDummy)
    #predxgs_prob_Future = clsb.predict_proba(futureDatapDummy)
    predxgs_Future = logreg.predict(futureDatapDummy)
    predxgs_prob_Future = logreg.predict_proba(futureDatapDummy)
    print(clf.classes_)
    
    predNB_Future = logreg.predict(futureDatapDummy)
    predNB_prob_Future = logreg.predict_proba(futureDatapDummy)
    
    predgnb_Future = gnb.predict(futureDatapDummy)
    predgnb_prob_Future = gnb.predict_proba(futureDatapDummy)
    
    predcat_Future = catmodel.predict(futureDatapDummy)
    predcat_prob_Future = catmodel.predict_proba(futureDatapDummy)
    
    predictionsFutureDF = pd.DataFrame(predictions_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
    predictionsFutureDFB = pd.DataFrame(predxgs_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
    predictionsFutureDFNB = pd.DataFrame(predgnb_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
    
    predictionsFutureDFcat = pd.DataFrame(predcat_prob_Future, columns=["WillMissProb", "WillNotMissProb"])
    
    print(FutureIDDF.shape)
    print(predictions_Future.shape)
    
    FutureIDDF = FutureIDDF.assign(Attend=predictions_Future)  
    FutureIDDF = FutureIDDF.assign(AttendB=predxgs_Future)  
    FutureIDDF = FutureIDDF.assign(AttendNB=predgnb_Future)  
    FutureIDDF = FutureIDDF.assign(Attendcat=predcat_Future)
    
    FutureIDDF = FutureIDDF.assign(MissProb=predictionsFutureDF["WillMissProb"].values)
    FutureIDDF = FutureIDDF.assign(MissProbB=predictionsFutureDFB["WillMissProb"].values)
    FutureIDDF = FutureIDDF.assign(MissProbNB=predictionsFutureDFNB["WillMissProb"].values)
    FutureIDDF = FutureIDDF.assign(MissProbcat=predictionsFutureDFcat["WillMissProb"].values)
    mask = FutureIDDF[ (FutureIDDF['AttendB']==0) &  (FutureIDDF['Attend']==0) ] 
    #    FutureIDDF["AttendALL"]
    FutureIDDF["AllAttend"] = 1
    #FutureIDDF.loc[(FutureIDDF['AttendB']==0) &  (FutureIDDF['Attend']==0), ["AllAttend"] ] = 0
    FutureIDDF["MissProbAll"]  = (FutureIDDF["MissProb"] * 1.25 +  FutureIDDF["MissProbB"] * 1.53 + FutureIDDF['MissProbNB'] * 4.4 + FutureIDDF['MissProbcat'] * 2.33   )/(1.25 + 1.53 + 4.4 + 2.33)     

    FutureIDDF.loc[(FutureIDDF['MissProbAll']>=0.5) , ["AllAttend"] ] = 0
    return FutureIDDF


# In[181]:


from sklearn.externals import joblib
from catboost import CatBoostClassifier

clf = joblib.load(xpath + '/RF_JL20132018.pkl') 
logreg = joblib.load( xpath + '/LR_JL20132018.pkl') 
gnb = joblib.load(xpath + '/NB_JL20132018.pkl') 
catmodel = CatBoostClassifier()
catmodel = catmodel.load_model(fname = xpath + '/CatBoost_JL20132018.pkl', format="cbm")

mcle = joblib.load(xpath + '/LabelBin.pkl') 

pred_outResult = PredictFuture(xpath + "/Appointments_03Sep2018_28Sep2018.csv" )
pred_outResult.to_csv(xpath + "/pred_outResult__03Sep2018_28Sep2018.csv",index=True, header=True, na_rep="NA")
print("Future prediction completed!!")


