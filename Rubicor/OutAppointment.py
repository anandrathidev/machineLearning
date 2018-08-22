
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
xpath = "/media/DataDrive/"
AppointCSV = "Appointments_10Aug2016_10Aug2018.csv"


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
#AppointdfO = pd.read_csv(xpath + "/Appointments_07Aug2018_10Aug2018.csv")


# In[8]:


AppointdfO.shape


# In[9]:


#print(AppointdfO['Hosp Code'].value_counts())
#FS  = AppointdfO['Hosp Code']=='FS' 
#FH = AppointdfO['Hosp Code']=='FH' 
#RPH = AppointdfO['Hosp Code']=='RPH' 
#print(AppointdfO['Hosp Code'][FS | FH])


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
dtJul = parser.parse("06 Aug 2018")
dtJan = parser.parse("01 Jan 2013")
print(dtJul)
print(dtJan)


# In[14]:


AppointdfOFSFH=AppointdfOFSFH.sort_values('Date Of Appointment').reset_index()
print(AppointdfOFSFH['Date Of Appointment'].min())
print(AppointdfOFSFH['Date Of Appointment'].max())
from dateutil.relativedelta import relativedelta
#AppointdfOFSFH = AppointdfOFSFH.loc[AppointdfOFSFH['Date Of Appointment'].between(dtJan, dtJul)]
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


print(AppointdfOFSFH[AppointdfOFSFH['Appointment Reason Code'] == '[No Value]'].count())
print(AppointdfOFSFH['Appointment Reason Code'].value_counts())


# In[21]:


mask = AppointdfOFSFH['Appointment Reason Code'] == '[No Value]'
AppointdfOFSFH['Appointment Reason Code'].loc[mask]  = 'NoVal'


# In[22]:


mask = AppointdfOFSFH['AccountTypeCode'] == '[No Value]'
AppointdfOFSFH['AccountTypeCode'].loc[mask]  = 'NoVal'


# In[23]:


#print(AppointdfOFSFH.replace({r'No Value': 'NoVal'}, regex=True))
AppointdfOFSFH= AppointdfOFSFH.replace({r'No Value': 'NoVal'}, regex=True)


# In[24]:


#AppointdfOFSFH.search(r'No Value', regex=True)
NoValmask = AppointdfOFSFH.applymap(lambda x:  r'No Value' in str(x))


# In[25]:


np.count_nonzero(NoValmask)


# In[26]:


print(AppointdfOFSFH['Appointment Reason Code'].value_counts())


# In[27]:


print(AppointdfOFSFH.columns)


# In[28]:


print(AppointdfOFSFH.dtypes)


# In[29]:


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


# In[30]:


AppointdfOFSFH=AppointdfOFSFH.dropna()
Appointdf,Y= prepareData(Appointdf=AppointdfOFSFH)
print(list(Appointdf.columns))
Appointdf.head(1).to_csv(xpath + "/AppointCSV.csv")
del AppointdfOFSFH


# In[31]:


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


# In[32]:


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
    


# In[33]:


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


mcle = MultiColumnLabelEncoder(columns=cat_vars)
mcle = MultiColumnLabelBin(columns=cat_vars)


# In[34]:


#print(len(mcle.all_classes_['Is Interpreter Required']))
#le = mcle.all_encoders_['Is Interpreter Required']
#print(le)
#print(le.classes_)


# In[35]:


#print(mcle.all_encoders_[ 'Is Interpreter Required' ].transform(Appointdf.loc[:, 'Is Interpreter Required'].astype(str).fillna('NA_VAL').values))


# In[36]:


mcle = mcle.fit(Appointdf)


# In[37]:


AppointdfNonDummy=Appointdf
print(AppointdfNonDummy.dtypes)
print(AppointdfNonDummy.Postcode.unique())
print(AppointdfNonDummy.dtypes)
print(AppointdfNonDummy.Postcode.unique())


# In[38]:


print(len((AppointdfNonDummy.columns)))
print(len(set(AppointdfNonDummy.columns)))

#Appointdf = Dummify(Appointdf=AppointdfNonDummy, cat_vars=cat_vars)
# transform the `df` data
Appointdft = mcle.transform(Appointdf)


# In[39]:


Appointdft.shape
Appointdft.head()


# In[40]:


Appointdft.dtypes


# In[41]:


print(Appointdft.head())
print(sum(Appointdft.isnull().sum()))


# In[42]:


#print(Appointdft['SourceAppointmentTypeCode'])


# In[43]:


#del AppointdfNonDummy


# In[44]:


print(len((Appointdft.columns)))
print(len(set(Appointdft.columns)))


# In[45]:


for c in sorted(Appointdft.columns): 
    print(c)
    
print((set(Appointdft.columns)))


# In[46]:


#for c in sorted(set(Appointdf.columns)): 
#    print(c)


# In[47]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#Appointdf[Appointdf.columns] = scaler.fit_transform(Appointdf)


# In[48]:


#X_train, X_test, y_train, y_test = train_test_split( Appointdf,  Y,  test_size=0.30,  random_state=61) 


# In[49]:


X_train=Appointdft
X_test=Appointdft
y_test=Y
y_train=Y

Appointdf = Appointdft
print("X_train: : {}".format(X_train.shape))
print("X_test: {}".format(X_test.shape))
print("Null: {}".format(Appointdft.isnull().sum()))


# In[50]:


print(pd.options.display.max_columns)
print(pd.options.display.max_rows)
pd.options.display.max_columns=1200
pd.options.display.max_rows=1200
Appointdft.head(5)


# In[51]:


#for c in X_train.columns:
#   print(c)
#AppointmentSequence	Age	IsFirstNonCancelledAppointment	Is Interpreter Required	MinTemp	MaxTemp	rainfall	SEIFAStatePercentile	PriorAttendanceRate	Appoint_Book_Length
Appointdft['AppointmentSequence'].fillna((Appointdft['AppointmentSequence'].mean()), inplace=True)
print('AppointmentSequence')
Appointdft['Age'].fillna( Appointdft['Age'].mean() , inplace=True)
print('Age')
Appointdft['IsFirstNonCancelledAppointment'].fillna( 0 , inplace=True)
print('IsFirstNonCancelledAppointment')
Appointdft['Is Interpreter Required'].fillna(0, inplace=True)
print('Is Interpreter Required')
Appointdft['MinTemp'].fillna((Appointdft['MinTemp'].mean()), inplace=True)
print('MinTemp')
Appointdft['MaxTemp'].fillna((Appointdft['MaxTemp'].mean()), inplace=True)
print('MaxTemp')
Appointdft['rainfall'].fillna( 0, inplace=True)
print('rainfall')
Appointdft['SEIFAStatePercentile'].fillna( (Appointdft['SEIFAStatePercentile'].mean()), inplace=True)
print('SEIFAStatePercentile')
Appointdft['PriorAttendanceRate'].fillna((Appointdft['PriorAttendanceRate'].mean()), inplace=True)
print('PriorAttendanceRate')
Appointdft['Appoint_Book_Length'].fillna( (Appointdft['Appoint_Book_Length'].mean() ), inplace=True)
print('Appoint_Book_Length')

print("Null: {}".format(Appointdft.isnull().sum(axis=0)))


# In[52]:


#Appointdft.fillna( 0 , inplace=True)
Appointdft[Appointdft.isnull().any(axis=1)].head()


# In[53]:


Appointdft.dropna(inplace=True)


# In[54]:


print(Appointdft.shape)
print(Y.shape)


# In[55]:


clf = None
clf = RandomForestClassifier(max_depth=15, 
                             min_samples_split=20, 
                             min_samples_leaf=11, 
                             n_estimators=371, 
                             n_jobs=11,
                             warm_start=True, 
                             max_features="sqrt",
                             oob_score=True)

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

scoring = ('roc_auc', 'recall', 'f1', 'accuracy')

print("CROSS VALIDATE  Random forest")
#scores = cross_validate(clf, Appointdft,  Y, scoring=scoring, cv=2, return_train_score=True)
print("Train Random forest")
clf.fit(Appointdft,  Y)
#print(scores)


# In[256]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm 
svmclf = svm.SVC()
from sklearn.gaussian_process.kernels import RBF
#gp_opt = GaussianProcessClassifier(kernel=100.0 * RBF(), copy_X_train=False)

#scores = cross_validate(gp_opt, Appointdf,  Y, scoring=None, cv=1, return_train_score=False)
#svmclf.fit(Appointdf, Y)
#print(scores)    
from catboost import CatBoostClassifier
catmodel = CatBoostClassifier(iterations=250, learning_rate=0.7, depth=5, loss_function='Logloss')
catmodel.fit(Appointdf,  Y)
# Get predicted classes
# Get predicted probabilities for each class
catpreds_proba = catmodel.predict_proba(Appointdf)


# In[57]:


from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import recall_score

logreg = linear_model.LogisticRegression(C=0.865, penalty='l1')
scoring = ('roc_auc', 'recall', 'f1', 'accuracy')
scores = cross_validate(logreg, Appointdf, Y, scoring=scoring, cv=2, return_train_score=False)
logreg.fit(Appointdf,  Y)
print(scores)    


# In[58]:


print(xpath)


# In[59]:


#from sklearn.externals import joblib
#joblib.dump(clf, xpath + '/RF_JL20132018.pkl') 
#joblib.dump(logreg, xpath + '/LR_JL20132018.pkl') 
#joblib.dump(logreg, xpath + '/LR_JL20132018.pkl') 
#catmodel.save_model(xpath + '/CatBoost_JL20132018.pkl', format="cbm", export_parameters=None)


# In[60]:


from lightgbm import LGBMClassifier
clsb = LGBMClassifier(objective='binary',num_leaves=31,learning_rate=0.051,n_estimators=151)
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Appointdf,  Y)

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
#scores = cross_validate(clsb, Appointdf,  Y, scoring=scoring, cv=2, return_train_score=False)
#clsb.fit(Appointdf,  Y)
#print(scores)    


# In[61]:


from sklearn.externals import joblib
joblib.dump(clf, xpath + '/RF_JL20132018.pkl') 
joblib.dump(logreg, xpath + '/LR_JL20132018.pkl') 
joblib.dump(gnb, xpath + '/NB_JL20132018.pkl') 
catmodel.save_model(xpath + '/CatBoost_JL20132018.pkl', format="cbm", export_parameters=None)


# In[61]:


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

FeatureImportance(forest=clf, X=Appointdf, fsize=150)

clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
               axis=0)
indices = np.argsort(clf.feature_importances_)[::-1]
X_train.shape[1] 
X_train.columns[indices[:10]]


plt.bar(range(10), clf.feature_importances_[indices[:10]],
         color="r", yerr=std[indices[:10]], align="center")


# In[259]:


# Predict 
y_train=Y
y_test=Y

predictions = clf.predict(X_test)
predictions_prob = clf.predict_proba(X_test)
#predxgs = clsb.predict(X_test)
catpreds_class = catmodel.predict(X_test)
#predxgs_prob = clsb.predict_proba(X_test)
print("evaluate ....")
print_evaluation_scores_multi_class(y_val=y_test, predicted=predictions)
print_evaluation_scores_multi_class(y_val=y_test, predicted=catpreds_class)
print("confusion matrix ....")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confMatrix = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted'], margins=True)
confMatrixCAT = pd.crosstab(y_test, catpreds_class, rownames=['True'], colnames=['Predicted'], margins=True)

print("confMatrix {}".format(confMatrix))
print("confMatrixCAT {}".format( confMatrixCAT))

#confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
#print("confMatrixPercent {}".format( confMatrixPercent))


# In[63]:


print(Y.value_counts())
ser = pd.Series(predictions)
print(ser.value_counts())

print(X_train.shape)
print(X_test.shape)


# In[64]:


# Predict 
lpredictions = logreg.predict(X_test)
lpredictions_prob = logreg.predict_proba(X_test)
#predxgs = clsb.predict(X_test)
#predxgs_prob = clsb.predict_proba(X_test)
print("evaluate ....")
print_evaluation_scores_multi_class(y_val=y_test, predicted=lpredictions)
#print_evaluation_scores_multi_class(y_val=y_test, predicted=predxgs)
print("confusion matrix ....")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
confMatrix = pd.crosstab(y_test, lpredictions, rownames=['True'], colnames=['Predicted'], margins=True)
#confMatrixXB = pd.crosstab(y_test, predxgs, rownames=['True'], colnames=['Predicted'], margins=True)

print("confMatrix {}".format(confMatrix))
#print("confMatrixXB {}".format( confMatrixXB))

#confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
#print("confMatrixPercent {}".format( confMatrixPercent))
print(Y.value_counts())
ser = pd.Series(lpredictions)
print(ser.value_counts())


# In[258]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(y_test, lpredictions, labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))


# In[65]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(y_test, lpredictions, labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))


# In[66]:



#FeatureImportance(forest=logreg, X=Appointdf, fsize=150)
sorted_index = np.argsort(logreg.coef_[0])
#[::-1]
#print(sorted_index[:10])
coef = logreg.coef_[0]
print(coef[sorted_index[0:10]])
print(X_train.columns[sorted_index[0:10]])

#top_three = np.argpartition(coefs, -3)[-3:]

#testData = pd.read_csv(xpath + AppointCSV)
#print(testData.shape)


# In[67]:


#testData.head(2)


# In[68]:


#testData[testData["class_0"].notnull()].head(2) 


# In[69]:


#testData = testData[testData["class_0"].notnull()] 
#print("test Data size {}".format(testData.shape))


# In[70]:


#testDatap,Yt= prepareData(Appointdf=testData)


# In[71]:


#testDataDummy = Dummify(Appointdf=testDatap, cat_vars=cat_vars)


# In[72]:


#print(testDataDummy.shape )
#print(X_train.shape)


# In[73]:


XtrainEmpty = Appointdf[0:0]


# In[74]:


#print(XtrainEmpty.shape)
#print(XtrainEmpty.dtypes)


# In[75]:



#resultdf = pd.concat([XtrainEmpty, testDataDummy], sort=False)
#resultdf = testDataDummy[list(XtrainEmpty.columns)]


# In[76]:


#print(resultdf.shape)


# In[77]:


#print(resultdf.columns[resultdf.isna().any()].tolist())
#NullCols = resultdf.columns[resultdf.isna().any()].tolist()
#for c in NullCols:
#    resultdf[c]=0
    


# In[78]:


#print(resultdf.columns[resultdf.isna().any()].tolist())


# In[79]:


## Predict 
#print("Predict ....")
#predictions_2weeks = clf.predict(resultdf)
#predictions_prob_2weeks = clf.predict_proba(resultdf)
#predxgs_2weeks = clsb.predict(resultdf)
#predxgs_prob_2weeks = clsb.predict_proba(resultdf)


# In[80]:


#print("evaluate ....")
#print_evaluation_scores_multi_class(y_val=Yt, predicted=predictions_2weeks)
#print_evaluation_scores_multi_class(y_val=Yt, predicted=predxgs_2weeks)


# In[81]:


#print("confusion matrix ....")
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#confMatrix = pd.crosstab(Yt, predictions_2weeks, rownames=['True'], colnames=['Predicted'], margins=True)
#confMatrixXB = pd.crosstab(Yt, predxgs_2weeks, rownames=['True'], colnames=['Predicted'], margins=True)

#print(confMatrix)
#print(confMatrixXB)

#confMatrixPercent = pd.crosstab(y_test, predictions, rownames=['True'], colnames=['Predicted']).apply(lambda r: 100.0 * r/r.sum())
#print(confMatrixPercent)


# In[82]:


#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
#unique_label = np.unique(y_test)
#print(pd.DataFrame(confusion_matrix(Yt, predictions_2weeks, labels=unique_label), 
#                   index=['true:{:}'.format(x) for x in unique_label], 
#                   columns=['pred:{:}'.format(x) for x in unique_label]))


# In[83]:


#predictionsDF = pd.DataFrame(predictions_prob_2weeks, columns=["WillMiss", "WillNotMiss"])
#print(predictionsDF.head(1))
#print(predictionsDF.shape)


# In[84]:


#resultDF_PredProb = predictionsDF.join(testData)


# In[85]:


#list(resultDF_PredProb.columns)
#list(resultdf.columns)


# In[86]:


#  resultDF = resultDF_PredProb[[
#     'class_0',
#     'class_1',
#     'WillMiss',
#     'WillNotMiss',
#     'Hosp Code',
#     'Appointment ID',
#     'Account Number']]
    


# In[166]:


futureDataO = pd.read_csv(xpath + "/Appointments_10Aug2018_21Aug2018.csv")

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
futureData.dropna(inplace=True)


# In[167]:


import datetime
futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
print("test Data size {}".format(futureData.shape))
futureData['Date Of Appointment'] = pd.to_datetime(futureData['Date Of Appointment'])
futureData = futureData[futureData['Date Of Appointment'] >= datetime.date(year=2018,month=7,day=21) ]
print("test after date  Data size {}".format(futureData.shape))
FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]


# In[168]:


FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
print("FutureIDDF size {}".format(FutureIDDF.shape))
print("FutureIDDF cols {}".format(FutureIDDF.columns))


# In[169]:


#futureData.Postcode.astype(str).str.contains('No Value', regex=False, na=False)
#futureData.Postcode.str.contains('No Value', regex=False, na=False)
#print(futureData.Postcode)

#futureData['Patient Attended'] = 0


# In[170]:


futureDatap,Yf=prepareData(Appointdf=futureData)
#futureDatapDummy = Dummify(Appointdf=futureDatap, cat_vars=cat_vars)
futureDatapDummy = mcle.transform(futureDatap)
print("futureDatapDummy {}".format(futureDatapDummy.shape))

futureDatapDummy['AppointmentSequence'].fillna((Appointdft['AppointmentSequence'].mean()), inplace=True)
print('AppointmentSequence')
futureDatapDummy['Age'].fillna( Appointdft['Age'].mean() , inplace=True)
print('Age')
futureDatapDummy['IsFirstNonCancelledAppointment'].fillna( 0 , inplace=True)
print('IsFirstNonCancelledAppointment')
futureDatapDummy['Is Interpreter Required'].fillna(0, inplace=True)
print('Is Interpreter Required')
futureDatapDummy['MinTemp'].fillna((Appointdft['MinTemp'].mean()), inplace=True)
print('MinTemp')
futureDatapDummy['MaxTemp'].fillna((Appointdft['MaxTemp'].mean()), inplace=True)
print('MaxTemp')
futureDatapDummy['rainfall'].fillna( 0, inplace=True)
print('rainfall')
futureDatapDummy['SEIFAStatePercentile'].fillna( (Appointdft['SEIFAStatePercentile'].mean()), inplace=True)
print('SEIFAStatePercentile')
futureDatapDummy['PriorAttendanceRate'].fillna((Appointdft['PriorAttendanceRate'].mean()), inplace=True)
print('PriorAttendanceRate')
futureDatapDummy['Appoint_Book_Length'].fillna( (Appointdft['Appoint_Book_Length'].mean() ), inplace=True)
print('Appoint_Book_Length')

print("Null: {}".format(Appointdft.isnull().sum(axis=0)))

futureDatapDummy.dropna(inplace=True)



# In[171]:


#XtrainEmpty = Appointdf[0:0]
#print(XtrainEmpty.shape)
#print(len(XtrainEmpty.columns))
#print(len(set(XtrainEmpty.columns)))
#print((XtrainEmpty.columns))


# In[172]:


#print("concat columns futureDatapDummy {}".format(futureDatapDummy.shape))
#frames=[XtrainEmpty, futureDatapDummy]
#XtrainEmpty_cols = set(XtrainEmpty.columns) 
#print("XtrainEmpty_cols {}" .format(len(XtrainEmpty_cols)))
#futureDatapDummy_cols = set(futureDatapDummy.columns)
#comm_cols = XtrainEmpty_cols.intersection(futureDatapDummy_cols)
#print(len(comm_cols))


# In[173]:


#futureDatapDummy = pd.concat([df[common_cols] for df in frames], ignore_index=True, sort=False)
#print(futureDatapDummy.shape)
#futureDatapDummy = futureDatapDummy[list(comm_cols)]
#futureDatapDummy = pd.concat([XtrainEmpty,futureDatapDummy])
#print(futureDatapDummy.shape)


# In[174]:


#print(len(XtrainEmpty_cols))
#print(len(comm_cols))
#print(len(set(XtrainEmpty_cols - comm_cols)))

#for ac in list(set(XtrainEmpty_cols - comm_cols)):
#    futureDatapDummy[ac]=0
#print(futureDatapDummy.shape)


# In[175]:


print("after concat futureDatapDummy {}".format(futureDatapDummy.shape))
print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
#NullCols = futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist()
#for c in NullCols:
#    futureDatapDummy[c]=0
print(futureDatapDummy.columns[futureDatapDummy.isna().any()].tolist())
futureDatapDummy.head(1)


# In[176]:


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


# In[261]:


predcat_Future = catmodel.predict(futureDatapDummy)
predcat_prob_Future = catmodel.predict_proba(futureDatapDummy)


# In[177]:



#predictions_Future = logreg.predict(futureDatapDummy)
#predictions_prob_Future = logreg.predict_proba(futureDatapDummy)
#predxgs_Future = logreg.predict(futureDatapDummy)
#predxgs_prob_Future = logreg.predict_proba(futureDatapDummy)
#print(logreg.classes_)


# In[ ]:



print(predcat_Future)
print(predcat_prob_Future)

print(predxgs_Future.shape)
print(FutureIDDF.shape)


# In[262]:


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


# In[180]:


#FutureDF_PredProb = predictionsFutureDF.join(FutureIDDF)
print(FutureIDDF.shape)
print(FutureIDDF.columns)
print(predictionsFutureDF.shape)
print(predictionsFutureDFB.shape)


# In[263]:


FutureIDDF.to_csv(xpath + "/PredictionJuly2018.csv",index=True, header=True, na_rep="NA")
predictionsFutureDF.to_csv(xpath + "/ProbsJuly2018.csv",index=True, header=True, na_rep="NA")
predictionsFutureDFB.to_csv(xpath + "/ProbsJuly2018B.csv",index=True, header=True, na_rep="NA")
predictionsFutureDFNB.to_csv(xpath + "/ProbsJuly2018NB.csv",index=True, header=True, na_rep="NA")
predictionsFutureDFcat.to_csv(xpath + "/ProbsJuly2018cat.csv",index=True, header=True, na_rep="NA")


# In[264]:


print(FutureIDDF.isnull().sum())
print(FutureIDDF.isna().sum())
print(FutureIDDF[FutureIDDF['Date Of Appointment']==pd.Timestamp('')].shape)


# In[265]:


FutureIDDF.head(5)


# In[266]:


print(FutureIDDF.dtypes)


# In[267]:


#outcomerph = pd.read_csv(xpath + "Appointments_01Aug2018_07Aug2018.csv")


# In[268]:


#outcomerph.columns
#outcomerph=outcomerph.rename(columns = {'Patient Attended':'SourceStatus'})


# In[269]:


#pred_out = pd.merge(futureData, outcomerph, left_on=['Appointment ID'], right_on=['Appointment ID'] )


# In[270]:


#pred_outResult =  pd.merge(outcomerph, FutureIDDF, left_on=['Appointment ID'], right_on=['Appointment ID'], how ='inner' ) 

#print(pred_outResult.head())


# In[271]:


print(futureData.columns)

pd.crosstab(FutureIDDF['Attend'], futureData['Patient Attended'])

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(futureData['Patient Attended'], FutureIDDF['Attend'], labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))
print(classification_report(futureData['Patient Attended'], FutureIDDF['Attend'], labels=unique_label))


# In[272]:


pd.crosstab(FutureIDDF['AttendB'], futureData['Patient Attended'])
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(futureData['Patient Attended'], FutureIDDF['AttendB'], labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))

print(classification_report(futureData['Patient Attended'], FutureIDDF['AttendB'], labels=unique_label))


# In[274]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(futureData['Patient Attended'], FutureIDDF['Attendcat'], labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))

print(classification_report(futureData['Patient Attended'], FutureIDDF['Attendcat'], labels=unique_label))


# In[275]:


pd.crosstab(FutureIDDF['AttendNB'], futureData['Patient Attended'])
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(futureData['Patient Attended'], FutureIDDF['AttendNB'], labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))
print(classification_report(futureData['Patient Attended'], FutureIDDF['AttendNB'], labels=unique_label))


# In[ ]:





# In[324]:


mask = FutureIDDF[ (FutureIDDF['AttendB']==0) &  (FutureIDDF['Attend']==0) ] 
#    FutureIDDF["AttendALL"]
FutureIDDF["AllAttend"] = 1
#FutureIDDF.loc[(FutureIDDF['AttendB']==0) &  (FutureIDDF['Attend']==0), ["AllAttend"] ] = 0
FutureIDDF["MissProbAll"]  = (FutureIDDF["MissProb"] * 5.3 +  FutureIDDF["MissProbB"] * 2.07 + FutureIDDF['MissProbNB'] * 0.26 + FutureIDDF['MissProbcat'] * 1.00   )/(5.3 + 2.07 + 0.26 + 1.00)     
FutureIDDF.loc[(FutureIDDF['MissProbAll']>=0.5) , ["AllAttend"] ] = 0

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
unique_label = np.unique(y_test)
print(pd.DataFrame(confusion_matrix(futureData['Patient Attended'], FutureIDDF['AllAttend'], labels=unique_label), 
                   index=['true:{:}'.format(x) for x in unique_label], 
                   columns=['pred:{:}'.format(x) for x in unique_label]))

print(classification_report(futureData['Patient Attended'], FutureIDDF['AllAttend'], labels=unique_label))


# In[ ]:





# In[297]:


pred_outResult.to_csv(xpath + "/pred_outResultJuly2018B.csv",index=True, header=True, na_rep="NA")


# In[298]:


print(futureData['Patient Attended'].shape)
print(FutureIDDF["MissProbAll"].shape)

print(np.unique(futureData['Patient Attended']))


# In[299]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

probs = pd.DataFrame(  {0: FutureIDDF["MissProbAll"], 1: 1- FutureIDDF["MissProbAll"] })
# The magic happens here
import matplotlib.pyplot as plt
import scikitplot as skplt
skplt.metrics.plot_lift_curve( futureData['Patient Attended'], probs)
plt.title('Wt Average', loc='left')
plt.legend()
plt.show()

skplt.metrics.plot_lift_curve( futureData['Patient Attended'], predxgs_prob_Future)
plt.title('Logit', loc='left')
plt.legend()
plt.show()
skplt.metrics.plot_lift_curve( futureData['Patient Attended'], predictions_prob_Future)
plt.title('RF', loc='left')
plt.legend()
plt.show()
skplt.metrics.plot_lift_curve( futureData['Patient Attended'], predgnb_prob_Future)
plt.title('NaiveBayes', loc='left')
plt.legend()
plt.show()


# In[325]:


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
    futureData = futureData[futureData['Date Of Appointment'] >= datetime.date(year=2018,month=7,day=21) ]
    print("test after date  Data size {}".format(futureData.shape))
    FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
    
    FutureIDDF = futureData[['Date Of Appointment', 'Appointment ID', 'AccountNumber']]
    print("FutureIDDF size {}".format(FutureIDDF.shape))
    print("FutureIDDF cols {}".format(FutureIDDF.columns))
    
    futureDatap,Yf=prepareData(Appointdf=futureData)
    #futureDatapDummy = Dummify(Appointdf=futureDatap, cat_vars=cat_vars)
    futureDatapDummy = mcle.transform(futureDatap)
    print("futureDatapDummy {}".format(futureDatapDummy.shape))
    
    futureDatapDummy['AppointmentSequence'].fillna((Appointdft['AppointmentSequence'].mean()), inplace=True)
    print('AppointmentSequence')
    futureDatapDummy['Age'].fillna( Appointdft['Age'].mean() , inplace=True)
    print('Age')
    futureDatapDummy['IsFirstNonCancelledAppointment'].fillna( 0 , inplace=True)
    print('IsFirstNonCancelledAppointment')
    futureDatapDummy['Is Interpreter Required'].fillna(0, inplace=True)
    print('Is Interpreter Required')
    futureDatapDummy['MinTemp'].fillna((Appointdft['MinTemp'].mean()), inplace=True)
    print('MinTemp')
    futureDatapDummy['MaxTemp'].fillna((Appointdft['MaxTemp'].mean()), inplace=True)
    print('MaxTemp')
    futureDatapDummy['rainfall'].fillna( 0, inplace=True)
    print('rainfall')
    futureDatapDummy['SEIFAStatePercentile'].fillna( (Appointdft['SEIFAStatePercentile'].mean()), inplace=True)
    print('SEIFAStatePercentile')
    futureDatapDummy['PriorAttendanceRate'].fillna((Appointdft['PriorAttendanceRate'].mean()), inplace=True)
    print('PriorAttendanceRate')
    futureDatapDummy['Appoint_Book_Length'].fillna( (Appointdft['Appoint_Book_Length'].mean() ), inplace=True)
    print('Appoint_Book_Length')
    
    print("Null: {}".format(Appointdft.isnull().sum(axis=0)))
    
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
    FutureIDDF["MissProbAll"]  = (FutureIDDF["MissProb"] * 5.3 +  FutureIDDF["MissProbB"] * 2.07 + FutureIDDF['MissProbNB'] * 0.26 + FutureIDDF['MissProbcat'] * 1.00   )/(5.3 + 2.07 + 0.26 + 1.00)     
    FutureIDDF.loc[(FutureIDDF['MissProbAll']>=0.5) , ["AllAttend"] ] = 0
    return FutureIDDF


# In[327]:


pred_outResult = PredictFuture(xpath + "/Appointments_25Aug2018_31Aug2018.csv" )
pred_outResult.to_csv(xpath + "/pred_outResult__25Aug2018_31Aug2018.csv",index=True, header=True, na_rep="NA")
print("Future prediction completed!!")

