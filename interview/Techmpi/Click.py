# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:09:58 2018

@author: he159490
"""

# In[]: Imports
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

# In[]: Init 
filepath = "/home/he159490/DS/Kaggle/SantanderValue//"
filepath = "F:/DataScience/TECHM/"
#filepath = "D:/Users/anandrathi/Documents/Work/Kaggle/Santander/"
data_file = filepath + "sample_data_clicks_conversions_events.csv"

# In[]: Init 
xdata = pd.read_csv(data_file)
# explore columns &  Data Type 
print("Num Cols {}".format(len(list(xdata.columns))))
# explore dtypes
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

# explore dtypes
xdata.drop(columns=["initial_tracking_uuid", "billable"], inplace=True)

# explore columns &  Data Type 
print(xdata.head(1))
print(xdata['event_created_at'].head())
print(xdata['created_at'].head())

print("Num Cols {}".format(len(list(xdata.columns))))
# explore dtypes
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

# In[]: Init 
# transform types
kwargs = {"created_at" : pd.to_datetime(xdata["created_at"]) }
xdata = xdata.assign( **kwargs )
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

# In[]: Init 
kwargs = {"event_created_at" : pd.to_datetime(xdata["event_created_at"]) }
xdata = xdata.assign( **kwargs )
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))


# In[]: Init 
# transform data
def splitIP(x):
    if isinstance(x, str):
        ipl = x.split(".")
        ipl.append( ipl[0] + "." + ipl[1] )
        ipl.append( ipl[0] + "." + ipl[1]  + "." + ipl[2] )
        return  pd.Series(ipl, index=[  'net1', 'net2', 'net3', 'net4' , 'net1.net2',  'net1.net2.net3' ])
    else:
        return  pd.Series([x,x,x,x,x,x], index=[  'net1', 'net2', 'net3', 'net4' , 'net1.net2',  'net1.net2.net3' ])
#Create IP address dataframe    
ipDF = pd.DataFrame(xdata["ip_address_str"].apply( lambda x: splitIP(x) ))
#Join ip address Data frame
xdata=xdata.join(ipDF)

# In[]: Init 
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

# In[]: Init 

uniqData = xdata.nunique()

# In[]: Init 
gbdate = xdata.groupby([ 'created_at' ]).count()

gbdusDay = xdata.set_index('created_at').groupby(pd.TimeGrouper('D')).agg(['min','max','count','nunique'])


# In[]: Init 
xdata.groupby(['created_at', 'net1.net2.net3']).count()
xdata.groupby(['group']).agg(['min','max','count','nunique'])

















