# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  

from gtabview import view
pd.set_option('display.max_columns', None) # Display any number of columns
pd.set_option('display.max_rows', 100) # Display any number of rows
pd.set_option('display.max_colwidth', -1)

raw_account = pd.read_csv('D:/Users/anandrathi/Documents/Work/InterView/RBL_70_30/RBL_70_30/raw_account_70.csv')
raw_data = pd.read_csv('D:/Users/anandrathi/Documents/Work/InterView/RBL_70_30/RBL_70_30/raw_data_70.csv')
raw_enquiry = pd.read_csv('D:/Users/anandrathi/Documents/Work/InterView/RBL_70_30/RBL_70_30/raw_enquiry_70.csv', low_memory=False)

raw_account.describe(include = 'all')
raw_data.describe() 
raw_enquiry.describe()
raw_account.columns[raw_account.isnull().any()].tolist()

raw_account.dt

view(raw_account)

row_ids = raw_account[raw_account["opened_dt"] != raw_account["dt_opened"]].index
erow_ids = raw_account[raw_account["opened_dt"] == raw_account["dt_opened"]].index
raw_account["opened_dt"].head(10)
raw_account["dt_opened"].head(10)

raw_account.columns
raw_account.isnull()

# List coloumns who have NA null values 
len(raw_account.columns[ pd.isnull(raw_account).sum() > 10 ])

Nullraw_account= raw_account[raw_account.columns[ pd.isnull(raw_account).sum() > 10 ]]
Nullraw_account.describe()
Nullraw_account.columns
view(Nullraw_account)

for cols in  raw_account.columns[ pd.isnull(raw_account).sum() > 10 ]:
    raw_account[cols].describe()

raw_account['opened_dt'].fillna( "01-01-1972", inplace=True)
raw_account['opened_dt'].isnull().sum()

raw_account[['opened_dt']] =  raw_account['opened_dt'].apply( lambda x :  x[0:6] + "-20" + x[:2])
raw_account['opened_dt'][raw_account['opened_dt'].str.len() > 11]
raw_account['opened_dt'].head()
pd.to_datetime(raw_account['opened_dt'],infer_datetime_format=True)
pd.to_datetime(raw_account['opened_dt'], format='%Y%m%d', errors='ignore')

sum(raw_account["typeofcollateral"].isnull() )
len(raw_account["typeofcollateral"][ raw_account["typeofcollateral"].isnull() == False])

raw_account['typeofcollateral'].unique()
raw_account['typeofcollateral'].fillna( 10.0, inplace=True)
raw_account['typeofcollateral'].unique()
raw_account["typeofcollateral"]  = raw_account["typeofcollateral"].astype('category')
raw_account["typeofcollateral"].describe()
# Impute 
raw_account["typeofcollateral"] 

