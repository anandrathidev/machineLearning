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

#view(raw_account)

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
#view(Nullraw_account)

for cols in  raw_account.columns[ pd.isnull(raw_account).sum() > 10 ]:
    raw_account[cols].describe()



# Start Imputing 
# =============================================================================
#        high_credit_amt  amt_past_due  suitfiledorwilfuldefaultorwritte  \
# count  1.774540e+05     8.760000e+02  0.0                                
# mean   1.756104e+05     2.583151e+04 NaN                                 
# std    9.842643e+05     2.030680e+05 NaN                                 
# min    1.000000e+00     1.000000e+00 NaN                                 
# 25%    2.297500e+04     1.535000e+02 NaN                                 
# 50%    4.522350e+04     1.209500e+03 NaN                                 
# 75%    1.060000e+05     7.663250e+03 NaN                                 
# max    1.800000e+08     4.869309e+06 NaN                                 
# 
#        writtenoffandsettled  valueofcollateral   creditlimit       cashlimit  \
# count  7195.000000           2.234000e+03       4.885200e+04  35282.000000     
# mean   3.301598              2.112677e+06       7.552881e+04  20608.503401     
# std    2.461591              3.655474e+06       7.174109e+04  24856.481410     
# min    0.000000              6.000000e+03       1.000000e+00  1.000000         
# 25%    3.000000              4.734630e+05       3.100000e+04  7500.000000      
# 50%    3.000000              1.060815e+06       6.000000e+04  12500.000000     
# 75%    4.000000              2.542350e+06       1.000000e+05  27000.000000     
# max    99.000000             8.302000e+07       2.500000e+06  1000000.000000   
# 
#        emiamount  writtenoffamounttotal  writtenoffamountprincipal  \
# count  0.0        1408.000000            1246.000000                 
# mean  NaN         39417.474432           31259.667737                
# std   NaN         63328.574121           56515.988520                
# min   NaN         1.000000               1.000000                    
# 25%   NaN         8149.500000            5149.750000                 
# 50%   NaN         20622.500000           14244.000000                
# 75%   NaN         43255.000000           32269.500000                
# max   NaN         705013.000000          645547.000000               
# 
#        settlementamount  paymentfrequency  actualpaymentamount  \
# count  8.560000e+02      63893.000000      4.105300e+04          
# mean   2.736667e+04      2.996212          3.016616e+04          
# std    5.466986e+04      0.086954          3.870822e+05          
# min    4.320000e+02      1.000000          1.000000e+00          
# 25%    7.037500e+03      3.000000          3.110000e+03          
# 50%    1.500000e+04      3.000000          7.590000e+03          
# 75%    2.929750e+04      3.000000          1.800000e+04          
# max    1.126000e+06      3.000000          6.785322e+07          
# 
# =============================================================================
# drop        disputeremarkscode2, dateofentryforerrorordisputerema, disputeremarkscode1
#       dateofentryforerrorcode, errorcode, dateofentryforcibilremarkscode, emiamount
#        suitfiledorwilfuldefaultorwritte, suitfiledorwilfuldefaultorwritte

len(raw_account.columns    )
raw_account = raw_account.drop(['disputeremarkscode2', 'dateofentryforerrorordisputerema', 'disputeremarkscode1',
       'dateofentryforerrorcode', 'errorcode', 'dateofentryforcibilremarkscode', 'emiamount',
        'suitfiledorwilfuldefaultorwritte', 'suitfiledorwilfuldefaultorwritte'], axis=1)    
len(raw_account.columns    )
    
raw_account['opened_dt'].fillna( "01-01-1972", inplace=True)
raw_account['opened_dt'].isnull().sum()

raw_account[['opened_dt']] =  raw_account['opened_dt'].apply( lambda x :  x[0:6] + "-20" + x[:2])
raw_account['opened_dt'][raw_account['opened_dt'].str.len() > 11]
raw_account['opened_dt'].head()
raw_account['opened_dt'].describe()
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

sum(raw_account["high_credit_amt"].isnull() )
sum(raw_account["high_credit_amt"]==0 )
raw_account["high_credit_amt"] 
raw_account['high_credit_amt'].fillna( 0, inplace=True)


sum(raw_account["amt_past_due"].isnull() )
sum(raw_account["amt_past_due"]==0 )
sum(raw_account["amt_past_due"]>0 )
raw_account['amt_past_due'].fillna( 0, inplace=True)


sum(raw_account["writtenoffandsettled"].isnull() )
sum(raw_account["writtenoffandsettled"]==0 )
sum(raw_account["writtenoffandsettled"]>0 )
raw_account['writtenoffandsettled'].fillna( 0, inplace=True)


sum(raw_account["writtenoffamountprincipal"].isnull() )
sum(raw_account["writtenoffamountprincipal"]==0 )
sum(raw_account["writtenoffamountprincipal"]>0 )
raw_account['writtenoffamountprincipal'].fillna( 0, inplace=True)

sum(raw_account["valueofcollateral"].isnull() )
sum(raw_account["valueofcollateral"]==0 )
sum(raw_account["valueofcollateral"]>0 )
raw_account['valueofcollateral'].fillna( 0, inplace=True)

sum(raw_account["creditlimit"].isnull() )
sum(raw_account["creditlimit"]==0 )
sum(raw_account["creditlimit"]>0 )
raw_account['creditlimit'].fillna( 0, inplace=True)

sum(raw_account["creditlimit"].isnull() )
sum(raw_account["creditlimit"]==0 )
sum(raw_account["creditlimit"]>0 )
raw_account['creditlimit'].fillna( 0, inplace=True)

sum(raw_account["cashlimit"].isnull() )
sum(raw_account["cashlimit"]==0 )
sum(raw_account["cashlimit"]>0 )
raw_account['cashlimit'].fillna( 0, inplace=True)

sum(raw_account["writtenoffamounttotal"].isnull() )
sum(raw_account["writtenoffamounttotal"]==0 )
sum(raw_account["writtenoffamounttotal"]>0 )
raw_account['writtenoffamounttotal'].fillna( 0, inplace=True)




