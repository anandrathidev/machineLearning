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

xdata.shape[0]

############################################################
#################### explore dtypes ########################
############################################################
xdata.drop(columns=["initial_tracking_uuid", "billable"], inplace=True)

# Explore columns &  Data Type 
print(xdata.head(1))

print("conversion at event_created_at: {}".format( xdata['event_created_at'].head(5)))
print("created_at: {}".format( xdata['created_at'].head(5)))

print("Na in created_at {} ".format( 100*np.sum(xdata['created_at'].isna()) /  xdata.shape[0]))
print("Na in event_created_at {} ".format( 100*np.sum(xdata['event_created_at'].isna()) /  xdata.shape[0]))

print("Num Cols {}".format(len(list(xdata.columns))))
# explore dtypes
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

uniqData = xdata.nunique()
print(uniqData.sort_values())
print(xdata.shape)
print(np.sum(xdata.isna()))

# Drop rows where created_at is NA 
xdata['created_at'].isna()
print(" xdata.shape Before drop NA in created_at {}".format(xdata.shape)  )
xdata = xdata[xdata['created_at'].isna()==False]
print(" xdata.shape After drop NA in created_at {}".format(xdata.shape)  )


# In[]: Init 
# transform types
############################################################
#################### transform types ########################
############################################################
kwargs = {"created_at" : pd.to_datetime(xdata["created_at"]) }
xdata = xdata.assign( **kwargs )
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

kwargs = {"event_created_at" : pd.to_datetime(xdata["event_created_at"]) }
xdata = xdata.assign( **kwargs )
print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))


# In[]: Init 
#################################################################################
#################### Split IP address to network vs host  #######################
#################################################################################
def splitIP(x):
    if isinstance(x, str):
        ipl = x.split(".")
        ipl.append( ipl[0] + "." + ipl[1] )
        ipl.append( ipl[0] + "." + ipl[1]  + "." + ipl[2] )
        return  pd.Series(ipl, index=[  'net1', 'net2', 'net3', 'net4' , 'net1.net2',  'net1.net2.net3' ])
    else:
        return  pd.Series(["Unknown","Unknown","Unknown","Unknown","Unknown","Unknown"], index=[  'net1', 'net2', 'net3', 'net4' , 'net1.net2',  'net1.net2.net3' ])
#Create IP address dataframe    
ipDF = pd.DataFrame(xdata["ip_address_str"].apply( lambda x: splitIP(x) ))
#Join ip address Data frame
xdata=xdata.join(ipDF)


# In[]: Init 
#################################################################################
#################### Split IP address to network vs host  #######################
#################################################################################
def splitDomainNames(x):
    if isinstance(x, str):
        ds2=[]
        dsl = x.split(".")
        if len(dsl) >=2:
            ds2.append( dsl[-2]  )
            ds2.append( dsl[-2] + "." + dsl[-1] )
        elif len(dsl) ==1:
            ds2.append( dsl[-1]  )
            ds2.append( dsl[-1] + "." + dsl[-1] )
        elif len(dsl) ==0:
            ds2.append( "Unknown"  )
            ds2.append( "Unknown" + "." + "Unknown" )
            
        return  pd.Series(ds2, index=[  'referrerdomain', 'referrerdomainExt'])
    else:
        return  pd.Series(["Unknown","Unknown.Unknown"], index=[ 'referrerdomain', 'referrerdomainExt' ])
    
#Create IP address dataframe
dsDF = pd.DataFrame(xdata["referrer_domain"].apply( lambda x: splitDomainNames(x) ))
#Join ip address Data frame
xdata=xdata.join(dsDF)
xdata=xdata.drop(columns=["referrer_domain"])

# In[]: Init 

print("Cols {}".format(list(zip( list(xdata.columns), list(xdata.dtypes)))))

# In[]: Init 
#################################################################################
#################### Find low var data                    #######################
#################### find overfitting data                #######################
#################### replace NA values                  #######################
#################### Consolidate categories                #######################
#################################################################################

uniqData = xdata.nunique()
print(uniqData.sort_values())
print(xdata.shape)
print(np.sum(xdata.isna()))

xdata['event_created_at'].isna()


# In[]: Init 
## Get Rid of data which is low/high variance &  importance

np.sum(xdata.isna())

colstoDrop = set()
def PrintUniqVals(x,c):
    print("{} {}".format(c ,x[c].unique()))

PrintUniqVals(xdata,"revenue") ## array([nan])
colstoDrop.add("revenue") 
PrintUniqVals(xdata,"payout") ## array([nan])
colstoDrop.add("payout") 

PrintUniqVals(xdata,"conversion_step_id") ## conversion_step_id [ nan 947.] already acptured by tracing id 
colstoDrop.add("conversion_step_id") 

PrintUniqVals(xdata,"referral_charge") ## referral_charge [nan  0.] no value
colstoDrop.add("referral_charge") 

PrintUniqVals(xdata,"referral_charge") ## referral_charge [nan  0.] no value
colstoDrop.add("referral_charge") 

PrintUniqVals(xdata,"device_os") ## keep 
xdata["device_os"].fillna('Unknown', inplace=True )
PrintUniqVals(xdata,"device_os") ## chk

PrintUniqVals(xdata,"device_os_version") ## too many , no business use case 
colstoDrop.add("device_os_version") 

colstoDrop.add("user_agent_language") ## should we co relate this with Country ???

xdata["country_id"].dtype
np.unique(xdata["country_id"])
xdata["country_id"].fillna(0)
xdata = xdata.assign(country_id= xdata["country_id"].fillna(0))

PrintUniqVals(xdata,"browser") ## too many , no business use case 
xdata["browser"].fillna('Unknown', inplace=True )
#xdata.assign(["browser"] = np.unique(xdata["browser"].str.replace(r'[0-9\.]', '', regex=True).str.strip().str.replace(r'\s+', '_', regex=True))
xdata = xdata.assign(browser=xdata["browser"].str.replace(r'[0-9\.]', '', regex=True).str.strip().str.replace(r'\s+', '_', regex=True))

PrintUniqVals(xdata,"state") ## state
PrintUniqVals(xdata,"city") ## state
colstoDrop.add("state") ## should we co relate this with Country ???
colstoDrop.add("city") ## should we co relate this with Country ???

xdata["campaign_target_category"].dtype
xdata["campaign_target_category"].fillna('Unknown', inplace=True )

# In[]: Init 
xdata=xdata.drop(columns=list(colstoDrop))

for cat in ["device_os","tracking_type_id", "advertiser_manager_id", "campaign_target_category", "affiliate_manager_id",
            "browser","country_id", "affiliate_id", "campaign_id","net1", "net2", "net3",
          "net1.net2","referrerdomain", "referrerdomainExt"]:
  xdata[cat]=xdata[cat].astype("category")  
    
    

# In[]: Init 

uniqData = xdata.nunique()
print(uniqData.sort_values())
print(xdata.shape)
print(np.sum(xdata.isna()))


# In[]: Init 
xdata = xdata.set_index('created_at')
xdata = xdata.assign(click=1)

gbdate = xdata.groupby([ 'created_at' ]).count()
gbdusDay = xdata.set_index('created_at').groupby(pd.TimeGrouper('D')).agg(['min','max','count','nunique'])


# In[]: Init 
#xdata.groupby(['created_at', 'net1.net2.net3']).count()
#xdata.groupby(['group']).agg(['min','max','count','nunique'])


#campaign1D = xdata.groupby(['campaign_id', 'created_at']).resample('1D', label='right', closed='right').sum()
CampAffilate1D = xdata.groupby(['campaign_id', 'affiliate_id', 'referrerdomain']).resample('1D', label='right', closed='right')['click'].agg('sum').reset_index()
CampAffilate1D = CampAffilate1D.set_index('created_at')
CampAffilate1D.dtypes
CampAffilate1D["click"].plot()
CampAffilate1Dcumsum =  CampAffilate1D["click"].cumsum()
CampAffilate1Dcumsum.plot()


CampAffilate = pd.pivot_table(xdata, columns=['affiliate_id', 'referrerdomain'],  values=['click'], index=['campaign_id', 'created_at'],
aggfunc={'affiliate_id': [ np.count_nonzero, lambda x: len(x.unique())],
'referrerdomain': [ np.count_nonzero, lambda x: len(x.unique())]}).reset_index()
    
CampAffilate.columns = [ x[0] + x[1].replace("<lambda>","_UniqueCount")
               .strip().replace("count_nonzero","_Count")
               .strip() for x in CampAffilate.columns]
     
CPDFList =  [ xdata[xdata['campaign_id']==camp]
  for camp in list(np.unique(xdata['campaign_id']))]
    
CampAffilate.resample('1D', label='right', closed='right').sum()




