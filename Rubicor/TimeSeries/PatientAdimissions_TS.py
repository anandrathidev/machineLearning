# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:19:12 2018

@author: Anand Rathi 
"""


import pandas as pd
import numpy as np
from fbprophet import Prophet
FPATHS = "C:/temp/DataScience/PatientAdimissions_TS/"
import datetime 
start = datetime.date(2010,1, 1) 

hods_dbo_tbledpresentations_init = pd.read_csv(FPATHS+ "hods_dbo_tbledpresentations_Header.csv")

hods_dbo_tbledpresentations_init.head(1)
hods_dbo_tbledpresentations_init.dtypes
list(hods_dbo_tbledpresentations_init.columns)
hods_dbo_tbledpresentations_init['ArrivalDateTime'] = pd.DatetimeIndex(hods_dbo_tbledpresentations_init['ArrivalDateTime'])
hods_dbo_tbledpresentations_init = hods_dbo_tbledpresentations_init[(hods_dbo_tbledpresentations_init['ArrivalDateTime'] > start) ]

print(start)
#Split Data Frame based on HospCode
HospCodeWiseDF = { hc:hods_dbo_tbledpresentations_init[hods_dbo_tbledpresentations_init["HospCode"] == hc]  for hc in  set(hods_dbo_tbledpresentations_init["HospCode"]) } 

HospCodeWiseDF.keys()

def ForecastArrivals(tsdataframe, ptext):
  tsdataframe = tsdataframe.rename(columns={'ArrivalDateTime': 'ds',
                          'Qty': 'y'})
  tsdataframe['ds'] = pd.DatetimeIndex(tsdataframe['ds'])
  tsdataframe['Qty']=1
  tsdataframe.dtypes
  tsdataframe = tsdataframe[["ds","y"]]
  tsdataframe = tsdataframe.set_index('ds')
  tsdataframe_5m =  tsdataframe.resample('15Min').sum()
  tsdataframe_5m["ds"] = tsdataframe_5m.index
  
  pat_model = Prophet(interval_width=0.95)
  pat_model.fit(tsdataframe_5m)
  future_dates =  pat_model.make_future_dataframe(periods=600, freq='H')
  future_dates.tail(100)
  forecast = pat_model.predict(future_dates)
  print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
  forecast.to_csv(FPATHS + "//ForteCast_" + ptext + ".csv" ) 

for hcode,prdItem in HospCodeWiseDF.items():
  ForecastArrivals(tsdataframe=prdItem , ptext=hcode)


hods_dbo_tbledpresentations_init['ArrivalDateTime'] = pd.DatetimeIndex(hods_dbo_tbledpresentations_init['ArrivalDateTime'])
hods_dbo_tbledpresentations_init['Qty']=1
hods_dbo_tbledpresentations_init.dtypes


## hods_dbo_tbledpresentations_init.resample('5Min').sum()
## hods_dbo_tbledpresentations_init.groupby(pd.TimeGrouper('5Min')).sum()


hods_dbo_tbledpresentations = hods_dbo_tbledpresentations_init.rename(columns={'ArrivalDateTime': 'ds',
                        'Qty': 'y'})

hods_dbo_tbledpresentations = hods_dbo_tbledpresentations[["ds","y"]]
hods_dbo_tbledpresentations = hods_dbo_tbledpresentations.set_index('ds')
hods_dbo_tbledpresentations_5m =  hods_dbo_tbledpresentations.resample('15Min').sum()
hods_dbo_tbledpresentations_5m["ds"] =  hods_dbo_tbledpresentations_5m.index

#fstplt = hods_dbo_tbledpresentations_5m.plot(figsize=(22, 22))
#fstplt.set_ylabel('Monthly Patients')
#fstplt.set_xlabel('Date')
#fstplt.show()
pat_model = Prophet(interval_width=0.95)
pat_model.fit(hods_dbo_tbledpresentations_5m)
future_dates =  pat_model.make_future_dataframe(periods=600, freq='H')
future_dates.tail(100)
forecast = pat_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

list(hods_dbo_tbledpresentations.columns)


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

