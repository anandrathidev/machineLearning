# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:05:58 2017

@author: rb117
"""

import pandas
from rpy2 import robjects
from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects.pandas2ri import py2ri, ri2py
import io  
import rpy2.rinterface as rinterface

base = importr('base')
stats = importr('stats')

testdf=pandas.read_csv('C:/Users/rb117/Documents/work/POC_analytics/Data/DATAMARTv2.tsv', sep = '\t', encoding = 'iso-8859-1', index_col=0)
testdf.describe()
list(testdf)
testdf.dtypes
len(testdf['first_name'])
first_name = robjects.StrVector(testdf['first_name'])
len(first_name )

rinterface.globalenv["campaign_result"]  = rinterface.baseenv['as.character'](rinterface.StrSexpVector( testdf['campaign_result'] ) ) 
rinterface.globalenv["campaign_result"]  = rinterface.baseenv['as.character'](rinterface.StrSexpVector( testdf['campaign_result'] ) ) 

rinterface.globalenv.keys()

rDataframe = rinterface.baseenv["data.frame"]
rColNames = rinterface.baseenv["names"]
rclass = rinterface.baseenv["class"]

rinterface.globalenv["testDF"] =   rDataframe( campaign_result = rinterface.globalenv["campaign_result"] , stringsAsFactors =  rinterface.BoolSexpVector((False, )) )
pytestDFDF3 = rinterface.globalenv["testDF"] 
pycolnames = rclass(pytestDFDF3)
pytestDFDF4  = robjects.DataFrame(pytestDFDF3)

pytestDFDF2 = robjects.r(" data.frame( campaign_result = campaign_result , stringsAsFactors = F )")

pytestDFDF  = robjects.DataFrame(rinterface.globalenv["testDF"])
len(pytestDFDF.colnames)

pycolnames = rclass(pytestDFDF)

pytestDFDF.slots("colnames")
str(pycolnames)
