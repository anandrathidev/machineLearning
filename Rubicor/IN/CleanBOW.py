# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:48:49 2018

@author: he159490
"""


import pandas as pd
import numpy as np

filepath = "F:/DataScience/INPatients/"

db_file = filepath + "InPatient_BedHistory_2018.db"

csvfile = filepath + "InPatient_BedHistory_2018.csv"

import sqlite3
conn = sqlite3.connect( db_file )
c = conn.cursor()

df = pd.read_csv(csvfile)
try:
    c.execute("""DROP TABLE `InPatient_BedHistory_2018`;""")
except :
 pass    

try:
    df.to_sql('InPatient_BedHistory_2018', con=conn)
except :
 pass    

c.execute("""
CREATE TABLE t1_backup AS SELECT `HospCode` , `admissionEpisodeUID` , `DateAtAdmission` , 
`AdmitDiagnosis` ,  `Age` , `PatientUID` , `CareType` , `PatientType` , `AdmissionType` , `InpatientEpisodeStatus` , `CodingStatus` , 
`InpatientEpisodeType` , `ReferralSource` , `ArrivalMeans` , `MentalHealthLegalStatus` , `HasEmergencyReadmitWithin28Days` , 
`IsEmergencyReadmitWithin28Days` , `IsBirth` , `IsProcedureOnDayOfAdmission` , `IsInpatientDeath` , `IsPrivatelyInsured` , 
`IsOrganProcurement` , `HasChadxCondition` , `OperationAccountNumber` , `TheatreLocationUID` , `OperationHospitalCode` , 
`OperationEpisodeUID` , `DateAtDischarge` , `OperationActualStartDatetime` , `OperationEndDatetime` , `RecoveryStartDatetime` , 
`RecoveryEndDatetime` , `DeathInRecoveryCount` , `OperationDurationMins` , `Ward` , `Room` , `InpatientLocationUID` , `BedType` , 
`TimeIn` , `TimeOut` , `OnLeave` , `CancelledAdmission` , `OnStandby` , `TimeAtDischarge` , `ClinicianAtDischarge` , 
`SpecialtyAtDischarge` , `InpatientLocationWardAtDischarge` , `ServiceAtDischarge` , `DischargeDestinationType` , `CurrentSpecialty` , 
`CurrentWard` , `CurrentService`   from `InPatient_BedHistory_2018`;
""")

c.execute("""DROP TABLE `InPatient_BedHistory_2018`;""")
c.execute("ALTER TABLE t1_backup RENAME TO `InPatient_BedHistory_2018`; ")

c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = 'NoValue' where CurrentWard   = '[No Value]'; ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),' ', '_') ; ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentWard = replace(trim(CurrentWard), ',', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = 'NoValue' where AdmitDiagnosis   = '[No Value]'; ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),',', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis = replace(trim(AdmitDiagnosis), ',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = 'NoValue' where PatientType   = '[No Value]'; ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),',', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  PatientType = replace(trim(PatientType), ',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  MentalHealthLegalStatus  = replace(trim(MentalHealthLegalStatus),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ArrivalMeans  = replace(trim(ArrivalMeans),'[Unknown]', 'Unknown'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'%', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'[Unknown]', 'Unknown'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  CodingStatus  = replace(trim(CodingStatus),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CodingStatus  = replace(trim(CodingStatus),' ', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeStatus  = replace(trim(InpatientEpisodeStatus),' ', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  AdmissionType  = replace(trim(AdmissionType),' ', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  CareType  = replace(trim(CareType),' ', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  DeathInRecoveryCount  = replace(trim(DeathInRecoveryCount),'[No Value]', 'NULL'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'/', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '-', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), ' ', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge = replace(trim(ServiceAtDischarge),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'-', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'&', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), '-', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), ' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), ',', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'&', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), '-', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), ' ', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), ',', '_'); ")


c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'[No Value]', 'NoValue'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'NoValue', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'#', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'+', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'?', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'&', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'(', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),')', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'/', '_'); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'\\', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService), '-', ''); ")
c.execute("Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService), ',', '_'); ")

c.execute("Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = lower(AdmitDiagnosis); ")

df = pd.read_sql_query("SELECT * FROM InPatient_BedHistory_2018", conn)

print(df.dtypes)

#print(df.columns)



##UIDS will go away
deleteList = [
'admissionEpisodeUID', 
'PatientUID', 
'OperationAccountNumber', 
'OperationEpisodeUID',
] 

##date_time
DateTimeList = [
'DateAtAdmission', 
'OperationActualStartDatetime', 
'OperationEndDatetime',
'RecoveryStartDatetime', 
'RecoveryEndDatetime', 
'TimeAtDischarge' 
]


# Continues 
ContinuesList = [
'Age', 
'OperationDurationMins' 
]

#BOW
BOW = [
'AdmitDiagnosis', 
]

#Binary
binaryList = [
'HasEmergencyReadmitWithin28Days', 
'IsEmergencyReadmitWithin28Days',
'IsBirth', 
'IsProcedureOnDayOfAdmission', 
'IsInpatientDeath',
'IsPrivatelyInsured', 
'IsOrganProcurement', 
'HasChadxCondition',
'CancelledAdmission',
'OnLeave', 
'OnStandby', 
]

# Categorycall
CategoryList = [
'HospCode', 
'CareType', 
'PatientType', 
'AdmissionType',
'InpatientEpisodeStatus', 
'CodingStatus', 
'InpatientEpisodeType',
'ArrivalMeans', 
'MentalHealthLegalStatus',
'ReferralSource', 
'TheatreLocationUID', 
'OperationHospitalCode',
'DateAtDischarge',
'BedType', 
'ClinicianAtDischarge',
'SpecialtyAtDischarge', 
'InpatientLocationWardAtDischarge',
'ServiceAtDischarge', 
'DischargeDestinationType', 
'CurrentSpecialty',
'DeathInRecoveryCount',
'CurrentService'
]


#Calculate 
CalculateRemoveList = [
'TimeOut' ,
'TimeIn',
'Ward', 
'Room', 
'InpatientLocationUID',
'CurrentWard', 
]

def Transform(df, DateTimeList , ContinuesList, binaryList , CategoryList ):

    for col in DateTimeList:
        print("DateTimeList: Convert {} to np.datetime64".format(col) )
        df[col] = df[col].astype(np.datetime64)
    
    for col in CategoryList :
        print("CategoryList: Convert {} to category".format(col) )
        df[col] = df[col].astype('category')
    
    for col in binaryList :
        print("binaryList: Convert {} to int8".format(col) )
        df[col] = df[col].fillna(0)
        df[col] = df[col].astype(np.int8)
    
    for col in ContinuesList:
        print("ContinuesList: Convert {} to int16".format(col) )
        df[col] = df[col].astype(np.int16)
    
Transform(df=df, 
          DateTimeList=DateTimeList, 
          ContinuesList=ContinuesList, 
          binaryList=binaryList, 
          CategoryList=CategoryList)


print(df.dtypes)
CancelledAdmission = df["CancelledAdmission"]
DeathInRecoveryCount = df["DeathInRecoveryCount"].unique()

print(DeathInRecoveryCount )
