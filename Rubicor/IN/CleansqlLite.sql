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

DROP TABLE `InPatient_BedHistory_2018`;
ALTER TABLE t1_backup RENAME TO `InPatient_BedHistory_2018`;

Update `InPatient_BedHistory_2018`  SET  CurrentWard  = 'NoValue' where CurrentWard   = '[No Value]';
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),' ', '_') ;
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'#', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'+', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'?', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'&', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'(', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),')', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'/', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentWard  = replace(trim(CurrentWard),'\\', '');
Update `InPatient_BedHistory_2018`  SET  CurrentWard = replace(trim(CurrentWard), ',', '_');

Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = 'NoValue' where AdmitDiagnosis   = '[No Value]';
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),',', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'(', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),')', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'/', '_');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'#', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'+', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'?', '');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),'&', '_');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = replace(trim(AdmitDiagnosis),' ', '_');
Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis = replace(trim(AdmitDiagnosis), ',', '_');


Update `InPatient_BedHistory_2018`  SET  PatientType  = 'NoValue' where PatientType   = '[No Value]';
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),',', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'(', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),')', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'/', '_');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'#', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'+', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'?', '');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),'&', '_');
Update `InPatient_BedHistory_2018`  SET  PatientType  = replace(trim(PatientType),' ', '_');
Update `InPatient_BedHistory_2018`  SET  PatientType = replace(trim(PatientType), ',', '_');



--Update `InPatient_BedHistory_2018`  SET  GP  = replace(trim(GP),'[No Value]', 'NoValue');
--Update `InPatient_BedHistory_2018`  SET  GP  = replace(GP,',', '') where GP   = '%,%';
--Update `InPatient_BedHistory_2018`  SET  GP  = replace(trim(GP),' ', '_');
--Update `InPatient_BedHistory_2018`  SET  GP  = replace(trim(GP),'[No_Value]', 'NoValue');

Update `InPatient_BedHistory_2018`  SET  MentalHealthLegalStatus  = replace(trim(MentalHealthLegalStatus),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  ArrivalMeans  = replace(trim(ArrivalMeans),'[Unknown]', 'Unknown');


Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'(', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),')', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'/', '_');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'\\', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),' ', '_');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'#', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'+', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'?', '');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'&', '_');
Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),',', '_');


Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'(', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),')', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'/', '_');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'\\', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'#', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'%', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'+', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'?', '');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),'&', '_');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),' ', '_');
Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeType  = replace(trim(InpatientEpisodeType),',', '_');


Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'[Unknown]', 'Unknown');


Update `InPatient_BedHistory_2018`  SET  CodingStatus  = replace(trim(CodingStatus),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  CodingStatus  = replace(trim(CodingStatus),' ', '_');


Update `InPatient_BedHistory_2018`  SET  InpatientEpisodeStatus  = replace(trim(InpatientEpisodeStatus),' ', '_');

Update `InPatient_BedHistory_2018`  SET  AdmissionType  = replace(trim(AdmissionType),' ', '_');

Update `InPatient_BedHistory_2018`  SET  CareType  = replace(trim(CareType),' ', '_');


Update `InPatient_BedHistory_2018`  SET  DeathInRecoveryCount  = replace(trim(DeathInRecoveryCount),'[No Value]', 'NULL');

Update `InPatient_BedHistory_2018`  SET  ReferralSource  = replace(trim(ReferralSource),'/', '_');

Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '+', '');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '#', '');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '?', '');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '&', '_');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), '-', '_');
Update `InPatient_BedHistory_2018`  SET  InpatientLocationWardAtDischarge  = replace(trim(InpatientLocationWardAtDischarge), ' ', '_');

Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge = replace(trim(ServiceAtDischarge),'\\', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'#', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'+', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'?', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'&', '_');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'(', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),')', '');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'/', '_');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),'-', '_');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),' ', '_');
Update `InPatient_BedHistory_2018`  SET  ServiceAtDischarge  = replace(trim(ServiceAtDischarge),',', '_');


Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'\\', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'#', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'+', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'?', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'&', '_');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'(', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),')', '');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType),'/', '_');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), '-', '_');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), ' ', '_');
Update `InPatient_BedHistory_2018`  SET  DischargeDestinationType = replace(trim(DischargeDestinationType), ',', '_');

Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'#', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'+', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'?', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'&', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'(', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),')', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'/', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty),'\\', '');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), '-', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), ' ', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentSpecialty = replace(trim(CurrentSpecialty), ',', '_');


Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'[No Value]', 'NoValue');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'NoValue', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'#', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'+', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'?', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'&', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'(', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),')', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'/', '_');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService),'\\', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService), '-', '');
Update `InPatient_BedHistory_2018`  SET  CurrentService = replace(trim(CurrentService), ',', '_');

Update `InPatient_BedHistory_2018`  SET  AdmitDiagnosis  = lower(AdmitDiagnosis);



