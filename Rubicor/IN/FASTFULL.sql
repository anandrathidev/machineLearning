declare @MyDate as varchar(256) = '20180101'

;With TheatreOperation AS
(
SELECT
  dOp.[AccountNumber]                   AS [AccountNumber] 
 ,dLoc.TheatreLocationUID                   AS [TheatreLocationUID]
,dOp.[HospCode]                        AS [OperationHospitalCode]
,fOp.[EpisodeUID]                                     AS OperationEpisodeUID
  ,CAST(CASE WHEN dActualStartDate.DateValue < '1900-JAN-01' THEN NULL ELSE dActualStartDate.DateValue END AS DATETIME)  
                + CASE WHEN fOp.ActualTheatreStartTimeUID < 0  
                           THEN NULL  
                           ELSE CONVERT(DATETIME, dOpActualTime.[24HourTime])  
                     END                       AS [OperationActualStartDatetime] 
  ,CAST(CASE WHEN dTheatreEnd.DateValue < '1900-JAN-01' THEN NULL ELSE dTheatreEnd.DateValue END AS DATETIME)  
                + CASE WHEN fOp.TheatreEndTimeUID < 0  
                           THEN NULL  
                           ELSE CONVERT(DATETIME, dOpEndTime.[24HourTime])  
                     END                       AS [OperationEndDatetime]
,CAST(CASE WHEN dRecovStartDate.DateValue < '1900-JAN-01' THEN NULL ELSE dRecovStartDate.DateValue END AS DATETIME)  
                + CASE WHEN fOp.RecoveryStartTimeUID < 0  
                           THEN NULL  
                           ELSE CONVERT(DATETIME, dRecovStartTime.[24HourTime])  
                     END                       AS [RecoveryStartDatetime] 
,CAST(CASE WHEN dRecovEndDate.DateValue < '1900-JAN-01' THEN NULL ELSE dRecovEndDate.DateValue END AS DATETIME)  
                + CASE WHEN fOp.RecoveryEndTimeUID < 0  
                           THEN NULL  
                           ELSE CONVERT(DATETIME, dRecovEndTime.[24HourTime])  
                     END                       AS [RecoveryEndDatetime] 
,dOp.[DeathInRecovery]                   AS [DeathInRecoveryCount]
,fOp.[DurationOfOperation]               AS [OperationDurationMins] 


FROM [BI_DW].[THEATRE].[factTheatreOperation]      AS fOp 
  INNER JOIN [BI_DW].[THEATRE].[dimTheatreOperation] AS dOp 
       ON fOp.OperationUID = dOp.OperationUID 
  INNER JOIN [BI_DW].[THEATRE].[dimTheatreLocation] AS dLoc 
         ON fOp.TheatreLocationUID = dLoc.TheatreLocationUID 
  INNER JOIN [BI_DW].[SHARED].[dimDate]            AS dActualStartDate 
     ON fOp.ActualTheatreStartDateUID = dActualStartDate.DateUID 
  INNER JOIN [BI_DW].[SHARED].[dimTime]            AS dOpActualTime 
    ON fOp.ActualTheatreStartTimeUID = dOpActualTime.TimeUID 
  INNER JOIN [BI_DW].[SHARED].[dimDate]            AS dTheatreEnd 
     ON fOp.TheatreEndDateUID = dTheatreEnd.DateUID 
  INNER JOIN [BI_DW].[SHARED].[dimTime]            AS dOpEndTime 
    ON fOp.TheatreEndTimeUID = dOpEndTime.TimeUID 
  INNER JOIN [BI_DW].[SHARED].[dimDate]            AS dRecovStartDate 
       ON fOp.RecoveryStartDateUID = dRecovStartDate.DateUID 
  INNER JOIN [BI_DW].[SHARED].[dimTime]            AS dRecovStartTime 
    ON fOp.RecoveryStartTimeUID = dRecovStartTime.TIMEUID 
    INNER JOIN [BI_DW].[SHARED].[dimDate]            AS dRecovEndDate 
       ON fOp.RecoveryEndDateUID = dRecovEndDate.DateUID 
  INNER JOIN [BI_DW].[SHARED].[dimTime]            AS dRecovEndTime 
     ON fOp.RecoveryEndTimeUID = dRecovEndTime.TIMEUID 

WHERE fOp.[IS_CURRENT] = 1 
    AND fOp.[IS_DELETED] = 0 

          AND
   (
    ( fOp.[RecordCount]  != 0    AND  fOp.[CancelledOperation] = 0   AND dActualStartDate.DateValue  > '1900-JAN-01' ) 
  OR
    ( fOp.[RecordCount]  != 1 AND fOp.[CancelledOperation] = 0  AND  dTheatreEnd.DateValue > '1900-JAN-01'  AND dActualStartDate.DateValue < '1900-JAN-01'  )
)



       AND fOp.ActualTheatreStartDateUID >=  @MyDate

)


SELECT 
-- Site detail 
      HospSite.[HospCode] as HospCode
      ,HospSite.[HospitalName] AS HospitalName
      ,FIE.[EpisodeUID]  as admissionEpisodeUID
      ,AdmitDate.[DateValue] as DateAtAdmission
      ,died.[AdmitDiagnosis]  as AdmitDiagnosis
      ,died.[GP] 
      ,FIE.AgeAtAdmissionUID as Age
      ,FIE.PatientUID 

-- Episode details - not connected to admission or discharge 

    ,die.[CareType]  as CareType
    ,die.[PatientType] as PatientType
    ,die.[AdmissionType] as AdmissionType
    ,die.[InpatientEpisodeStatus]  as InpatientEpisodeStatus
    ,die.SourceCodingStatus  as CodingStatus
    ,die.[InpatientEpisodeType]  as InpatientEpisodeType
    ,die.[ReferralSource]  as ReferralSource
    ,die.[ArrivalMeans]  as ArrivalMeans
    ,die.MentalHealthLegalStatus  as MentalHealthLegalStatus
    ,die.HasEmergencyReadmitWithin28Days as HasEmergencyReadmitWithin28Days
    ,die.IsEmergencyReadmitWithin28Days  as IsEmergencyReadmitWithin28Days
    ,die.IsBirth as IsBirth
    ,die.IsProcedureOnDayOfAdmission  as IsProcedureOnDayOfAdmission
    ,die.IsInpatientDeath  as IsInpatientDeath
    ,die.IsPrivatelyInsured as IsPrivatelyInsured
    ,die.[IsOrganProcurement]  as IsOrganProcurement
    ,CASE WHEN HospitalAcquiredDiagnosisEpisode = 1 tHEn 1 else 0 end   as HasChadxCondition
   , op.[AccountNumber] as OperationAccountNumber
      ,op.[TheatreLocationUID]
      ,[OperationHospitalCode] as  OperationHospitalCode
      ,OperationEpisodeUID as OperationEpisodeUID
      ,DischDate.[DateValue] as  DateAtDischarge
      ,op.[OperationActualStartDatetime]  as OperationActualStartDatetime
      ,op.OperationEndDatetime  as OperationEndDatetime
      ,op.RecoveryStartDatetime as RecoveryStartDatetime
      ,op.RecoveryEndDatetime as RecoveryEndDatetime
      ,op.DeathInRecoveryCount as DeathInRecoveryCount
      ,op.OperationDurationMins as OperationDurationMins

      ,bh.[Ward]
      ,bh.[Room]
      ,bh.[InpatientLocationUID]
      ,bh.[BedType]
      ,bh.[TimeIn]
      ,bh.[TimeOut]
      ,bh.[OnLeave]    
      ,bh.[CancelledAdmission] 
      ,bh.[OnStandby]
            ,DischTime.[12HourTime]   as TimeAtDischarge
            ,DischClinician.[MDCode]  as ClinicianAtDischarge
            ,DischSpec.[SpecialtyCode]  as SpecialtyAtDischarge 
            ,IL.Ward as InpatientLocationWardAtDischarge
            ,DischService.[Service]  as ServiceAtDischarge
            ,DIE.[DischargeDestinationType]  as DischargeDestinationType

-- Current detail 
            ,CurrentSpec.[Specialty]  as CurrentSpecialty
            ,CurrentWard.[Ward]  as CurrentWard
            ,CurrentService.[Service]   as CurrentService
            ,CurrentService.[HealthService]   as CurrentHealthService

  FROM 
  --[IDEA].[INPATIENT].[AdmittedPatient]  AS inad
[BI_DW].[INPATIENT].[factInpatientEpisode] AS FIE  WITH(NOLOCK)
INNER JOIN BI_DW.INPATIENT.dimInpatientEpisode AS DIE ON DIE.InpatientEpisodeUID = FIE.InpatientEpisodeUID
INNER JOIN BI_DW.INPATIENT.dimInpatientEpisodeDetail AS DIED ON DIED.InpatientEpisodeDetailUID = FIE.InpatientEpisodeDetailUID
INNER JOIN [BI_DW].[SHARED].[dimSite] AS HospSite   WITH(NOLOCK)  ON HospSite.SiteUID = fie.SiteUID 
INNER JOIN BI_DW.INPATIENT.DimInpatientLocation AS IL WITH(NOLOCK)      ON IL.InpatientLocationUID = FIE.InpatientLocationWardAtDischargeUID

INNER JOIN BI_DW.SHARED.dimDate AS AdmitDate  WITH(NOLOCK)  ON AdmitDate.DateUID = fie.DateAtAdmissionUID 
INNER JOIN BI_DW.SHARED.dimTime AS DischTime  WITH(NOLOCK)  ON DischTime.TIMEUID = fie.TimeAtDischargeUID 
INNER JOIN BI_DW.SHARED.dimDate AS DischDate WITH(NOLOCK)  ON DischDate.DateUID = fie.DateAtDischargeUID 
INNER JOIN BI_DW.SHARED.dimSpecialty AS DischSpec  WITH(NOLOCK) ON DischSpec.SpecialtyUID = fie.SpecialtyAtDischargeUID

       INNER JOIN [BI_DW].[INPATIENT].[dimInpatientLocation]  AS DischWard WITH(NOLOCK)  ON DischWard.InpatientLocationUID = fie.InpatientLocationWardAtDischargeUID
       INNER JOIN [BI_DW].[SHARED].[dimClinician] AS DischClinician WITH(NOLOCK) ON DischClinician.ClinicianUID = fie.ClinicianAtDischargeUID 
       INNER JOIN [BI_DW].[SHARED].[dimService] AS DischService WITH(NOLOCK)  ON DischService.ServiceUID =  fie.ServiceSpecialtyAtDischargeUID


-- Current Details
       INNER JOIN BI_DW.SHARED.dimSpecialty AS CurrentSpec WITH(NOLOCK) ON CurrentSpec.SpecialtyUID = fie.SpecialtyUID
       INNER JOIN IDEA.SHARED.dimPatient AS CurrentPatient WITH(NOLOCK)  ON CurrentPatient.PatientUID = fie.PatientUID
       INNER JOIN [BI_DW].[INPATIENT].[dimInpatientLocation]  AS  CurrentWard WITH(NOLOCK) ON CurrentWard.InpatientLocationUID = fie.InpatientLocationWardUID
       INNER JOIN [BI_DW].[SHARED].[dimClinician] AS CurrentClinician WITH(NOLOCK) ON CurrentClinician.ClinicianUID = fie.ClinicianUID 
       INNER JOIN [BI_DW].[SHARED].[dimService] AS CurrentService WITH(NOLOCK)  ON CurrentService.ServiceUID =  fie.ServiceSpecialtyUID

LEFT JOIN [BI_DW].[INPATIENT].[BedHistory] AS bh ON bh.EpisodeUID = FIE.EpisodeUID 
LEFT JOIN  TheatreOperation AS op ON  op.OperationEpisodeUID = FIE.EpisodeUID
  WHERE  
  1=1
AND  FIE.[DateAtDischargeUID]  >= @MyDate
 
