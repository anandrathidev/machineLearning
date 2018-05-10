# -*- coding: utf-8 -*-

"""
Spyder Editor

This is a Program to failrly assign new doctors thier choice of stream file.
for Five terms
"""

import itertools
import pandas as pd
import numpy as np
import random
import copy
import logging
import sys, getopt
from collections import OrderedDict

def GenerateDummyData():
  PreviouslyAssignedStremas={}
  doctors = [ "Dr.{0:03d}".format(d) for d in range(1,201)]
  Streams = [ s for s in range(1,61)  ]
  choiceMatchRatingPoints={}
  DrChoose = []
  for d in doctors:
    optShuffle = copy.deepcopy(Streams)
    random.shuffle(optShuffle)
    choices={"Doctor":d}
    for i,o in enumerate(optShuffle):
      choices["Opt{0:03d}".format(i+1)] = o
    DrChoose.append(choices)
    PreviouslyAssignedStremas[d] = set()
    GdrsToBeIgnoredStreamWise = { s: set() for s in Streams}
    choiceMatchRatingPoints[d]=0.0
  return pd.DataFrame(DrChoose), doctors, Streams, PreviouslyAssignedStremas, GdrsToBeIgnoredStreamWise,choiceMatchRatingPoints

def GetUnique(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def PrepareData(prefMatrix, streamsDF):
  PreviouslyAssignedStremas={}
  doctors = list(prefMatrix.index)
  #Streams = [ s for s in range(1,61)  ]
  Streams = set(streamsDF["Stream"])

  choiceMatchRatingPoints={}
  DrChoose = []
  for d in doctors:
    #optShuffle = copy.deepcopy(Streams)
    #random.shuffle(optShuffle)
    print(d)
    choices={"Doctor":d}
    prefOptions = [int(choice) for choice in list(prefMatrix.loc[d]) if choice in Streams]
    #Streams.update(prefOptions)
  for d in doctors:
    choices={"Doctor":d}
    chkStream = set(list(Streams))
    prefOptions = [int(choice) for choice in list(prefMatrix.loc[d])]
    prefOptions.extend(Streams)
    prefOptions = list(OrderedDict.fromkeys(prefOptions))
    for i,o in enumerate(list(prefOptions)):
      choices["Opt{0:03d}".format(i+1)] = o
    DrChoose.append(choices)
    PreviouslyAssignedStremas[d] = set()
    GdrsToBeIgnoredStreamWise = { s: set() for s in Streams}
    choiceMatchRatingPoints[d]=0
  return pd.DataFrame(DrChoose), doctors, Streams, PreviouslyAssignedStremas, GdrsToBeIgnoredStreamWise,choiceMatchRatingPoints


def GetDocListBestPref(rnd,
                       stream,
                       drsToBeIgnored,
                       drchoice,
                       drscore,
                       unassignedDocs,
                       streamDoc,
                       choice,
                       choiceMatchRatingPoints,
                       doc_Streams,
                       minStream,
                       maxStream):
  bestPrefer=[]
  ###### First find list of doctors already assigned in this stream
  alreadyAssigned=[]
  list(StreamsDF.columns)
  if stream in streamDoc:
    alreadyAssigned = streamDoc[stream]

  ###### Is this stream already full ?
  if len(alreadyAssigned)>=maxStream:
    return

  ###### Give priority to doctors who were missed out on thier first choice earlier
  drscoreSorted = list(drscore.sort_values(by=["Score"], ascending=[True]).index)
  drtChoiceSet = set(list(drchoice[drchoice["Opt{0:03d}".format(choice)]==stream].index))
  drtChoiceList = [sd for sd in drscoreSorted if sd in drtChoiceSet]
  #logger.error(" GetDocListBestPref::stream={}  choice={} drtChoiceList= {} drscoreSorted={}".format(stream, choice, drtChoiceList, drscoreSorted))

  ###### Remove doctor who are already assigned to this stream in early occaions
  ###### Also Remove doctor who are already assigned to ANY stream in THIS occaions
  drtChoiceList = [dr for dr in drtChoiceList if ((not dr in drsToBeIgnored[stream]) and (dr in unassignedDocs)) ]
  logger.info(" GetDocListBestPref::drtChoiceList={}  ".format(len(drtChoiceList)))
  bestPrefer.extend(drtChoiceList)

  if (len(bestPrefer) >= maxStream-len(alreadyAssigned)):
    bestPrefer = bestPrefer[0:maxStream-len(alreadyAssigned)]

  if len(bestPrefer)>0:
    for doctor in bestPrefer:
      ###### Doctors to be ignored for that steam hence forth
      drsToBeIgnored[stream].add(doctor)

      #logger.info("Before GetDocListBestPref::unassignedDocs = {}".format(len(unassignedDocs)))
      ###### This Doctors is no longer availaible for this TERM
      unassignedDocs.discard(doctor)

      choiceMatchRatingPoints[doctor]+=100.0/choice
      drscore["Score"].loc[doctor]=choiceMatchRatingPoints[doctor]
      drscore["Choice_In_Term_{0:02d}".format(rnd)].loc[doctor]= "{0:03d}".format(choice)
      if stream in streamDoc:
        streamDoc[stream].append(doctor)
      else:
        streamDoc[stream]= [doctor]
      doc_Streams[doctor]+= " " + str(stream) + " "
  return

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("-c", "--pref", help="Prefrence File name")
  parser.add_argument("-p", "--path", help="Path")
  parser.add_argument(
    '-d', '--debug',
    help="log lots of debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
    )
  parser.add_argument(
    '-v', '--verbose',
    help="Be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
    )

  args = parser.parse_args()
  logging.basicConfig(level=args.loglevel)  
  inputPath=args.path
  preffile=args.pref
  #inputPath="C:/temp/DataScience/"
  #preffile="RMO Preference Selection.xlsx"
  logger = logging.getLogger('myapp')
  #hdlr = logging.FileHandler('C:/Users/he103788/temp.log')
  hdlr = logging.FileHandler( inputPath +  '/DocAssign_temp.log')
  logger.addHandler(hdlr)
  AssignRounds=[]
  AssignRoundsDFList=[]

  """main(inputPath=inputPath,
       preffile=preffile,
       AssignRoundsDFList=AssignRoundsDFList,
       AssignRounds=AssignRounds
       )
   """

  print ("Input path is {} Pref file is {}".format( inputPath, inputPath + "/" + preffile))
  prefmatPath = inputPath + "/" + preffile
  ###### Load excel sheet
  prefSheet = pd.ExcelFile(prefmatPath)
  prefMatDF_Init = prefSheet.parse(prefSheet.sheet_names[0], header=[0, 1])
  ###### extract list of Preference columns names
  #PrefList =  [colt for colt in list(prefMatDF_Init.columns) if colt[0] == 'From the Streams List, please place your preferences in order from 1 - 50']
  ###### Create a new dataframe prefMatDF with just doctor names
  prefMatDF = pd.DataFrame( {'Name':prefMatDF_Init[('Name', 'Open-Ended Response')]})
  list(prefMatDF_Init.columns)
  ###### add preference columns to new dataframe prefMatDF
  for colt in list(prefMatDF_Init.columns) :
    if (colt[0] == 'From the Streams List, please place your preferences in order from 1 - 50'):
      try:
        prefMatDF[colt[1]] = prefMatDF_Init[colt].astype(int)
      except ValueError as e:
        prefMatDF[colt[1]] = prefMatDF_Init[colt]

  ###### remove NAN values rows , that is doctors who selected non preferences
  prefMatNNDF = prefMatDF[~prefMatDF.isnull().any(axis=1)]
  prefMatDF = prefMatNNDF
  prefMatDF = prefMatDF.set_index('Name')

  ###### remove duplicate doctors
  prefMatDF = prefMatDF[~prefMatDF.index.duplicated(keep='first')]

  prefMatDF.to_csv(inputPath + "/" + "prefMatDF.csv", index=True)
  
  ###### Load Streams capacity
  StreamsDF = pd.read_csv(inputPath + "/" + "StreamSize.csv")
  
  ###### Initialize few data structures
  DrChooseDF, doctors, Streams, PreviouslyAssignedStremas, GdrsToBeIgnoredStreamWise, ChoiceMatchRatingPoints = PrepareData(prefMatDF, StreamsDF)

  ###### create dict for Max in each stream
  StreamsDF= StreamsDF.set_index('Stream')
  streamsMax = StreamsDF.to_dict()
  
  DrScoreDF = DrChooseDF[["Doctor"]]
  DrScoreDF["Score"] = 0
  DrScoreDF = DrScoreDF.set_index('Doctor')

  DrChooseDF = DrChooseDF.set_index('Doctor')
  list(DrChooseDF.columns)
  #choice1 = DrChooseDF.groupby(["Opt001"]).size()
  len(list(DrChooseDF.columns))

  Rounds = [1,2,3,4,5]
  minStream=1
  maxStream=5
  AssignRounds=[]
  AssignRoundsDFList=[]
  for rnd in Rounds:
    UnassignedDocs = set(doctors)
    doc_Streams={ doc: "" for doc in list(UnassignedDocs) }
    streamDoc={}
    MaxTry = len(UnassignedDocs)+1
    currTry=0
    DrScoreDF["Choice_In_Term_{0:02d}".format(rnd)]= "000"
    while(len(UnassignedDocs)>0):
      if(currTry>MaxTry):
         raise Exception('exiting from possible infinite loop possibliy bad data like doctors duplicate choice') 
      currTry+=1   
      logger.info(" Round {} len(UnassignedDocs)= {}".format(rnd, len(UnassignedDocs)))
      logger.info(" UnassignedDocs= {}".format(UnassignedDocs))

      for choice in range(1,len(list(DrChooseDF.columns))+1):
        for stream in Streams:
          maxStream=streamsMax['MaxIntake'][stream]
          logger.error(" Round {} choice {} drsToBeIgnored[{}]= {}".format(rnd, choice, stream , GdrsToBeIgnoredStreamWise[stream]))
          GetDocListBestPref(
                          rnd=rnd,
                          stream=stream ,
                         drsToBeIgnored=GdrsToBeIgnoredStreamWise,
                         drchoice=DrChooseDF,
                         drscore=DrScoreDF,
                         unassignedDocs=UnassignedDocs,
                         streamDoc=streamDoc,
                         choice=choice,
                         choiceMatchRatingPoints=ChoiceMatchRatingPoints,
                         doc_Streams=doc_Streams,
                         minStream=1,
                         maxStream=maxStream)
    for stream in streamDoc:
      for doc in streamDoc[stream]:
        PreviouslyAssignedStremas[doc].add(stream)
    doc_Streams["Term"]=rnd
    AssignRoundsDFList.append(doc_Streams)
    AssignRounds.append(streamDoc)

  # StreamsDF = pd.DataFrame( { 
  #     "Stream" : list({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 54, 55, 56, 57, 58, 59, 60, 66, 67, 68, 69, 70, 72, 75, 76, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 184, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208}),
  #     } )
  #StreamsDF["MaxIntake"] = 5
  #StreamsDF["MinIntake"] = 1
  
  AssignRoundsDF = pd.DataFrame(AssignRoundsDFList)
  AssignRoundsDF.to_csv(inputPath + "/" + "Allocation.csv", index=True)
  AssignRoundsDF.T.to_csv(inputPath + "/" + "TAllocation.csv", index=True)
  DrScoreDF.to_csv(inputPath + "/" + "DrFairnessScoreDF.csv", index=True)
  
  