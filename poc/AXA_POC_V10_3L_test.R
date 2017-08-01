
list.of.packages <- c( "lubridate","ggplot2","MASS","dplyr","e1071","ROSE","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest", "lift", "plotROC")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


c( "lubridate","ggplot2","MASS","dplyr","e1071","ROSE","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest")



#install.packages('missForest')

library(plotROC)
library(lubridate)
library(ggplot2)
library(MASS)
library(dplyr)
library(e1071)
library(ROSE)

library(caret)
library(caretEnsemble)
library(MLmetrics)
library(pROC)
library(ROCR)
library(reshape)
library(cluster)
library(fpc)
library(missForest)
#install.packages('mlbench')
library(mlbench)

# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************
testdatamart_init <- read.csv('C:/Users/rb117/Documents/work/POC_analytics/Data/poc_DATAMART_validation_set.tsv', stringsAsFactors = FALSE, sep = '\t')
str(testdatamart_init)

# ***************************************************************************
testdatamart <- testdatamart_init
#testdatamart$adress <- testdatamart_init$name

#### . . . .   add new data types   ----
testdatamart$FirstNameLen <- nchar(testdatamart$first_name,  allowNA = T)
testdatamart$LastNameLen <- nchar(testdatamart$name,  allowNA = T)

testdatamart$adressLen <- nchar(testdatamart$adress,  allowNA = T)

#### . . . .   change data types   ----
testdatamart$date_of_birth <- as.Date(testdatamart$date_of_birth,format='%d/%m/%Y')
#testdatamart$campaign_result <- as.factor(testdatamart$campaign_result)
testdatamart$gender <- as.factor(testdatamart$gender)
testdatamart$zipcode <- as.factor(testdatamart$zipcode)
testdatamart$city <- as.factor(testdatamart$city)
testdatamart$date_of_contrat_subscription <-  as.Date(testdatamart$date_of_contrat_subscription,format='%d/%m/%Y')
testdatamart$phone_number_used <- as.factor(testdatamart$phone_number_used)
testdatamart$other_phone_fiability <- as.factor(testdatamart$other_phone_fiability)
testdatamart$phone_type <- as.factor(testdatamart$phone_type)

#insurance_contrat_value
#insurance_contrat_value

testdatamart_clean <-  subset(testdatamart, select=-c(first_name, name, adress, zipcode, city))
str(testdatamart_clean)
summary(testdatamart_clean)


#impute other_phone_fiability 
summary(testdatamart_clean$other_phone_fiability)
testdatamart_clean$other_phone_fiability[testdatamart_clean$other_phone_fiability==""] <- 'plausible'

summary(datamart_clean$phone_type)

#### . . . .  dob in weeks    ----
testdatamart_clean$age_in_weeks <-  as.numeric(difftime(as.Date(Sys.Date()) , testdatamart_clean$date_of_birth , units = c("weeks")))
testdatamart_clean$doc_weeks  <- as.numeric(difftime( as.Date(Sys.Date()), testdatamart_clean$date_of_contrat_subscription, units = c("weeks")))


testdatamart_campaign_result  <- subset(testdatamart_clean, select = -c(date_of_contrat_subscription, FirstNameLen,
                                                          date_of_birth ) ) 


# ***************************************************************************
#                   TEST MODELING   ---- ENSEMBLE
# ***************************************************************************



mytrans <- function(x) {
  #return(log(x))) ## 
  #return(1/(1+exp(-x))) ## Better than logx(x)
  #return( sqrt(x)) ## same as sigmoid
  return(x) ## 
}

dummyFyTest <- function(inDF) {
  xtrain <- inDF
  xtrain$age_in_weeks <- mytrans(scale(as.numeric(xtrain$age_in_weeks)))
  xtrain$doc_weeks  <-  mytrans(scale(as.numeric(xtrain$doc_weeks)))
  xtrain$city_population <-  mytrans(scale(xtrain$city_population))
  xtrain$insurance_contrat_value <-  mytrans(scale(xtrain$insurance_contrat_value))
  #xtrain$age_when_sign <- log(xtrain$age_when_sign)
  #previous_na_action <- options('na.action')
  #options(na.action='na.pass')
  xtrain_dummy <- as.data.frame( model.matrix( ~. , xtrain ))
  xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  
  #clus <- kmeans(xtrain_dummy, centers=5)
  # Fig 01
  #plotcluster(xtrain_dummy, clus$cluster)
  #xtrain_dummy$k <- clus$cluster
  #xtrain_dummy$k <- as.factor(xtrain_dummy$k)
  #xtrain_dummy <- as.data.frame( model.matrix( ~. , xtrain_dummy))
  #xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  #colnames(xtrain_dummy)[5] <- "phone_number_usedAXA_et_CEDRICOM"
  names(xtrain_dummy)[names(xtrain_dummy) == 'phone_number_usedAXA et CEDRICOM'] <- 'phone_number_usedAXA_et_CEDRICOM'
  names(xtrain_dummy)[names(xtrain_dummy) == 'other_phone_fiabilitynon trouve'] <- 'other_phone_fiabilitynon_trouve'
  #colnames(xtrain_dummy)[8] <- "other_phone_fiabilitynon_trouve"
  #xtrain_dummy[is.na(xtrain_dummy),]
#table(xtrain_dummy$k,Train_campaign_result$campaign_result )
  return(xtrain_dummy)
  }

#xtrain.pca <- prcomp(xtrain, center = F, scale. = F) 
#plot(xtrain_dummy.pca, type = "l")
#names(xtrain_dummy.pca)





#..... Remove ID column
sum(is.na(testdatamart))

sum(is.na(testdatamart_campaign_result))
testdatamart_campaign_result[which(is.na(testdatamart_campaign_result)),]

sapply(testdatamart_campaign_result, function(x) sum(is.na(x)))
testdatamart_campaign_result_NULL <-  testdatamart_campaign_result[is.na(testdatamart_campaign_result),]

id_contrat <- testdatamart_campaign_result$id_contrat


#..... Impute null values 
testdatamart_campaign_resultBal <- subset(testdatamart_campaign_result, select = -c(id_contrat))
imputationResults <- missForest(testdatamart_campaign_resultBal)
testdatamart_campaign_resultBal <- imputationResults$ximp
sum(is.na(testdatamart_campaign_resultBal))
testdatamart_campaign_resultBal[which(is.na(DFAnalize_campaign_resultBal)),]

str(testdatamart_campaign_resultBal)

test_x_dummy <- dummyFyTest(testdatamart_campaign_resultBal) 
imputationResults <- missForest(test_x_dummy)
test_x_dummy <- imputationResults$ximp

str(test_x_dummy)
# Test on Models
str(x_dummy)

#importance_svm <- varImp(models_svm)


# ***************************************************************************
#                   EVALUATE MODELS   ---- Test data 
# ***************************************************************************

ModelListRF <- list(RF=models_rf )
ModelList <- list(SVM=models_svm, LB=models_LogitBoost, C5=models_c5, GBM=models_gbm, DNN=models_nn , KNN=models_knn)
names(ModelList)
PredictModel <-  function (x,testData,y) {
  pred_ <- predict(x, testData )
  table(y, pred_)
  return(pred_) 
}

FinalTestPredictModelProb <-  function (x,testData) {
  pred_  <- predict(x, testData )
  pred_Prob <- predict(x, testData, type = "prob" , probability = TRUE  )
  me <- list(
    prediction = pred_,
    pred_Prob = pred_Prob
  )

  return(me) 
}


FinalpredsProbList <-  sapply(ModelList, FinalTestPredictModelProb, simplify = F, testData=test_x_dummy )
FinalRFpredsProbList <-  sapply(ModelListRF, FinalTestPredictModelProb, simplify = F, testData=testdatamart_campaign_resultBal )


Finalpred_prob_svm <- as.data.frame(attr(FinalpredsProbList$SVM$pred_Prob, "probabilities"))
Finalpred_prob_c5 <- as.data.frame(FinalpredsProbList$C5$pred_Prob)
Finalpred_prob_LB  <- as.data.frame(FinalpredsProbList$LB$pred_Prob)
Finalpred_prob_RF <- as.data.frame(FinalRFpredsProbList$RF$pred_Prob)
Finalpred_prob_GBM <- as.data.frame(FinalpredsProbList$GBM$pred_Prob)
Finalpred_prob_DNN <- as.data.frame(FinalpredsProbList$DNN$pred_Prob)
Finalpred_prob_KNN <- as.data.frame(FinalpredsProbList$KNN$pred_Prob)

Finalpred_svm <- FinalpredsProbList$SVM$prediction
Finalpred_c5 <- FinalpredsProbList$C5$prediction
Finalpred_LB  <-FinalpredsProbList$LB$prediction
Finalpred_RF <- FinalRFpredsProbList$RF$prediction
Finalpred_GBM <- FinalpredsProbList$GBM$prediction
Finalpred_DNN <- FinalpredsProbList$DNN$prediction
Finalpred_KNN <- FinalpredsProbList$KNN$prediction

#install.packages('lift')

Finalpred_list <- list(SVM=Finalpred_svm, C5=Finalpred_c5, LB=Finalpred_LB, RF=Finalpred_RF, GBM=Finalpred_GBM, DNN=Finalpred_DNN)

FinalpredProbsAll <- data.frame(  svmNON=Finalpred_prob_svm$NON, svmOUI=Finalpred_prob_svm$OUI, 
                             c5NON=Finalpred_prob_c5$NON, c5OUI=Finalpred_prob_c5$OUI, 
                             LBNON=Finalpred_prob_LB$NON, LBOUI=Finalpred_prob_LB$OUI, 
                             RFNON=Finalpred_prob_RF$NON, RFOUI=Finalpred_prob_RF$OUI, 
                             DNNNON=Finalpred_prob_DNN$NON, DNNOUI=Finalpred_prob_DNN$OUI,
                             KNNNON=Finalpred_prob_DNN$NON, KNNOUI=Finalpred_prob_DNN$OUI,
                             gbmNON=Finalpred_prob_GBM$NON, gbmOUI=Finalpred_prob_GBM$OUI
)

FinalensembleModelList <- list(LB=stack_models_LogitBoost, 
                          C5=stack_models_c5, 
                          RF=stack_models_rf, 
                          GBM=stack_model_gbm,
                          KNN=stack_models_KNN
                          )

FinalensemblePred <- sapply(FinalensembleModelList, FinalTestPredictModelProb, simplify = F, testData=FinalpredProbsAll )

Finalepreds <- sapply(X = FinalensemblePred, FUN = function(x ) {x$prediction}, simplify = F, USE.NAMES = T  )


enPerfMyROCList <-  mapply(PerfMyROC, prediction = epreds, aname = names(epreds), MoreArgs = list(y=y),  SIMPLIFY = FALSE)



predProbsAllNON <- subset(FinalpredProbsAll, select = c(  svmNON, c5NON, LBNON, RFNON, gbmNON, KNNNON, DNNNON)) 
predProbsAllNON$stack_gbm_NON <- ensemblePred$GBM$pred_Prob$NON
predProbsAllNON$stack_LogitBoost_NON  <- ensemblePred$LB$pred_Prob$NON
predProbsAllNON$stack_c5_NON <- ensemblePred$C5$pred_Prob$NON
predProbsAllNON$stack_rf_NON <- ensemblePred$RF$pred_Prob$NON
predProbsAllNON$stack_KNN_NON <- ensemblePred$KNN$pred_Prob$NON


######################

trainProbsAllOUI <- subset(trainProbsAll, select = c(  svmOUI, c5OUI, LBOUI, RFOUI, gbmOUI, KNNOUI, DNNOUI)) 
trainProbsAllOUI$stack_gbm_OUI <- ensemblePredTrain$GBM$pred_Prob$OUI
trainProbsAllOUI$stack_LogitBoost_OUI  <- ensemblePredTrain$LB$pred_Prob$OUI
trainProbsAllOUI$stack_c5_OUI <- ensemblePredTrain$C5$pred_Prob$OUI
trainProbsAllOUI$stack_rf_OUI <- ensemblePredTrain$RF$pred_Prob$OUI
trainProbsAllOUI$stack_KNN_OUI <- ensemblePredTrain$KNN$pred_Prob$OUI


predProbsAllOUI <- subset(predProbsAll, select = c(  svmOUI, c5OUI, LBOUI, RFOUI, gbmOUI, KNNOUI, DNNOUI)) 
predProbsAllOUI$stack_gbm_OUI <- ensemblePred$GBM$pred_Prob$OUI
predProbsAllOUI$stack_LogitBoost_OUI  <- ensemblePred$LB$pred_Prob$OUI
predProbsAllOUI$stack_c5_OUI <- ensemblePred$C5$pred_Prob$OUI
predProbsAllOUI$stack_rf_OUI <- ensemblePred$RF$pred_Prob$OUI
predProbsAllOUI$stack_KNN_OUI <- ensemblePred$KNN$pred_Prob$OUI

#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI

gm_mean = function(x, na.rm=TRUE){
  x = x+0.001
  prod(x[x > 0], na.rm=na.rm) 
  exp(mean(log(x)))
}

gm_mean = function(x, na.rm=TRUE){
  x = x+0.001
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}



#-- Find the maximum of probablity NON
predProbsAllNON2 <- predProbsAllNON
predProbsAllNON2$Mean <- apply(predProbsAllNON, 1, mean)
predProbsAllNON2$Geom <- apply(predProbsAllNON, 1, gm_mean)


#-- Find the maximum of probablity 
predProbsAllOUI2 <- predProbsAllOUI
predProbsAllOUI2$Final <- apply(predProbsAllOUI, 1, max)
predProbsAllOUI2$Mean <- apply(predProbsAllOUI, 1, mean)
predProbsAllOUI2$Geom <- apply(predProbsAllOUI, 1, gm_mean)

predProbsAllOUI2$Diff <- (predProbsAllOUI2$Mean- predProbsAllNON2$Mean)
predProbsAllOUI2$NewFinal <- predProbsAllOUI2$Final +  predProbsAllOUI2$Diff
predProbsAllOUI2$NewFinal <- (predProbsAllOUI2$NewFinal - min(predProbsAllOUI2$NewFinal)) / (max(predProbsAllOUI2$NewFinal)- min(predProbsAllOUI2$NewFinal)) 
predProbsAllOUI2$NewFinal <- (predProbsAllOUI2$Final + predProbsAllOUI2$NewFinal)/2


trainProbsAllNON2 <- trainProbsAllNON
trainProbsAllNON2$Mean <- apply(trainProbsAllNON, 1, mean)
trainProbsAllNON2$Geom <- apply(trainProbsAllNON, 1, gm_mean)

trainProbsAllOUI2 <- trainProbsAllOUI
trainProbsAllOUI2$Final <- apply(trainProbsAllOUI, 1, max)
trainProbsAllOUI2$Mean <- apply(trainProbsAllOUI, 1, mean)
trainProbsAllOUI2$Geom <- apply(trainProbsAllOUI, 1, gm_mean)

trainProbsAllOUI2$Diff <- (trainProbsAllOUI2$Mean- trainProbsAllNON2$Mean)
trainProbsAllOUI2$NewFinal <- trainProbsAllOUI2$Final +  trainProbsAllOUI2$Diff
trainProbsAllOUI2$NewFinal <- (trainProbsAllOUI2$NewFinal - min(trainProbsAllOUI2$NewFinal)) / (max(trainProbsAllOUI2$NewFinal)- min(trainProbsAllOUI2$NewFinal)) 
trainProbsAllOUI2$NewFinal <- (trainProbsAllOUI2$Final + trainProbsAllOUI2$NewFinal)/2

RtrainProbsAllOUI3RD_NB <- data.frame(Max = predProbsAllOUI2$NewFinal, Mean = predProbsAllOUI2$Mean, MeanMax = predProbsAllOUI2$NewFina)
RtrainProbsAllOUI3RD_NB$Class <- y

RtestProbsAllOUI3RD_NB <- data.frame(Max = trainProbsAllOUI2$NewFinal, Mean = trainProbsAllOUI2$Mean, MeanMax = trainProbsAllOUI2$NewFina)
RtestProbsAllOUI3RD_NB$Class <- ytrain



predProbsAllOUI2$Class <- y
predProbsAllOUI2 <- as.data.frame( model.matrix( ~. , predProbsAllOUI2))
predProbsAllOUI2 <- subset(predProbsAllOUI2, select=-`(Intercept)`)
predProbsAllOUI2$ClassNON <- ifelse(predProbsAllOUI2$ClassOUI==0,1,0)
predProbsAllOUI2$Class <- y

predProbsAllOUI2TOTAL <-  nrow(predProbsAllOUI2)

# for cumsum Final
predProbsAllOUI2 <- predProbsAllOUI2[order(predProbsAllOUI2$Final, decreasing = F),]
predProbsAllOUI2$OUISum  <- cumsum(predProbsAllOUI2$ClassOUI)
predProbsAllOUI2$NONSum  <- cumsum(predProbsAllOUI2$ClassNON)
predProbsAllOUI2$Percent <- predProbsAllOUI2$OUISum/ predProbsAllOUI2$NONSum

# for cumsum mean 
predProbsAllOUI2 <- predProbsAllOUI2[order(predProbsAllOUI2$Mean, decreasing = F),]
predProbsAllOUI2$OUISumMean  <- cumsum(predProbsAllOUI2$ClassOUI)
predProbsAllOUI2$NONSumMean  <- cumsum(predProbsAllOUI2$ClassNON)
predProbsAllOUI2$PercentMean <- predProbsAllOUI2$OUISumMean/ (predProbsAllOUI2$OUISumMean+predProbsAllOUI2$NONSumMean)

# for cumsum NewFinal
predProbsAllOUI2 <- predProbsAllOUI2[order(predProbsAllOUI2$NewFinal, decreasing = F),]
predProbsAllOUI2$OUISumNF  <- cumsum(predProbsAllOUI2$ClassOUI)
predProbsAllOUI2$NONSumNF  <- cumsum(predProbsAllOUI2$ClassNON)
predProbsAllOUI2$PercentNF <- predProbsAllOUI2$OUISumNF/ (predProbsAllOUI2$OUISumNF+predProbsAllOUI2$NONSumNF)


#predProbsAllOUI2$ID <- Test_campaign_result_ID


pseq <- seq(from = 0, to = 1, by = 0.1)


aggDiffFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, NewFinal>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100)/max(ct$NON + ct$OUI,1)
}


aggFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Final>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100)/max(ct$NON + ct$OUI,1)
}

aggFUNMean <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Mean>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100) /max(ct$NON + ct$OUI,1)
}

aggFUNGeom <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Geom>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  (ct$OUI*100) /max(ct$NON + ct$OUI,1)
}


StraggDiffFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, NewFinal>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}


StraggFUN <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Final>=n, select =  Class )))
  ct <- cast(ct, ~Var1, value = 'Freq')
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

StraggFUNMean <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Mean>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  #(ct$OUI*100) /max(ct$NON + ct$OUI,1)
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

StraggFUNGeom <- function(n) { 
  ct <- as.data.frame(table(subset(predProbsAllOUI2, Geom>=n, select =  Class )))
  ct<- cast(ct, ~Var1, value = 'Freq')
  #(ct$OUI*100) /max(ct$NON + ct$OUI,1)
  return (paste0( " ( OUI:", (ct$OUI), ")/(TOTAL:" , max(ct$NON + ct$OUI,1) , ")"  ))
}

scoreratioMax <- sapply(pseq, FUN=aggFUN)
scoreratioDiffMax <- sapply(pseq, FUN=aggDiffFUN)
scoreratioMean <- sapply(pseq, FUN=aggFUNMean)
scoreratioGeomMean <- sapply(pseq, FUN=aggFUNGeom)

StrscoreratioMax <- sapply(pseq, FUN=StraggFUN)
StrscoreratioDiffMax  <- sapply(pseq, FUN=StraggDiffFUN)
StrscoreratioMean <- sapply(pseq, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq, FUN=StraggFUNGeom)

ScoreconclusionDF <- data.frame(Score=pseq, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDF$Diff  <- scoreratioDiffMax
ScoreconclusionDF$PercentMean  <- scoreratioMean
ScoreconclusionDF$RatioMean  <- StrscoreratioMean
ScoreconclusionDF$PercentGeomMean  <- scoreratioGeomMean
ScoreconclusionDF$RatioGeomMean  <- StrscoreratioGeomMean
ScoreconclusionDF$PercentDiff  <- StrscoreratioDiffMax


ggplot(ScoreconclusionDF ) + 
  geom_line(aes(Score, y=Percent), stat = "identity", color="blue" ) +
  geom_line(aes(Score, y=PercentMean), stat = "identity", color="darkgoldenrod") +
  geom_line(aes(Score, y=PercentGeomMean), stat = "identity", color="black") +
  geom_line(aes(Score, y=Diff), stat = "identity", color="red") +
  scale_color_manual(values=c("Max"="blue",  "Mean" = "darkgoldenrod", "GeomMean" = "black" ), 
                     name="Score Vs % \n color legend", guide="legend") + 
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Percentage " )

ggplot(ScoreconclusionDF )  + 
  geom_line(aes(Score, y=Percent), stat = "identity", color="blue") +
  geom_line(aes(Score, y=PercentMean), stat = "identity", color="darkgoldenrod") +
  geom_line(aes(Score, y=PercentGeomMean), stat = "identity", color="black") +
  scale_color_manual(values=c("Max"="blue",  "Mean" = "darkgoldenrod", "GeomMean" = "yellow" ), 
                     name="Score Vs % \n color legend", guide="legend") + 
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity " )





################################ above 0.75
pseq75 <- seq(from = 0, to = 1, by = 0.001)

scoreratioMax <- sapply(pseq75, FUN=aggFUN)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)
scoreratioGeomMean <- sapply(pseq75, FUN=aggFUNGeom)
scoreratioDiffMax <- sapply(pseq75, FUN=aggDiffFUN)


StrscoreratioMax <- sapply(pseq75, FUN=StraggFUN)
StrscoreratioMean <- sapply(pseq75, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq75, FUN=StraggFUNGeom)
StrscoreratioDiffMax <- sapply(pseq75, FUN=StraggDiffFUN)

ScoreconclusionDFMax <- data.frame(Score=pseq75, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDFMean <- data.frame(Score=pseq75, Percent=scoreratioMean, Ratio= StrscoreratioMean)
ScoreconclusionDFDiffMax <- data.frame(Score=pseq75, Percent=scoreratioDiffMax, Ratio= StrscoreratioDiffMax)
ScoreconclusionDFMean <- subset(ScoreconclusionDFMean,ScoreconclusionDFMean$Percent>0 )
#scoreratioMean <- scoreratioMean[-11] 
ScoreconclusionDFGeomMean <- data.frame(Score=pseq75, Percent=scoreratioGeomMean, Ratio= StrscoreratioGeomMean, diff=ScoreconclusionDFDiffMax)

ggplot(ScoreconclusionDFMax, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity[Max of all] " )

ggplot(ScoreconclusionDFMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Mean of all] " )

ggplot(ScoreconclusionDFGeomMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Geometric Mean of all] " )

ggplot(ScoreconclusionDFDiffMax, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Diff of all] " )


plotLift(predicted = predProbsAllOUI2$Final, predProbsAllOUI2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Mean, predProbsAllOUI2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI2$Geom, predProbsAllOUI2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )
plotLift(predicted = predProbsAllOUI2$Diff, predProbsAllOUI2$ClassOUI, main= "Diff Probabity - Lift", n.buckets = 10 )


lift2 <- lift( Class ~ Final+Mean+Geom+Diff, data = predProbsAllOUI2, class = "OUI")
lift2 <- lift( Class ~ Diff, data = predProbsAllOUI2, class = "OUI")
xyplot(lift2, auto.key = list(columns = 4)  )

predProbsAllOUI2Ordered <-   predProbsAllOUI2[order(predProbsAllOUI2$Final, decreasing = T),]
nrow(predProbsAllOUI2Ordered)

predProbsAllOUI2Ordered$decile  <-  dplyr::ntile(predProbsAllOUI2Ordered$Final,10)
table(predProbsAllOUI2Ordered$Class, predProbsAllOUI2Ordered$decile)

predProbsAllOUI2OrderedMean <-   predProbsAllOUI2[order(predProbsAllOUI2$Mean, decreasing = T),]
predProbsAllOUI2OrderedMean$Meandecile  <-  dplyr::ntile(predProbsAllOUI2OrderedMean$Mean,10)
table(predProbsAllOUI2OrderedMean$Class, predProbsAllOUI2OrderedMean$Meandecile)

predProbsAllOUI2OrderedDiff <-   predProbsAllOUI2[order(predProbsAllOUI2$Diff, decreasing = T),]
predProbsAllOUI2OrderedDiff$Diffecile  <-  dplyr::ntile(predProbsAllOUI2OrderedDiff$Diff,10)
table(predProbsAllOUI2OrderedDiff$Class, predProbsAllOUI2OrderedDiff$Diffecile)


predProbsAllOUI2OrderedDiff <-   predProbsAllOUI2[order( predProbsAllOUI2Ordered$Final+ predProbsAllOUI2$Diff , decreasing = T),]
predProbsAllOUI2OrderedDiff$Diffecile  <-  dplyr::ntile(predProbsAllOUI2OrderedDiff$Diff,10)
table(predProbsAllOUI2OrderedDiff$Class, predProbsAllOUI2OrderedDiff$Diffecile)


RtrainProbsAllOUI3RD_NB <- data.frame(Max = predProbsAllOUI2$NewFinal, Mean = predProbsAllOUI2$Mean, MeanMax = predProbsAllOUI2$NewFina)
#RtrainProbsAllOUI3RD_NB$Class <- y

RtestProbsAllOUI3RD_NB <- data.frame(Max = trainProbsAllOUI2$NewFinal, Mean = trainProbsAllOUI2$Mean, MeanMax = trainProbsAllOUI2$NewFina)
#RtestProbsAllOUI3RD_NB$Class <- ytrain

NB3RD_Model <- train(x = RtrainProbsAllOUI3RD_NB,y=y, trControl=fitControl, method='nb', metric="Accuracy" )
pred_NB3RD  <- predict(object = NB3RD_Model, RtestProbsAllOUI3RD_NB )
pred_Prob_NB3RD <- predict(NB3RD_Model, RtestProbsAllOUI3RD_NB, type = "prob" , probability = TRUE  )
table(ytrain, pred_NB3RD)
cf_NB3RD <- confusionMatrix(data = ytrain, pred_NB3RD, positive = levels(pred_NB3RD)[2])



# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
#                   EVALUATE MODELS   ---- OUI - NON 
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************
# ***************************************************************************

# ***************************************************************************
#   Train model
# ***************************************************************************


trainProbsAllOUI_NON <- subset(xpredProbsAll, select = c(  svmOUI , c5OUI, LBOUI, RFOUI, gbmOUI)) 

trainProbsAllOUI_NON$svmOUI_svmNON  <- xpredProbsAll$svmOUI - xpredProbsAll$svmNON
trainProbsAllOUI_NON$c5OUI_c5NON  <- xpredProbsAll$c5OUI - xpredProbsAll$c5NON
trainProbsAllOUI_NON$LBOUI_LBNON  <- xpredProbsAll$LBOUI - xpredProbsAll$LBNON
trainProbsAllOUI_NON$RFOUI_RFNON  <- xpredProbsAll$RFOUI - xpredProbsAll$RFNON
trainProbsAllOUI_NON$gbmOUI_gbmNON  <- xpredProbsAll$gbmNON - xpredProbsAll$gbmNON


trainProbsAllOUI_NON$stack_gbm_OUI <- ensemblePredTrain$GBM$pred_Prob$OUI
trainProbsAllOUI_NON$stack_LogitBoost_OUI  <- ensemblePredTrain$LB$pred_Prob$OUI
trainProbsAllOUI_NON$stack_c5_OUI <- ensemblePredTrain$C5$pred_Prob$OUI
trainProbsAllOUI_NON$stack_rf_OUI <- ensemblePredTrain$RF$pred_Prob$OUI
trainProbsAllOUI_NON$stack_svm_OUI <- ensemblePredTrain$SVM$pred_Prob$OUI

trainProbsAllOUI_NON$stack_gbm_OUI_NON <- ensemblePredTrain$GBM$pred_Prob$OUI - ensemblePredTrain$GBM$pred_Prob$NON
trainProbsAllOUI_NON$stack_LogitBoost_OUI_NON  <- ensemblePredTrain$LB$pred_Prob$OUI  - ensemblePredTrain$LB$pred_Prob$NON
trainProbsAllOUI_NON$stack_c5_OUI_NON <- ensemblePredTrain$C5$pred_Prob$OUI - ensemblePredTrain$C5$pred_Prob$NON
trainProbsAllOUI_NON$stack_rf_OUI_NON <- ensemblePredTrain$RF$pred_Prob$OUI - ensemblePredTrain$RF$pred_Prob$NON
trainProbsAllOUI_NON$stack_svm_OUI_NON <- ensemblePredTrain$SVM$pred_Prob$OUI - ensemblePredTrain$SVM$pred_Prob$NON


#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI


#-- Find the maximum of probablity 


stack_models_svm_3Level  <- svm(x = trainProbsAllOUI_NON2, y=ytrain, kernel="polynomial", 
                                cost=1,
                                gamma=0.5, 
                                probability=T, scale = T )




# ***************************************************************************
#   Pred model
# ***************************************************************************


predProbsAllOUI_NON <- subset(predProbsAll, select = c(  svmOUI , c5OUI, LBOUI, RFOUI, gbmOUI)) 

predProbsAllOUI_NON$svmOUI_svmNON  <- predProbsAll$svmOUI - predProbsAll$svmNON
predProbsAllOUI_NON$c5OUI_c5NON  <- predProbsAll$c5OUI - predProbsAll$c5NON
predProbsAllOUI_NON$LBOUI_LBNON  <- predProbsAll$LBOUI - predProbsAll$LBNON
predProbsAllOUI_NON$RFOUI_RFNON  <- predProbsAll$RFOUI - predProbsAll$RFNON
predProbsAllOUI_NON$gbmOUI_gbmNON  <- predProbsAll$gbmNON - predProbsAll$gbmNON


predProbsAllOUI_NON$stack_gbm_OUI <- ensemblePred$GBM$pred_Prob$OUI
predProbsAllOUI_NON$stack_LogitBoost_OUI  <- ensemblePred$LB$pred_Prob$OUI
predProbsAllOUI_NON$stack_c5_OUI <- ensemblePred$C5$pred_Prob$OUI
predProbsAllOUI_NON$stack_rf_OUI <- ensemblePred$RF$pred_Prob$OUI
predProbsAllOUI_NON$stack_svm_OUI <- ensemblePred$SVM$pred_Prob$OUI

predProbsAllOUI_NON$stack_gbm_OUI_NON <- ensemblePred$GBM$pred_Prob$OUI - ensemblePred$GBM$pred_Prob$NON
predProbsAllOUI_NON$stack_LogitBoost_OUI_NON  <- ensemblePred$LB$pred_Prob$OUI  - ensemblePred$LB$pred_Prob$NON
predProbsAllOUI_NON$stack_c5_OUI_NON <- ensemblePred$C5$pred_Prob$OUI - ensemblePred$C5$pred_Prob$NON
predProbsAllOUI_NON$stack_rf_OUI_NON <- ensemblePred$RF$pred_Prob$OUI - ensemblePred$RF$pred_Prob$NON
predProbsAllOUI_NON$stack_svm_OUI_NON <- ensemblePred$SVM$pred_Prob$OUI - ensemblePred$SVM$pred_Prob$NON



#pred_prob_stack_svm <- as.data.frame(attr(ensemblePred$SVM$pred_Prob, "probabilities"))
#predProbsAllOUI$stack_svm_OUI <- pred_prob_stack_svm$OUI



pred_3dLevel  <- predict(stack_models_svm_3Level, predProbsAllOUI_NON2 )
pred_Prob3dLevel <- predict(stack_models_svm_3Level, predProbsAllOUI_NON2, type = "prob" , probability = TRUE  )
table(y, pred_Prob3dLevel)
cf <- confusionMatrix(data = y, pred_Prob3dLevel, positive = levels(pred_Prob3dLevel)[2])
cf



pseq75 <- seq(from = 0, to = 1, by = 0.001)

scoreratioMax <- sapply(pseq75, FUN=aggFUN)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)
scoreratioMean <- sapply(pseq75, FUN=aggFUNMean)

StrscoreratioMax <- sapply(pseq75, FUN=StraggFUN)
StrscoreratioMean <- sapply(pseq75, FUN=StraggFUNMean)
StrscoreratioGeomMean <- sapply(pseq75, FUN=StraggFUNGeom)

ScoreconclusionDFMax <- data.frame(Score=pseq75, Percent=scoreratioMax, Ratio= StrscoreratioMax)
ScoreconclusionDFMean <- data.frame(Score=pseq75, Percent=scoreratioMean, Ratio= StrscoreratioMean)
ScoreconclusionDFMean <- subset(ScoreconclusionDFMean,ScoreconclusionDFMean$Percent>0 )
#scoreratioMean <- scoreratioMean[-11] 
ScoreconclusionDFGeomMean <- data.frame(Score=pseq75, Percent=scoreratioGeomMean, Ratio= StrscoreratioGeomMean)

ggplot(ScoreconclusionDFMax, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score   ") + ggtitle(" Score Vs Probablity[Max of all] " )

ggplot(ScoreconclusionDFMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Mean of all] " )

ggplot(ScoreconclusionDFGeomMean, aes(Score, y=Percent)) + geom_line(stat = "identity") +
  xlab("Score") + ylab("Actual Percent of OUI.  >= score  ") + ggtitle(" Score Vs Probablity[Geometric Mean of all] " )

if(F) {
  
plotLift(predicted = predProbsAllOUI_NON2$Final, predProbsAllOUI_NON2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Mean, predProbsAllOUI_NON2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Geom, predProbsAllOUI_NON2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )

plotLift(predicted = predProbsAllOUI_NON2$Final, predProbsAllOUI_NON2$ClassOUI, main= "Max Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Mean, predProbsAllOUI_NON2$ClassOUI, main= "Mean Probabity - Lift", n.buckets = 10 , cumulative = T)
plotLift(predicted = predProbsAllOUI_NON2$Geom, predProbsAllOUI_NON2$ClassOUI, main= "Geometric Mean Probabity - Lift", n.buckets = 10 )
}

lift2 <- lift( Class ~ Final+Mean+Geom, data = predProbsAllOUI_NON2, class = "OUI")
xyplot(lift2, auto.key = list(columns = 3))

