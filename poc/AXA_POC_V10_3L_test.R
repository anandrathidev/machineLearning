
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
  xtrain$LastNameLen <-  mytrans(scale(xtrain$LastNameLen))
  xtrain$adressLen <-  mytrans(scale(xtrain$adressLen))
  
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
  names(xtrain_dummy)[names(xtrain_dummy) == 'phone_number_usedAXA et CEDRICOM'] <- 'phone_number_usedAXA_et_CEDRICOM'
  names(xtrain_dummy)[names(xtrain_dummy) == 'other_phone_fiabilitynon trouve'] <- 'other_phone_fiabilitynon_trouve'
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


FinalpredProbsAllNON <- subset(FinalpredProbsAll, select = c(  svmNON, c5NON, LBNON, RFNON, gbmNON, KNNNON, DNNNON)) 
FinalpredProbsAllNON$stack_gbm_NON <- FinalensemblePred$GBM$pred_Prob$NON
FinalpredProbsAllNON$stack_LogitBoost_NON  <- FinalensemblePred$LB$pred_Prob$NON
FinalpredProbsAllNON$stack_c5_NON <- FinalensemblePred$C5$pred_Prob$NON
FinalpredProbsAllNON$stack_rf_NON <- FinalensemblePred$RF$pred_Prob$NON
FinalpredProbsAllNON$stack_KNN_NON <- FinalensemblePred$KNN$pred_Prob$NON


######################
FinalpredProbsAllOUI <- subset(FinalpredProbsAll, select = c(  svmOUI, c5OUI, LBOUI, RFOUI, gbmOUI, KNNOUI, DNNOUI)) 
FinalpredProbsAllOUI$stack_gbm_OUI <- FinalensemblePred$GBM$pred_Prob$OUI
FinalpredProbsAllOUI$stack_LogitBoost_OUI  <- FinalensemblePred$LB$pred_Prob$OUI
FinalpredProbsAllOUI$stack_c5_OUI <- FinalensemblePred$C5$pred_Prob$OUI
FinalpredProbsAllOUI$stack_rf_OUI <- FinalensemblePred$RF$pred_Prob$OUI
FinalpredProbsAllOUI$stack_KNN_OUI <- FinalensemblePred$KNN$pred_Prob$OUI

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
FinalpredProbsAllNON2 <- FinalpredProbsAllNON
FinalpredProbsAllNON2$Mean <- apply(FinalpredProbsAllNON, 1, mean)


#-- Find the maximum of probablity 
FinalpredProbsAllOUI2 <- FinalpredProbsAllOUI
FinalpredProbsAllOUI2$Final <- apply(FinalpredProbsAllOUI, 1, max)
FinalpredProbsAllOUI2$Mean <- apply(FinalpredProbsAllOUI, 1, mean)

FinalpredProbsAllOUI2$Diff <- (FinalpredProbsAllOUI2$Mean- FinalpredProbsAllNON2$Mean)
FinalpredProbsAllOUI2$NewFinal <- FinalpredProbsAllOUI2$Final +  FinalpredProbsAllOUI2$Diff
FinalpredProbsAllOUI2$NewFinal <- (FinalpredProbsAllOUI2$NewFinal - min(FinalpredProbsAllOUI2$NewFinal)) / (max(FinalpredProbsAllOUI2$NewFinal)- min(FinalpredProbsAllOUI2$NewFinal)) 
FinalpredProbsAllOUI2$NewFinal <- (FinalpredProbsAllOUI2$Final + FinalpredProbsAllOUI2$NewFinal)/2

FinalRtestProbsAllOUI3RD_NB <- data.frame(Max = FinalpredProbsAllOUI2$Final, 
                                          Mean = FinalpredProbsAllOUI2$Mean, 
                                          MeanMax = FinalpredProbsAllOUI2$NewFinal)

Finalpred_NB3RD  <- predict(object = NB3RD_Model, FinalRtestProbsAllOUI3RD_NB )
Finalpred_Prob_NB3RD <- predict(NB3RD_Model, FinalRtestProbsAllOUI3RD_NB, type = "prob" , probability = TRUE  )

FinalScore <- data.frame(NB = (Finalpred_Prob_NB3RD$OUI - Finalpred_Prob_NB3RD$NON)  , 
                           Mean = FinalpredProbsAllOUI2$Mean, 
                           MeanMax = FinalpredProbsAllOUI2$NewFinal,
                           y = Finalpred_NB3RD
                            
)
FinalScore$NBDiff <- (FinalScore$NB - min(FinalScore$NB) ) / (max(FinalScore$NB) - min(FinalScore$NB))
FinalScore$TotMean <- (FinalScore$NBDiff+FinalScore$Mean+FinalScore$MeanMax)/3

FinalScore_withID <- data.frame(id=id_contrat, FinalScore=round(FinalScore$TotMean, digits = 4) ,  y = FinalScore$y )

write.csv(x = FinalScore_withID, file="C:/Users/rb117/Documents/work/POC_analytics/Data//result_norm.csv", sep = ',')
#write.csv(x = FinalScore_withID, file="C:/Users/rb117/Documents/work/POC_analytics/Data//result_full.csv", sep = ',')
FinalScore_withID_full <-  read.csv("C:/Users/rb117/Documents/work/POC_analytics/Data//result_full.csv",  sep = ',')
FinalScore_withID_norm <-  read.csv("C:/Users/rb117/Documents/work/POC_analytics/Data//result_norm.csv",  sep = ',')

Finalcomparison <- compare(FinalScore_withID_full,FinalScore_withID_norm,allowAll=TRUE)
Finalcomparisonmerge  <- merge(x=FinalScore_withID_full, y=FinalScore_withID_norm,  by.x = "id", by.y = "id", all.y = F)
View(Finalcomparisonmerge)
table(Finalcomparisonmerge$y.x, Finalcomparisonmerge$y.y)
testdatamart_campaign_result2 <- testdatamart_campaign_result
testdatamart_campaign_result2$y <- FinalScore_withID$y
testdatamart_campaign_result2$FinalScore <- FinalScore_withID$FinalScore

install.packages("scatterplot3d")
#install.packages("plot3D")
#install.packages("rgl")
#library(plot3D)
library(rgl)
#plot3D::scatter3D( FinalScore ~ insurance_contrat_value|phone_type|gender , data=testdatamart_campaign_result2 )
#rgl::plot3d( FinalScore ~ insurance_contrat_value , data=testdatamart_campaign_result2 )
#df <- testdatamart_campaign_result2
library(scatterplot3d)
with(testdatamart_campaign_result2,
       plot3d(FinalScore ~ age_in_weeks, type='h', lwd=10, col=rainbow(3)))