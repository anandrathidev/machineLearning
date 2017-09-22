install.packages('MASS')
install.packages('car')
install.packages('h2o')
install.packages('caret')

library(h2o)
library(MASS)
library(car)
library(h2o)
library(caret)
library(ggplot2)
library(splines)

setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_04_PredictiveAnalytics_II/NuralNetworks/AssignmentNeuralNetworks/")
#setwd("C:/NeuralNetwors")

##---- Checkpoint 1 - Data Understanding and Preparation of Master File

trainFull <- read.csv("telecom_nn_train.csv")
##----Understand data , chk for na , duplicates etc
summary(trainFull)
str(trainFull)
sum(is.na(trainFull))

trainFull <- na.omit(trainFull)

validationsize <- round(nrow(trainFull)*0.20)
validationsampleIdx <- sample(nrow(trainFull), validationsize, replace=FALSE)

validationsample <- trainFull[validationsampleIdx,]
trainsample <- trainFull[-validationsampleIdx,]

##---- Checkpoint 2 - EDA




h2o.init(nthreads=-1, startH2O = TRUE, min_mem_size = "3g", max_mem_size = "3g") #ip = "localhost", port = 54321, nthreads=-1, startH2O = TRUE, min_mem_size = "3g")
h2o.removeAll()

#h2oTrain <- as.h2o( trainsample )
#h2oValidation <- as.h2o(validationsample)

##---- Checkpoint 3 - Data Preparation 
#-- Normlise data etc
h2otrainFull <- trainFull
h2otrainFull$MonthlyCharges <- unlist(scale( trainFull$MonthlyCharges )[,1])
h2otrainFull$TotalCharges <- unlist(scale( trainFull$TotalCharges )[,1])
h2otrainFull$tenure <- unlist(scale( trainFull$tenure )[,1])
h2otrainFull  <- as.h2o( h2otrainFull  )
h2otrainFullorig  <- as.h2o( trainFull  )


# Convert the first column to a factor
churnlabel <- "Churn"
independentVars <- setdiff(names(h2otrainFull), churnlabel)
#h2oTrain[,churnlabel] <- as.factor(h2oTrain[,churnlabel])
#h2oValidation[,churnlabel] <- as.factor(h2oValidation[,churnlabel])
h2otrainFull[,churnlabel] <- as.factor(h2otrainFull[,churnlabel])

set.seed(100)
# Perform 5-fold cross-validation on the training_frame

#distribution = "AUTO", #multinomial
#activation = "TanhWithDropout", #RectifierWithDropout
#hidden_dropout_ratio = c(0.1,0.1,0.1 ),
#validation_frame = h2oValidation,
#l1 = 5e-6,
#epochs = 90

#---- Checkpoint 4 - Model Building ( 300 marks per model)
PlotNN <-  function(churnperfNN2) {
  nx=churnperfNN2[,c('neurons')] 
  plot(x=nx, y=churnperfNN2[,c('churnr2Sig')] , ylim=c(0, 1), col="red", xlab="Nerons Sigmoid / epoch" )
  lines(x=nx, churnperfNN2[,c('churnNoErrorSig')], col="blue")
  lines(x=nx, churnperfNN2[,c('churnYesErrorSig')], col="green")
  lines(x=nx, churnperfNN2[,c('churnNoErrorSig')]+churnperfNN2[,c('churnYesErrorSig')], col="black")
  lines(x=nx, churnperfNN2[,c('churnAccuracySig')], col="violetred")
  
  plot(x=nx, y = churnperfNN2[,c('churnr2Rect')], ylim=c(0, 1), col="red", xlab="Nerons Rect / epoch" )
  lines(x=nx, churnperfNN2[,c('churnNoErrorRect')], col="blue")
  lines(x=nx, churnperfNN2[,c('churnYesErrorRect')], col="green")
  lines(x=nx, churnperfNN2[,c('churnNoErrorRect')]+churnperfNN2[,c('churnYesErrorRect')], col="black")
  lines(x=nx, churnperfNN2[,c('churnAccuracyRect')], col="violetred")
}

PlotNNSig <-  function(churnperfNN2) {
  nx=churnperfNN2[,c('epoch')] 
  plot(x=nx, y=churnperfNN2[,c('churnr2Sig')] , ylim=c(0, 1), col="red", xlab="Nerons Sigmoid / epoch" )
  lines(x=nx, churnperfNN2[,c('churnNoErrorSig')], col="blue")
  lines(x=nx, churnperfNN2[,c('churnYesErrorSig')], col="green")
  lines(x=nx, churnperfNN2[,c('churnNoErrorSig')]+churnperfNN2[,c('churnYesErrorSig')], col="black")
  lines(x=nx, churnperfNN2[,c('churnAccuracySig')], col="violetred")

}


###---First Model 

LearnNNChurnNNNoEpoch <- function(independentVar, churnName,trainFull, S=31, E=168) {
  neuronList <- list()
  churnr2list  <- list()
  churnNoErrorList <- list()
  churnYesErrorList <- list()
  for (i in S:E) {
    churnModel <- h2o.deeplearning(x = independentVar,
                                   y =  churnName,
                                   seed=100,
                                   reproducible=T,
                                   training_frame = trainFull,
                                   hidden = c(i,i,i) )
    
    # Test the model on the test data
    churntPrediction <- h2o.predict(churnModel, trainFull)
    churnperf <- h2o.performance(model = churnModel, newdata = trainFull)
    confmatrix <-  h2o.confusionMatrix(churnperf)
    
    churnr2list[[i]] <- h2o.r2(churnperf)
    churnNoErrorList[[i]]  <- confmatrix$Error[1]
    churnYesErrorList[[i]] <- confmatrix$Error[2]
    neuronList[[i]] <- i
  }
  
  churnr2        <- unlist(churnr2list)
  churnNoError   <- unlist(churnNoErrorList)
  churnYesError  <-  unlist(churnYesErrorList)
  neurons  <-  unlist(neuronList)
  myresult <-  data.frame(neurons, churnr2, churnNoError, churnYesError)
  return(myresult)
}

churnperfNN <- LearnNNChurnNNNoEpoch(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, S=1,E=211)
plot(churnperfNN[,2], ylim=c(0, 1), col="red", xlab="Nerons" )
lines(churnperfNN[,3], col="blue")
lines(churnperfNN[,4], col="green")
lines(churnperfNN[,3]+churnperfNN[,4], col="black")


###---Second  Model , Iterate on Neorons  plus epochs

LearnNNChurnNN <- function(independentVar, churnName,trainFull, SeqChurn=seq(10, 50, 2), epochsTune=0) {
  neuronList <- list()
  churnr2listSig  <- list()
  churnNoErrorListSig <- list()
  churnYesErrorListSig <- list()
  churnAccuracySig <- list()
  
  churnr2listRect  <- list()
  churnNoErrorListRect <- list()
  churnYesErrorListRect <- list()
  churnAccuracyRect  <- list()
  actList <- list()
  epochs=epochsTune
  for (i in SeqChurn) {
    print( paste0("iteration number:", i ))
    if (epochsTune==0) {
      epochs=i
    }
    churnModel <- h2o.deeplearning(x = independentVar,
                                   y =  churnName,
                                   distribution =  "multinomial",
                                   activation="Tanh",
                                   seed=100,
                                   training_frame = trainFull,
                                   hidden = c(i,i,i) ,
                                   epochs=epochs)
    
    # Test the model on the test data
    churntPrediction <- h2o.predict(churnModel, trainFull)
    churnperf <- h2o.performance(model = churnModel, newdata = trainFull)
    confmatrix <-  h2o.confusionMatrix(churnperf)
    
    neuronList[[i]] <- i
    
    churnr2listSig[[i]] <- h2o.r2(churnperf)
    churnNoErrorListSig[[i]]  <- confmatrix$Error[1]
    churnYesErrorListSig[[i]] <- confmatrix$Error[2]
    churnAccuracySig[[i]] <- mean(churnperf@metrics$thresholds_and_metric_scores$accuracy)
    
    
    churnModel <- h2o.deeplearning(x = independentVar,
                                   y =  churnName,
                                   distribution =  "multinomial",
                                   activation="Rectifier",
                                   seed=100,
                                   training_frame = trainFull,
                                   hidden = c(i,i,i) ,
                                   epochs=epochs)
    
    churntPrediction <- h2o.predict(churnModel, trainFull)
    churnperf <- h2o.performance(model = churnModel, newdata = trainFull)
    confmatrix <-  h2o.confusionMatrix(churnperf)
    
    churnr2listRect[[i]] <- h2o.r2(churnperf)
    churnAccuracyRect[[i]] <- mean(churnperf@metrics$thresholds_and_metric_scores$accuracy)
    churnNoErrorListRect[[i]]  <- confmatrix$Error[1]
    churnYesErrorListRect[[i]] <- confmatrix$Error[2]
  }
  
  churnr2Sig        <- unlist(churnr2listSig)
  churnNoErrorSig   <- unlist(churnNoErrorListSig)
  churnYesErrorSig  <- unlist(churnYesErrorListSig)
  churnAccuracySig  <- unlist(churnAccuracySig)
  
  churnr2Rect        <- unlist(churnr2listRect)
  churnNoErrorRect   <- unlist(churnNoErrorListRect)
  churnYesErrorRect  <- unlist(churnYesErrorListRect)
  churnAccuracyRect  <- unlist(churnAccuracyRect)
  
  neurons  <-  unlist(neuronList)
  
  myresult <-  data.frame(neurons, churnr2Sig, churnNoErrorSig, churnYesErrorSig, churnAccuracySig, churnr2Rect, churnNoErrorRect, churnYesErrorRect, churnAccuracyRect )
  myresult <- data.frame(sapply(myresult, function(px) { smooth.spline(px,nknots=5)$y  } ))
  
  return(myresult)
}




churnperfNN2 <- LearnNNChurnNN(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, SeqChurn=seq(1, 121, 2), epochsTune=0)
churnperfNNOrig2 <- churnperfNN2
write.csv(churnperfNN2, file = "churnperfNN.csv", row.names = T)

churnperfNN2 <- read.csv(file = "churnperfNN.csv", header = T )
churnperfNNSmooth2 <- data.frame(sapply(churnperfNN2, function(px) { smooth.spline(px,nknots=5)$y  } ))
churnperfNN2 <- churnperfNNSmooth2
PlotNN(churnperfNN2)

#-- From above 
#-- Accuracy , Rsqr improces significantly as epochs grows stabalises araound epochs=110 

churnperfNNOrig2 <- LearnNNChurnNN(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, SeqChurn=seq(121, 221, 20), epochsTune=21)

write.csv(churnperfNNOrig2, file = "churnperfNN2.csv", row.names = T)
churnperfNN2 <- churnperfNNOrig2
PlotNN(churnperfNN)

# Neurons from 121 to 221  epochsTune=21
# Accuracy stays @ 0.80 and r2 at 0.4-0.5


churnperfNNOrig2 <- LearnNNChurnNN(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, SeqChurn=seq(121, 181, 40), epochsTune=200)
write.csv(churnperfNNOrig2, file = "churnperfNN2.csv", row.names = T)


# we reduce max  neurons from 221 to 181 
# becuse higher neorons are no longer improving performanceneurons a
# Neurons from 121 to  181  epochsTune=220
# Accuracy stays @ 0.80 and r2 at 0.4-0.5


LearnNNChurnNNIterateepochs <- function(independentVar, churnName,trainFull, neurons=111, epochsTune=seq(421,1000,100)) {
  neuronList <- list()
  epochList <- list()
  churnr2listSig  <- list()
  churnNoErrorListSig <- list()
  churnYesErrorListSig <- list()
  churnAccuracySig <- list()
  
  actList <- list()
  epochs=epochsTune
  for (i in epochsTune) {
    print( paste0("iteration number:", i ))
    churnModel <- h2o.deeplearning(x = independentVar,
                                   y =  churnName,
                                   distribution =  "multinomial",
                                   activation="Tanh",
                                   seed=100,
                                   training_frame = trainFull,
                                   hidden = c(neurons,neurons,neurons) ,
                                   epochs=i)
    
    # Test the model on the test data
    churntPrediction <- h2o.predict(churnModel, trainFull)
    churnperf <- h2o.performance(model = churnModel, newdata = trainFull)
    confmatrix <-  h2o.confusionMatrix(churnperf)
    
    churnr2listSig[[i]] <- h2o.r2(churnperf)
    churnNoErrorListSig[[i]]  <- confmatrix$Error[1]
    churnYesErrorListSig[[i]] <- confmatrix$Error[2]
    churnAccuracySig[[i]] <- mean(churnperf@metrics$thresholds_and_metric_scores$accuracy)
    
    neuronList[[i]] <- neurons
    epochList[[i]] <- i
  }
  
  churnr2Sig        <- unlist(churnr2listSig)
  churnNoErrorSig   <- unlist(churnNoErrorListSig)
  churnYesErrorSig  <- unlist(churnYesErrorListSig)
  churnAccuracySig  <- unlist(churnAccuracySig)
  
  neurons  <-  unlist(neuronList)
  epoch <- unlist(epochList)
  myresult <- data.frame(neurons, epoch, churnr2Sig, churnNoErrorSig, churnYesErrorSig, churnAccuracySig )
  #myresult <- data.frame(sapply(myresult, function(px) { smooth.spline(px,nknots=5)$y  } ))
  
  return(myresult)
}

churnperfNNOrig3 <- LearnNNChurnNNIterateepochs(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, neurons=111, epochsTune=seq(521,721,100))
write.csv(churnperfNNOrig3, file = "churnperfNN3.csv", row.names = T)
PlotNNSig(churnperfNNOrig3)

# Result
#neurons epoch churnr2Sig churnNoErrorSig churnYesErrorSig churnAccuracySig
#1     111   521  0.8096673      0.03725007       0.06020328        0.9217226
#2     111   621  0.7147727      0.04793207       0.12040657        0.8917216
#3     111   721  0.7762989      0.04026294       0.08444097        0.9124599

churnperfNNOrig4 <- LearnNNChurnNNIterateepochs(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, neurons=111, epochsTune=seq(221,321,100))
write.csv(churnperfNNOrig4, file = "churnperfNN4.csv", row.names = T)
PlotNNSig(churnperfNNOrig4)

#--neurons epoch churnr2Sig churnNoErrorSig churnYesErrorSig churnAccuracySig
#--1     111   221  0.7169009      0.04245412        0.1282252        0.8957252
#--2     111   321  0.7807913      0.02711586        0.1211884        0.9095588
churnperfNNOrig5 <- LearnNNChurnNNIterateepochs(independentVar=independentVars, churnName=churnlabel, trainFull=h2otrainFull, neurons=111, epochsTune=seq(221,321,100))


###---Final   Model , Iterate on Neorons  plus epochs

#My Best Model 
#   neurons epoch churnr2Sig    churnNoErrorSig  churnYesErrorSig churnAccuracySig
#     121   1024  0.8096673      0.03725007       0.06020328        0.9217226


#activation="TanhWithDropout",
#l1 = 1e-8,
#hidden_dropout_ratio = c(0.1,0.1,0.1 ),

validationsize <- round(nrow(trainFull)*0.20)
validationsampleIdx <- sample(nrow(trainFull), validationsize, replace=FALSE)

validationsample <- trainFull[validationsampleIdx,]
trainsample <- trainFull[-validationsampleIdx,]

h2ovalidationsample <- as.h2o( validationsample  )
h2otrainsample <- as.h2o( trainsample  )

churnModelBest <- h2o.deeplearning(x = independentVars,
                                   y =  churnlabel,
                                   distribution =  "multinomial",
                                   seed=100,
                                   activation="TanhWithDropout",
                                   l1 = 1e-7,
                                   hidden_dropout_ratio = c(0.001,0.001,0.001 ),
                                   training_frame = h2otrainsample,
                                   validation_frame = h2ovalidationsample,
                                   hidden = c(181,181,181) ,
                                   epochs=1024)

#-- Evaluation on Full Data: 
churntPredictionBest <- h2o.predict(churnModelBest, h2otrainFull)
churnperfBest <- h2o.performance(model = churnModelBest, newdata = h2otrainFull)
confmatrixBest <-  h2o.confusionMatrix(churnperfBest)
h2o.r2(churnperfBest)
mean(data.frame(h2o.accuracy(churnperfBest))$accuracy)
mean(h2o.sensitivity(churnperfBest)$tpr)
mean(h2o.specificity(churnperfBest)$tnr)

churntPredictionBest <- h2o.predict(churnModelBest, h2ovalidationsample)
churnperfBest <- h2o.performance(model = churnModelBest, newdata = h2ovalidationsample)
confmatrixBest <-  h2o.confusionMatrix(churnperfBest)
h2o.r2(churnperfBest)
mean(data.frame(h2o.accuracy(churnperfBest))$accuracy)
mean(h2o.sensitivity(churnperfBest)$tpr)
mean(h2o.specificity(churnperfBest)$tnr)


##############################################################################
#My Best Model 

#Hyper Parameters : 
#  epochs:1024
#Neurons : 181  
# Layers :3
#hidden_dropout_ratio: 0.1%
#l1:  1e-7
#activation:"TanhWithDropout"

##############################################################################
## Results:
##  Train & Test on Training Data 
## Evaluation on Full Training  data :
##  h2o.r2(churnperfBest) : 0.7239016
##  mean(data.frame(h2o.accuracy(churnperfBest))$accuracy) : 0.9004239
## max(data.frame(h2o.accuracy(churnperfBest))$accuracy) :  0.9346856
## Train on 80% Subset Training Data :
##  Evaluation on Full Training  data :
##  h2o.r2(churnperfBest) : 0.1588853
##  mean(data.frame(h2o.accuracy(churnperfBest))$accuracy) : 0.698247
## max(data.frame(h2o.accuracy(churnperfBest))$accuracy):  0.7860041
## mean(h2o.sensitivity(churnperfBest)$tpr): 0.6508268
## mean(h2o.specificity(churnperfBest)$tnr): 0.7148589
## Evaluation on data :
##  h2o.r2(churnperfBest) : 0.1769442
## mean(data.frame(h2o.accuracy(churnperfBest))$accuracy) : 0.7139594
## max(data.frame(h2o.accuracy(churnperfBest))$accuracy) : 0.7796954
## mean(h2o.sensitivity(churnperfBest)$tpr) : 0.592751
## mean(h2o.specificity(churnperfBest)$tnr) : 0.7572004


