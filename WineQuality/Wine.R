##
##  The task is to build a model to predict the wine-quality using 
##  the independent attributes and identify the attributes which 
##  impact quality the most.
##

## check & Install R packages 
list.of.packages <- c( "lubridate","ggplot2","MASS","dplyr","e1071","DMwR","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest", "lift", "plotROC", "compare", "xgboost")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#install.packages('missForest')
library(plotROC)
library(lubridate)
library(ggplot2)
library(MASS)
library(dplyr)
library(e1071)
library(ROSE)
library(DMwR)

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
library(compare)
library(xgboost)
# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************

dpath =  "C:/Users/rb117/Documents/personal/WineQuality/"
#dpath = "D:/Users/anandrathi/Documents/Work/InterView/Wine/"

wine_red_init <- read.csv( paste0(dpath , "winequality-red.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_red_init)
wine_white_init <- read.csv( paste0(dpath , "winequality-white.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_white_init)

wine_red_init$quality <- as.factor(wine_red_init$quality)
wine_white_init$quality <- as.factor(wine_white_init$quality)

summary(wine_red_init$quality)
summary(wine_white_init$quality)

str(wine_red_init$quality)
str(wine_white_init$quality)

#df <- wine_red_init
# look into Package ‘unbalanced’
FuncRedbalance <- function(df ) {
  dforig <- df
  df$quality <- as.numeric(as.character(df$quality))

  wine_3_5 <- df[which(df$quality==3 | df$quality==5),]
  wine_4_5 <- df[which(df$quality==4 | df$quality==5),]
  wine_7_5 <- df[which(df$quality==7 | df$quality==5),]
  wine_8_5 <- df[which(df$quality==8 | df$quality==5),]

  wine_3_5$quality <- as.factor(wine_3_5$quality)
  wine_4_5$quality <- as.factor(wine_4_5$quality)
  wine_7_5$quality <- as.factor(wine_7_5$quality)
  wine_8_5$quality <- as.factor(wine_8_5$quality)
  
  wine_bal_3_5 <- ROSE(quality ~ . , data = wine_3_5, seed=42)$data
  wine_bal_4_5 <- ROSE(quality ~ . , data = wine_4_5, seed=42)$data
  wine_bal_7_5 <- ROSE(quality ~ . , data = wine_7_5, seed=42)$data
  wine_bal_8_5 <- ROSE(quality ~ . , data = wine_8_5, seed=42)$data
  
  wine_bal_3_5$quality <- as.numeric(as.character(wine_bal_3_5$quality))
  wine_bal_4_5$quality <- as.numeric(as.character(wine_bal_4_5$quality))
  wine_bal_7_5$quality <- as.numeric(as.character(wine_bal_7_5$quality))
  wine_bal_8_5$quality <- as.numeric(as.character(wine_bal_8_5$quality))
  
  df <- rbind(wine_bal_3_5,wine_bal_4_5, wine_bal_7_5, wine_bal_8_5)
  df <- df[df$quality!=5,]
  df5 <- dforig[dforig$quality==5,]
  df6 <- dforig[dforig$quality==6,]
  df <- rbind(df,df5, df6)
  df$quality <- as.factor(df$quality)
  return (df)
}

FuncWhitebalance <- function(df ) {
  dforig <- df
  df$quality <- as.numeric(as.character(df$quality))
  wine_3_6 <- df[which(df$quality==3 | df$quality==6),]
  wine_4_6 <- df[which(df$quality==4 | df$quality==6),]
  wine_5_6 <- df[which(df$quality==5 | df$quality==6),]
  wine_7_6 <- df[which(df$quality==7 | df$quality==6),]
  wine_8_6 <- df[which(df$quality==8 | df$quality==6),]
  wine_9_6 <- df[which(df$quality==9 | df$quality==6),]
  
  wine_3_6$quality <- as.factor(wine_3_6$quality)
  wine_4_6$quality <- as.factor(wine_4_6$quality)
  wine_5_6$quality <- as.factor(wine_5_6$quality)
  wine_7_6$quality <- as.factor(wine_7_6$quality)
  wine_8_6$quality <- as.factor(wine_8_6$quality)
  wine_9_6$quality <- as.factor(wine_9_6$quality)
  
  wine_bal_3_6 <- ROSE(quality ~ . , data = wine_3_6, seed=42)$data
  wine_bal_4_6 <- ROSE(quality ~ . , data = wine_4_6, seed=42)$data
  wine_bal_5_6 <- ROSE(quality ~ . , data = wine_5_6, seed=42)$data
  wine_bal_7_6 <- ROSE(quality ~ . , data = wine_7_6, seed=42)$data
  wine_bal_8_6 <- ROSE(quality ~ . , data = wine_8_6, seed=42)$data
  wine_bal_9_6 <- ROSE(quality ~ . , data = wine_9_6, seed=42)$data
  
  wine_bal_3_6$quality <- as.numeric(as.character(wine_bal_3_6$quality))
  wine_bal_4_6$quality <- as.numeric(as.character(wine_bal_4_6$quality))
  wine_bal_5_6$quality <- as.numeric(as.character(wine_bal_5_6$quality))
  wine_bal_7_6$quality <- as.numeric(as.character(wine_bal_7_6$quality))
  wine_bal_8_6$quality <- as.numeric(as.character(wine_bal_8_6$quality))
  wine_bal_9_6$quality <- as.numeric(as.character(wine_bal_9_6$quality))
  
  df <- rbind(wine_bal_3_6, wine_bal_4_6, wine_bal_5_6, wine_bal_7_6, wine_bal_8_6, wine_bal_9_6)
  df <- df[df$quality!=6,]
  df6 <- dforig[dforig$quality==6,]
  df <- rbind(df, df6)
  df$quality <- as.factor(df$quality)
  return (df)
}

wine_red_bal <- FuncRedbalance(wine_red_init)
wine_white_bal <- FuncWhitebalance(wine_white_init)

summary(wine_red_init$quality)
summary(wine_red_bal$quality)

summary(wine_white_init$quality)
summary(wine_white_bal$quality)

if(F) {
  wine_red_init$RedWhite <- 1
  wine_white_init $RedWhite <- 2
  # Merge both data 
  wine_all_init <- rbind(wine_red_init, wine_white_init)
  str(wine_all_init)
  set.seed(42)
}


## stratified  sampling 
## split the data using stratified  sampling 
df <- wine_red_bal
train.index <- createDataPartition(df$quality, p = .7, list = FALSE)
trainRed <- df[ train.index,]
testRed  <- df[-train.index,]
prop.table(table(df$quality))
prop.table(table(trainRed$quality))
prop.table(table(testRed$quality))

df <- wine_white_bal
train.index <- createDataPartition(df$quality, p = .7, list = FALSE)
trainWhite <- df[ train.index,]
testWhite  <- df[-train.index,]
prop.table(table(df$quality))
prop.table(table(trainWhite$quality))
prop.table(table(testWhite$quality))


## Build Model 

install.packages("archdata")
library(archdata) 
BuildModel <- function(df) {
  set.seed(42)
  levels(df$quality) <- make.names(levels(factor(df$quality)))
  fitControl <- trainControl( method = "repeatedcv", number = 7, savePredictions = T, classProbs = T,  summaryFunction = multiClassSummary, seeds=42)
  data_label <- as.numeric(as.character(df[,"quality"]))
  numberOfClasses <- max(length(unique(data_label)))
  numberOfClasses <- max((unique(data_label)))+1
  
  xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = numberOfClasses)
  nround    <- 20 # number of XGBoost rounds
  cv.nfold  <- 10
  data_label <- as.numeric(as.character(df[,"quality"]))
  data_matrix <- xgb.DMatrix(data = as.matrix(subset(df, select = -c(quality))) , label = data_label)
  model_xgs <- xgb.train(params = xgb_params, data = data_matrix, nrounds = nround)
  
  names <-  colnames(subset( df, select=-c(quality)))
  importance_matrix = xgb.importance(feature_names = names, model = model_xgs)
  plot(importance_matrix)
  
  models_rf <- train(quality ~ . , data=df, trControl=fitControl, method='rf', metric="Accuracy" , verbose = F, importance = TRUE)
  importance_rf <- varImp(models_rf)
  
  models_nn <- train(quality ~ . , data=df, trControl=fitControl, method='dnn', metric="Accuracy" )
  importance_nn <- varImp(models_nn)
  
  models_knn <- train(quality ~ . , data=df,  trControl=fitControl, method='knn', metric="Accuracy" )
  importance_knn <- varImp(models_nn)
  #gp = xgb.plot.importance(importance_matrix)
  #print(gp) 
  plot(importance_matrix, main =" Important variables- XGBoost")
  plot(importance_rf, main =" Important variables- RF")
  plot(importance_nn, main =" Important variables- Deep NN")
  plot(importance_knn, main =" Important variables- K NN")
  
}
df = trainRed
BuildModel(trainRed)
BuildModel(trainWhite)
