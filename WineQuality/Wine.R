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

## Check NA 
sum(is.na(wine_red_init))
sum(is.na(wine_white_init))

#df <- wine_red_init
# look into Package 'unbalanced'
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

BuildModel <- function(df) {
  set.seed(42)
  fitControl <- trainControl( method = "repeatedcv", number = 7, savePredictions = T, classProbs = T,  summaryFunction = multiClassSummary)
  #data_label <- as.numeric(as.character(df[,"quality"]))
  data_label <- df[,"quality"]
  numberOfClasses <- length(unique(data_label))+1
  xgb_params <- list("objective" = "multi:softprob", "eval_metric" = "mlogloss", "num_class" = numberOfClasses)
  nround    <- 20 # number of XGBoost rounds
  cv.nfold  <- 10
  #data_label <- as.numeric(as.character(df[,"quality"]))
  data_matrix <- xgb.DMatrix(data = as.matrix(subset(df, select = -c(quality))) , label = data_label)
  #model_xgs <- xgb.train(params = xgb_params, data = data_matrix, nrounds = nround)
  
  #names <-  colnames(subset( df, select= - c(quality)))
  #importance_matrix = xgb.importance(feature_names = names, model = model_xgs)

  models_rf <- train(quality ~ . , data=df, trControl=fitControl, method='rf', metric="Accuracy" , verbose = F, importance = TRUE)
  importance_rf <- varImp(models_rf)
  
  #gp = xgb.plot.importance(importance_matrix)
  #plot(importance_rf, main =" Important variables- RF")
  #plot(importance_nn, main =" Important variables- Deep NN")
  #return (list(xgs = model_xgs, rf = models_rf))
  return (list( rf = models_rf))
}

dfred = trainRed
levels(dfred$quality)  <- c("three", "four", "five", "six", "seven", "eight")
dfwhite = trainWhite
levels(dfwhite$quality)  <- c("three", "four", "five", "six", "seven", "eight", "nine")

df = dfred
modelListRed <- BuildModel(dfred)
modelListWhite <- BuildModel(dfwhite)
names(modelListRed)
names(modelListWhite)

PredictModel <-  function (x,testData,y) {
  pred_ <- predict(x, newdata = testData )
  table(y, pred_)
  return(pred_) 
}

PredictModelProb <-  function (x,testData,y) {
  pred_  <- predict(x, testData )
  pred_Prob <- predict(x, newdata = testData, type = "prob" , probability = TRUE  )
  table(y, pred_)
  cf <- confusionMatrix(data = y, pred_ )
  me <- list(
    prediction = pred_,
    pred_Prob = pred_Prob,
    confusionMatrix = cf
  )
  
  return(me) 
}

dfTestred = testRed
levels(dfTestred$quality)  <- c("three", "four", "five", "six", "seven", "eight")
dfTestwhite = testWhite
levels(testWhite$quality)  
levels(dfTestwhite$quality)  <- c("three", "four", "five", "six", "seven", "eight", "nine")

# On Train 
predsProbListRed <-  sapply(modelListRed, PredictModelProb, simplify = F, testData=subset(dfred, select = -c(quality) ) , y=dfred$quality )
names(predsProbListRed)
predsProbListWhite <-  sapply(modelListWhite, PredictModelProb, simplify = F, testData=subset(dfwhite, select = -c(quality) ) , y=dfwhite$quality )
names(predsProbListWhite)

# On test  
predsProbListRedTEST <-  sapply(modelListRed, PredictModelProb, simplify = F, testData=subset(dfTestred, select = -c(quality) ) , y=dfTestred$quality )
names(predsProbListRedTEST)
predsProbListWhiteTEST <-  sapply(modelListWhite, PredictModelProb, simplify = F, testData=subset(dfTestwhite, select = -c(quality) ) , y=dfTestwhite$quality )
names(predsProbListWhiteTEST)
#rfwhite <- modelListWhite$rf

#trainProbList <-  sapply(modelListWhite, PredictModelProb, simplify = F, testData= subset(dfwhite, select = -c(quality) ), y=dfwhite$quality )

predsProbListRedTEST$rf$confusionMatrix
'
Confusion Matrix and Statistics

Reference
Prediction three four five six seven eight
three    66   19    6   2     4     0
four     24   55   13   5     6     1
five      2    4  161  36     1     0
six       1    1   34 136    14     5
seven     0   14    5  19    73    14
eight     0    0    2   6    24    67

Overall Statistics

Accuracy : 0.6805          
95% CI : (0.6474, 0.7123)
No Information Rate : 0.2695          
P-Value [Acc > NIR] : < 2.2e-16       

Kappa : 0.6064          
Mcnemars Test P-Value : NA              

Statistics by Class:
  
  Class: three Class: four Class: five Class: six Class: seven Class: eight
Sensitivity               0.70968     0.59140      0.7285     0.6667      0.59836      0.77011
Specificity               0.95736     0.93260      0.9282     0.9107      0.92550      0.95634
Pos Pred Value            0.68041     0.52885      0.7892     0.7120      0.58400      0.67677
Neg Pred Value            0.96266     0.94693      0.9026     0.8919      0.92950      0.97226
Prevalence                0.11341     0.11341      0.2695     0.2488      0.14878      0.10610
Detection Rate            0.08049     0.06707      0.1963     0.1659      0.08902      0.08171
Detection Prevalence      0.11829     0.12683      0.2488     0.2329      0.15244      0.12073
Balanced Accuracy         0.83352     0.76200      0.8284     0.7887      0.76193      0.86323
'



predsProbListWhiteTEST$rf$confusionMatrix
"
Confusion Matrix and Statistics

Reference
Prediction three four five six seven eight nine
three   263   17   20   4     8    10    3
four     26  162  102  10    39     7    5
five     11   60  343  87    32     6    6
six       2    5   45 556    29    17    5
seven     0    3   52  96   208    70   28
eight     3    4   30  32   100   143   39
nine      0    1    0   7    22     5  288

Overall Statistics

Accuracy : 0.6519         
95% CI : (0.6346, 0.669)
No Information Rate : 0.263          
P-Value [Acc > NIR] : < 2.2e-16      

Kappa : 0.586          
Mcnemar's Test P-Value : < 2.2e-16      

Statistics by Class:

                     Class: three Class: four Class: five Class: six Class: seven Class: eight Class: nine
Sensitivity               0.86230     0.64286      0.5794     0.7020      0.47489      0.55426     0.77005
Specificity               0.97709     0.93150      0.9165     0.9536      0.90323      0.92445     0.98673
Pos Pred Value            0.80923     0.46154      0.6294     0.8437      0.45514      0.40741     0.89164
Neg Pred Value            0.98436     0.96617      0.8990     0.8997      0.90995      0.95677     0.96801
Prevalence                0.10130     0.08369      0.1966     0.2630      0.14547      0.08569     0.12421
Detection Rate            0.08735     0.05380      0.1139     0.1847      0.06908      0.04749     0.09565
Detection Prevalence      0.10794     0.11657      0.1810     0.2189      0.15178      0.11657     0.10727
Balanced Accuracy         0.91969     0.78718      0.7479     0.8278      0.68906      0.73935     0.87839
"
