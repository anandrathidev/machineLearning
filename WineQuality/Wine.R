##
##  The task is to build a model to predict the wine-quality using 
##  the independent attributes and identify the attributes which 
##  impact quality the most.
##

## check & Install R packages 
list.of.packages <- c( "lubridate","ggplot2","MASS","dplyr","e1071","DMwR","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest", "lift", "plotROC", "compare")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)


c( "lubridate","ggplot2","MASS","dplyr","e1071","DMwR","caret","caretEnsemble","MLmetrics","pROC","ROCR","reshape","cluster","fpc","missForest")
#install.packages('missForest')
library(plotROC)
library(lubridate)
library(ggplot2)
library(MASS)
library(dplyr)
library(e1071)
#library(ROSE)
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

# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************

#dpath =  "C:/Users/rb117/Documents/personal/WineQuality/"
dpath = "D:/Users/anandrathi/Documents/Work/InterView/Wine/"

wine_red_init <- read.csv( paste0(dpath , "winequality-red.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_red_init)
wine_white_init <- read.csv( paste0(dpath , "winequality-white.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_white_init)

wine_red_init$quality <- as.factor(wine_red_init$quality)
wine_white_init$quality <- as.factor(wine_white_init$quality)

str(wine_red_init$quality)
str(wine_white_init$quality)

summary(wine_red_init$quality)

# look into Package ‘unbalanced’

wine_red_bal <- SMOTE(quality ~ ., wine_red_init, perc.over = 10000, perc.under=500)
table(wine_red_bal$quality)

wine_red_bal <- ROSE(quality ~ ., data = wine_red_init, seed = 42)$data
wine_white_bal <- ROSE(quality ~ ., data = wine_white_init, seed = 42)$data

if(F) {
  
  wine_red_init$RedWhite <- 1
  wine_white_init $RedWhite <- 2
  
  # Merge both data 
  wine_all_init <- rbind(wine_red_init, wine_white_init)
  str(wine_all_init)
  set.seed(42)
  
}


## Build Model 
