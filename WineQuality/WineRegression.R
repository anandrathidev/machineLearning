##
##  The task is to build a model to predict the wine-quality using 
##  the independent attributes and identify the attributes which 
##  impact quality the most.
##

## check & Install R packages 
list.of.packages <- c("ggplot2","corrplot", "car")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(ggplot2)
library(corrplot)
library(caret)
library(car)

# ***************************************************************************
#                   LOAD DATA  ----
# ***************************************************************************

dpath =  "C:/Users/rb117/Documents/personal/WineQuality/"
#dpath = "D:/Users/anandrathi/Documents/Work/InterView/Wine/"

wine_red_init <- read.csv( paste0(dpath , "winequality-red.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_red_init)
wine_white_init <- read.csv( paste0(dpath , "winequality-white.csv"),stringsAsFactors = FALSE, sep = ';')
str(wine_white_init)

#wine_red_init$quality <- as.factor(wine_red_init$quality)
#wine_white_init$quality <- as.factor(wine_white_init$quality)

summary(wine_red_init$quality)
summary(wine_white_init$quality)

str(wine_red_init$quality)
str(wine_white_init$quality)

wine_red_init$red <- 1
wine_white_init$red <- 0

wine_all_init <- rbind(wine_red_init, wine_white_init)
str(wine_all_init)
library(corrplot)
wcor <- cor(subset(wine_all_init, select = -quality))
corrplot(wcor)


#strong correllation 
#density/alchohol  -0.686745422 

df <- wine_all_init
str(df)
train.index <- createDataPartition(df$quality, p = .9, list = FALSE)
train <- df[ train.index,]
test  <- df[-train.index,]

all.mod1 = lm(quality ~ ., data = train)
summary(all.mod1)
vif(all.mod1)
