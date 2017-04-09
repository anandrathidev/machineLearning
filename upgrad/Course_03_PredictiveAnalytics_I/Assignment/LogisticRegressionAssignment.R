

#install.packages("car") #for VIF
#install.packages("Hmisc")
#install.packages("ROCR")
#install.packages("caret")
#install.packages("caTools")


library(car) #for VIF
library(Hmisc)
library(ROCR)
library(caret)
library(caTools)

#Checkpoint 1: Data Understanding and Data Exploration
#TASK: You have to just predict whether the customer will default or not.
setwd(dir = "C:\\Users\\rb117\\Documents\\Personal\\Upgrad\\LogisticRegression\\")
german_credit <- read.csv("german.csv")

#Understand Data
summary(german_credit)

sum(is.null(german_credit))==0

# Chec if any thing needs to be in factors
str(german_credit)


#Checkpoint 2: Data Cleaning and Transformation
#In this step, you first need to identify the missing values
sum(is.null(german_credit))==0

#and outliers for each variable and impute them accordingly.
quantile(german_credit$Duration.in.month, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Credit.amount, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Installment.rate.in.percentage.of.disposable.income, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Present.residence.since, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Age.in.Years, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Number.of.existing.credits.at.this.bank., c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Number.of.people.being.liable.to.provide.maintenance.for., c(0.95,0.96,0.97,0.98,0.99,1))

# No Outliers Found
#Generate dummy variables for factor variables. 
#(Certain variables which should be ideally of factor type might be of 
# character or integer type. 
# you need to first convert such variables into factor variables using "as.factor()")

set.seed(1000)
split_german_credit = sample.split(german_credit$Default_status, SplitRatio = 0.7)
table(split_german_credit)
german_credit_train = german_credit[split_german_credit,]
german_credit_test = german_credit[!(split_german_credit),]




