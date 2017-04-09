#------------------------------------------------------------------------------------------------------------------------------
#-----------------------------Churn Assignment -----------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
#install.packages('class')

library(MASS)
library(car)
library(e1071)
library(stats)
library(gplots)
library(ROCR)
library(lattice)
library(ggplot2)
library(caret)
library(class)
library(caTools)
library(survival)

library(Hmisc)


#####################################################################
#--Checkpoint-1: Data Understanding and Preparation of Master File.
#####################################################################
#--Load the 3 files given in separate data frames.
#--Collate the data together in one single file.
#setwd("C:/Sandy Files +++/Upgrad files/Course 3_PA1 _ Churn Assignment _ group")
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\TelecomChurnCaseStudy\\")

cust_data <- read.csv("customer_data.csv", stringsAsFactors = F)
churn_data <- read.csv("churn_data.csv", stringsAsFactors = F) 
internet_data <- read.csv("internet_data.csv", stringsAsFactors = F)

# Merge all three data set to form one final  

churn <- Reduce(function(x, y) merge(x, y, all=TRUE), list(cust_data, churn_data, internet_data))

sum(is.na(churn))==0



dupchurn <- churn[which(duplicated(subset(churn))==TRUE),]

str(churn)

# Outlier check - no outliers present in data
ggplot(churn,aes(x=TotalCharges, y=TotalCharges)) + geom_boxplot()
ggplot(churn,aes(x=tenure, y=tenure)) + geom_boxplot()
ggplot(churn,aes(x=MonthlyCharges, y=MonthlyCharges)) + geom_boxplot()

# Remove NA from the analsis as it forms 0.3%of the total data
churn_1 <- na.omit(churn)

str(churn_1)



# for Gender  as it is a binary variable
churn_1$gender <- as.factor(churn_1$gender)
levels(churn_1$gender) <- c(1,0)
churn_1$gender <- as.numeric(levels(churn_1$gender))[churn_1$gender]

# for senior citizen  as it is a binary variable
churn_1$SeniorCitizen <- as.factor(churn_1$SeniorCitizen)
levels(churn_1$SeniorCitizen) <- c(1,0)
churn_1$SeniorCitizen <- as.numeric(levels(churn_1$SeniorCitizen))[churn_1$SeniorCitizen]

# for Partner  as it is a binary variable
churn_1$Partner <- as.factor(churn_1$Partner)
levels(churn_1$Partner) <- c(1,0)
churn_1$Partner <- as.numeric(levels(churn_1$Partner))[churn_1$Partner]

# for Dependents  as it is a binary variable
churn_1$Dependents <- as.factor(churn_1$Dependents)
levels(churn_1$Dependents) <- c(1,0)
churn_1$Dependents <- as.numeric(levels(churn_1$Dependents))[churn_1$Dependents]

# for PhoneService  as it is a binary variable
churn_1$PhoneService <- as.factor(churn_1$PhoneService)
levels(churn_1$PhoneService) <- c(1,0)
churn_1$PhoneService <- as.numeric(levels(churn_1$PhoneService))[churn_1$PhoneService]

# for Contract  as it is a 3 level variable
churn_1$Contract <- as.factor(churn_1$Contract)
summary(churn_1$Contract)
churn_1$Contract <- as.factor(churn_1$Contract)
levels(churn_1$Contract) <- c(0,1,2)
churn_1$Contract <- as.numeric(levels(churn_1$Contract))[churn_1$Contract]

# for PaperlessBilling  as it is a binary variable
churn_1$PaperlessBilling <- as.factor(churn_1$PaperlessBilling)
summary(churn_1$PaperlessBilling)
levels(churn_1$PaperlessBilling) <- c(0,1)
churn_1$PaperlessBilling <- as.numeric(levels(churn_1$PaperlessBilling))[churn_1$PaperlessBilling]

# for churn  as it is a binary variable
churn_1$Churn <- as.factor(churn_1$Churn)
summary(churn_1$Churn)
levels(churn_1$Churn) <- c(0,1)
churn_1$Churn <- as.numeric(levels(churn_1$Churn))[churn_1$Churn]

# for MultipleLines  as it is a 3 level variable
churn_1$MultipleLines <- as.factor(churn_1$MultipleLines)
summary(churn_1$MultipleLines)
churn_1$MultipleLines <- as.factor(churn_1$MultipleLines)
levels(churn_1$MultipleLines) <- c(1,0,2)
churn_1$MultipleLines <- as.numeric(levels(churn_1$MultipleLines))[churn_1$MultipleLines]


# for OnlineSecurity  as it is a 3 level variable
churn_1$OnlineSecurity <- as.factor(churn_1$OnlineSecurity)
summary(churn_1$OnlineSecurity)
churn_1$OnlineSecurity <- as.factor(churn_1$OnlineSecurity)
levels(churn_1$OnlineSecurity) <- c(1,0,2)
churn_1$OnlineSecurity <- as.numeric(levels(churn_1$OnlineSecurity))[churn_1$OnlineSecurity]

# for OnlineBackup  as it is a 3 level variable
churn_1$OnlineBackup <- as.factor(churn_1$OnlineBackup)
summary(churn_1$OnlineBackup)
churn_1$OnlineBackup <- as.factor(churn_1$OnlineBackup)
levels(churn_1$OnlineBackup) <- c(1,0,2)
churn_1$OnlineBackup <- as.numeric(levels(churn_1$OnlineBackup))[churn_1$OnlineBackup]

# for DeviceProtection  as it is a 3 level variable
churn_1$DeviceProtection <- as.factor(churn_1$DeviceProtection)
summary(churn_1$DeviceProtection)
churn_1$DeviceProtection <- as.factor(churn_1$DeviceProtection)
levels(churn_1$DeviceProtection) <- c(1,0,2)
churn_1$DeviceProtection <- as.numeric(levels(churn_1$DeviceProtection))[churn_1$DeviceProtection]

# for TechSupport  as it is a 3 level variable
churn_1$TechSupport <- as.factor(churn_1$TechSupport)
summary(churn_1$TechSupport)
churn_1$TechSupport <- as.factor(churn_1$TechSupport)
levels(churn_1$TechSupport) <- c(1,0,2)
churn_1$TechSupport <- as.numeric(levels(churn_1$TechSupport))[churn_1$TechSupport]

# for StreamingTV  as it is a 3 level variable
churn_1$StreamingTV <- as.factor(churn_1$StreamingTV)
summary(churn_1$StreamingTV)
churn_1$StreamingTV <- as.factor(churn_1$StreamingTV)
levels(churn_1$StreamingTV) <- c(1,0,2)
churn_1$StreamingTV <- as.numeric(levels(churn_1$StreamingTV))[churn_1$StreamingTV]

# for StreamingMovies  as it is a 3 level variable
churn_1$StreamingMovies <- as.factor(churn_1$StreamingMovies)
summary(churn_1$StreamingMovies)
churn_1$StreamingMovies <- as.factor(churn_1$StreamingMovies)
levels(churn_1$StreamingMovies) <- c(1,0,2)
churn_1$StreamingMovies <- as.numeric(levels(churn_1$StreamingMovies))[churn_1$StreamingMovies]

churn_1$Churn <- as.factor(churn_1$Churn)

dummy1 <- as.data.frame(model.matrix( ~ PaymentMethod -1, data = churn_1))
dummy2 <- as.data.frame(model.matrix( ~ InternetService -1, data = churn_1))

# Scaling the numeric variables 
churn_1$tenure1 <- scale(churn_1$tenure)
churn_1$MonthlyCharges1 <- scale(churn_1$MonthlyCharges)
churn_1$TotalCharges1 <- scale(churn_1$TotalCharges)

churn_final <- cbind(dummy1[,-1], dummy2[,-1], churn_1[, c(2,4:13,15:18,23:25,22)] )

# EDA plts for variable information
#Continous variables
tenure_plot <- ggplot(churn_final, aes(x=tenure)) 
tenure_plot <- tenure_plot + geom_density()
tenure_plot

monthly_charge_plot <- ggplot(churn_1, aes(x=MonthlyCharges)) 
monthly_charge_plot <- monthly_charge_plot + geom_density()
monthly_charge_plot

Total_charge_plot <- ggplot(churn_1, aes(x=TotalCharges)) 
Total_charge_plot <- Total_charge_plot + geom_density()
Total_charge_plot

# Categorical variables
gender_plot <- ggplot(churn_1, aes(x=gender))
gender_plot <- gender_plot  + geom_bar() 
gender_plot

contract_plot <- ggplot(churn_1, aes(x=Contract))
contract_plot <- contract_plot  + geom_bar() 
contract_plot

PaymentMethod_plot <- ggplot(churn_1, aes(x=PaymentMethod))
PaymentMethod_plot <- PaymentMethod_plot + theme(axis.text.x = element_text(angle = 90, hjust = 1)) + geom_bar() 
PaymentMethod_plot





# Splitting into training and testing
set.seed(9850)
s=sample(1:nrow(churn_final),0.7*nrow(churn_final))

# training data contains 70% of the data
knn_train=churn_final[s,]

#testing data contains 30% of the data
knn_test=churn_final[-s,]

knn_train_target <- knn_train[,24]
knn_test_target <- knn_test[,24]


str(knn_train)

#-----------------------------------------------------------------------------------------------------------------------------------
#--------------------Model for KNN -------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------



#Using the train() command to find the best K.
KNNmodel <- train(
  Churn~., 
  data=knn_train,
  method='knn',
  tuneGrid=expand.grid(.k=1:50),
  metric='Accuracy',
  trControl=trainControl(method='repeatedcv', number=10, repeats=15))
KNNmodel
plot(KNNmodel)


Churn_knnM <- knn(train=knn_train,test=knn_test, cl=knn_train_target, k = 21)

Churn_knnMq <- as.numeric(Churn_knnM)

table(Churn_knnM, knn_test_target)
confusionMatrix(Churn_knnM, knn_test_target, positive = "1")




library(ROCR)
#calculating the values for ROC curve
pred <- prediction(Churn_knnMq, knn_test_target)
perf <- performance(pred,"tpr","fpr")



# changing params for the ROC plot - width, etc
par(mar=c(5,5,2,2),xaxs = "i",yaxs = "i",cex.axis=1.3,cex.lab=1.4)

# plotting the ROC curve
plot(perf,col="black",lty=3, lwd=3)

# calculating AUC
auc <- performance(pred,"auc")
auc#0.8541834

#-----------------------------------------------------------------------------------------------------------------------------------
#--------------------Model for Naive Bayes -------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------


model_naive <- naiveBayes(Churn ~. , data = knn_train)

pred <- predict(model_naive, knn_test)
table(pred, knn_test_target)

confusionMatrix(pred, knn_test_target)

pred1 <- as.numeric(pred)

library(ROCR)
#calculating the values for ROC curve
pred_naive <- prediction(pred1, knn_test_target)
perf <- performance(pred_naive,"tpr","fpr")


# plotting the ROC curve
plot(perf,col="black",lty=3, lwd=3)

# calculating AUC
auc <- performance(pred_naive,"auc")
auc #0.7609306


#-----------------------------------------------------------------------------------------------------------------------------------
#--------------------model for logistic regression------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------


# Model with all variables
initial_model = glm(Churn ~ .-Churn, data = knn_train, family = "binomial")
summary(initial_model)

# Stepwise selection of variables
best_model = step(initial_model,direction = "both")
summary(best_model)

vif(best_model)


Model_1 <- glm(formula = Churn ~ `PaymentMethodElectronic check` + `InternetServiceFiber optic` + 
                 MultipleLines + OnlineBackup + DeviceProtection + StreamingTV + 
                 StreamingMovies + SeniorCitizen + Dependents + tenure + Contract + 
                 PaperlessBilling + MonthlyCharges1 + TotalCharges1, family = "binomial", 
               data = knn_train)

summary(Model_1)

vif(Model_1)



Model_2 <- glm(formula = Churn ~ `PaymentMethodElectronic check` + `InternetServiceFiber optic` + 
                 StreamingTV + 
                 StreamingMovies + Dependents + Contract + 
                 PaperlessBilling +  TotalCharges1, family = "binomial", 
               data = knn_train)

summary(Model_2)

vif(Model_2)


# Model performance evaluation  
#-----------------------------------------------------------------------------------------------------
knn_train$predicted_prob = round(predict(Model_2, data=knn_train, type = "response"))
table(knn_train$Churn, knn_train$predicted_prob > 0.5)
rcorr.cens(knn_train$predicted_prob,knn_train$Churn) # 1st argument is your vector of predicted probabilities, 2nd observed values of outcome variable


#C Index            Dxy           S.D.              n        missing     uncensored Relevant Pairs     Concordant      Uncertain 
#7.027808e-01   4.055616e-01   1.478051e-02   4.922000e+03   0.000000e+00   4.922000e+03   9.375242e+06   6.588740e+06   0.000000e+00 

knn_test$predicted_prob = round(predict(Model_2, newdata=knn_test, type = "response"))
rcorr.cens(knn_test$predicted_prob,knn_test$Churn)

#C Index            Dxy           S.D.              n        missing     uncensored Relevant Pairs     Concordant      Uncertain 
#7.116373e-01   4.232746e-01   2.213567e-02   2.110000e+03   0.000000e+00   2.110000e+03   1.770992e+06   1.260304e+06   0.000000e+00

#KS-statistic for the model---------------------------------------------------------------------------------------------------------------------

model_score <- prediction(knn_train$predicted_prob,knn_train$Churn)

model_perf <- performance(model_score, "tpr", "fpr")

ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf, "x.values")[[1]])

ks = max(ks_table)

which(ks_table == ks)

# KS for train = 2

model_score_test <- prediction(knn_test$predicted_prob,knn_test$Churn)

model_perf_test <- performance(model_score_test, "tpr", "fpr")

ks_table_test <- attr(model_perf_test, "y.values")[[1]] - (attr(model_perf_test, "x.values")[[1]])
ks_test = max(ks_table_test)
which(ks_table_test == ks_test)

# KS for train = 2

2/2110  #  = [1] 0.0009478673



table(knn_test$predicted_prob,knn_test$Churn)

confusionMatrix(knn_test$predicted_prob,knn_test$Churn)



library(ROCR)
#calculating the values for ROC curve
model_score_test <- prediction(knn_test$predicted_prob,knn_test$Churn)

model_perf_test <- performance(model_score_test, "tpr", "fpr")


# plotting the ROC curve
plot(model_perf_test,col="black",lty=3, lwd=3)

# calculating AUC
auc <- performance(model_score_test,"auc")
auc #0.7116373


#-----------------------------------------------------------------------------------------------------------------------------------
#--------------------model for SVM------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------
library(Formula)
str(knn_train)
tune.svm = tune(svm, Churn ~.-predicted_prob, data = knn_train, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 100)))
summary(tune.svm)

best.mod =tune.svm$best.model

ypred[1,1] = predict(best.mod, knn_test)
table(predicted=ypred, knn_test$Churn)
confusionMatrix(ypred, knn_test$Churn)

model_svm_test <- prediction(ypred,knn_test$Churn)

model_perf_test <- performance(model_svm_test, "tpr", "fpr")


# plotting the ROC curve
plot(model_perf_test,col="black",lty=3, lwd=3)

# calculating AUC
auc <- performance(model_score_test,"auc")
auc #0.7116373



knn_test1 <- knn_test
str(knn_test1)
knn_test1$Churn <- as.numeric(knn_test1$Churn)
plot(best.mod, knn_test1)


model.svm.2 = svm(Churn ~.-predicted_prob, data = knn_test1, kernel = "linear", cost = 0.01, scale = F)  
summary(model.svm.2)
plot(model.svm.2, knn_train)

ypred = predict(model.svm.2, knn_test)
table(predicted=ypred, knn_test$Churn)
confusionMatrix(ypred, knn_test$Churn)










