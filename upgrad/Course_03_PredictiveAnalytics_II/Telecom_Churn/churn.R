#install.packages("MASS")
#install.packages("car")
#install.packages("e1071")
#install.packages("ROCR")
#install.packages("caret")

library(MASS)
library(car)
library(e1071)
library(ROCR)
library(caret)
library(caTools)
library(class)

set.seed(100) #-- set seed early so we never miss it


#-- 1. Business Understanding
#-- 2. Data Understanding
#-- 3. What is the Company's Business Objective?
#--Data Understanding
#--Data Preparation
#--Model Building
#--Model Evaluation
#-- Presentation of results

#####################################################################
#--Checkpoint-1: Data Understanding and Preparation of Master File.
#####################################################################
#--Load the 3 files given in separate data frames.
#--Collate the data together in one single file.
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\TelecomChurnCaseStudy\\")
cust_data <- read.csv("customer_data.csv", stringsAsFactors = F)
churn_data <- read.csv("churn_data.csv", stringsAsFactors = F) 
internet_data <- read.csv("internet_data.csv", stringsAsFactors = F)

#
sum(is.na(cust_data))==0
sum(is.na(churn_data))==0
sum(is.na(internet_data))==0

dfMerge <- function(x, y) {
  df <- merge(x, y, by= "customerID", all.x= TRUE, all.y= TRUE)
  return(df)
}

churn <- Reduce(dfMerge, list(cust_data, churn_data, internet_data))
remove(cust_data, churn_data, internet_data)

#####################################################################

#####################################################################
#-- Checkpoint -2: Exploratory Data Analysis
#####################################################################

#-- Make bar charts displaying the relationship between the 
#-- target variable and various other features and report them.
bar_plots_1 <- ggplot(churn, aes(x=Churn, y = gender)) + geom_bar(stat="count") + stat_count(geom = "text", aes(label = paste(round((..count..)), "%")), vjust = 5)

grid.arrange(factor_plots_1,
             factor_plots_2,
             factor_plots_3,
             factor_plots_4,
             factor_plots_5,
             factor_plots_6,
             ncol = 2, top = "Univariate Analysis Factors 1")


#####################################################################
#-- Checkpoint -3: Data Preparation
#####################################################################

#--Perform de-duplication of data.
dupchurn <- churn[ which( duplicated(churn$customerID) ==TRUE), ]
#dupchurn <- churn[which(duplicated(subset(churn, select=-customerID ))==TRUE),]
#---- How many duplicates 
#nrow(dupchurn) # 22
#churn<- churn[!duplicated(subset(churn, select=-customerID )), ]
remove(dupchurn)

#Clean up objects
#--Bring the data in the correct format
str(churn)

#---- 'data.frame':	7043 obs. of  21 variables:
#---- $ customerID      : chr  "0002-ORFBO" "0003-MKNFE" "0004-TLHLJ" "0011-IGKFF" ...
#---- [Factor]$ gender          : chr  "Female" "Male" "Male" "Male" ...
#---- [Factor]$ SeniorCitizen   : int  0 0 0 1 1 0 1 0 1 0 ...
#---- [Factor]$ Partner         : chr  "Yes" "No" "No" "Yes" ...
#---- [Factor]$ Dependents      : chr  "Yes" "No" "No" "No" ...
#---- $ tenure          : int  9 9 4 13 3 9 71 63 7 65 ...
#---- [Factor]$ PhoneService    : chr  "Yes" "Yes" "Yes" "Yes" ...
#---- [Factor]$ Contract        : chr  "One year" "Month-to-month" "Month-to-month" "Month-to-month" ...
#---- [Factor]$ PaperlessBilling: chr  "Yes" "No" "Yes" "Yes" ...
#---- [Factor]$ PaymentMethod   : chr  "Mailed check" "Mailed check" "Electronic check" "Electronic check" ...
#---- $ MonthlyCharges  : num  65.6 59.9 73.9 98 83.9 ...
#---- $ TotalCharges    : num  593 542 281 1238 267 ...
#---- [Factor]$ Churn           : chr  "No" "No" "Yes" "Yes" ...
#---- [Factor]$ MultipleLines   : chr  "No" "Yes" "No" "No" ...
#---- [Factor]$ InternetService : chr  "DSL" "DSL" "Fiber optic" "Fiber optic" ...
#---- [Factor]$ OnlineSecurity  : chr  "No" "No" "No" "No" ...
#---- [Factor]$ OnlineBackup    : chr  "Yes" "No" "No" "Yes" ...
#---- [Factor]$ DeviceProtection: chr  "No" "No" "Yes" "Yes" ...
#---- [Factor]$ TechSupport     : chr  "Yes" "No" "No" "No" ...
#---- [Factor]$ StreamingTV     : chr  "Yes" "No" "No" "Yes" ...
#---- [Factor]$ StreamingMovies : chr  "No" "Yes" "No" "Yes" ...

churn$gender  <- as.factor(churn$gender)
churn$SeniorCitizen  <- as.factor(churn$SeniorCitizen)
churn$Partner  <- as.factor(churn$Partner)
churn$Dependents  <- as.factor(churn$Dependents)
churn$PhoneService  <- as.factor(churn$PhoneService)
churn$Contract  <- as.factor(churn$Contract)
churn$PaperlessBilling  <- as.factor(churn$PaperlessBilling)
churn$PaymentMethod   <- as.factor(churn$PaymentMethod)
churn$Churn  <- as.factor(churn$Churn)
churn$MultipleLines  <- as.factor(churn$MultipleLines)
churn$InternetService  <- as.factor(churn$InternetService) 
churn$OnlineSecurity  <- as.factor(churn$OnlineSecurity)
churn$OnlineBackup  <- as.factor(churn$OnlineBackup) 
churn$DeviceProtection  <- as.factor(churn$DeviceProtection)
churn$TechSupport  <- as.factor(churn$TechSupport)
churn$StreamingTV  <- as.factor(churn$StreamingTV)    
churn$StreamingMovies  <- as.factor(churn$StreamingMovies)

str(churn)
#--Find the variables having missing values and impute them.  
churn_chk_na <- churn[data.frame(which(is.na(churn), arr.ind=TRUE))$row,]
boxplot(churn$TotalCharges, na.rm = T )
churn$TotalCharges[which(is.na(churn$TotalCharges))] <- mean(churn$TotalCharges, na.rm = T)
sum(is.na(churn))==0
na.omit(churn)

remove(churn_chk_na)

#--   Data Prep:
#--   Prepare data for K-NN/logisticreg/ (Don't forget to convert categorical variables to numeric)
churn_dummy_gender  <- data.frame(model.matrix( ~gender ,data=churn))
churn_dummy_SeniorCitizen  <- data.frame(model.matrix( ~SeniorCitizen ,data=churn))
churn_dummy_Partner  <- data.frame(model.matrix( ~Partner ,data=churn))
churn_dummy_Dependents  <- data.frame(model.matrix( ~Dependents ,data=churn))
churn_dummy_PhoneService  <- data.frame(model.matrix( ~PhoneService ,data=churn))
churn_dummy_Contract  <- data.frame(model.matrix( ~Contract ,data=churn))
churn_dummy_PaperlessBilling  <- data.frame(model.matrix( ~PaperlessBilling ,data=churn))
churn_dummy_PaymentMethod   <- data.frame(model.matrix( ~PaymentMethod ,data=churn))
#churn_dummy_Churn  <- data.frame(model.matrix( ~Churn ,data=churn))
churn_dummy_MultipleLines  <- data.frame(model.matrix( ~MultipleLines ,data=churn))
churn_dummy_InternetService  <- data.frame(model.matrix( ~InternetService ,data=churn)) 
churn_dummy_OnlineSecurity  <- data.frame(model.matrix( ~OnlineSecurity ,data=churn))
churn_dummy_OnlineBackup  <- data.frame(model.matrix( ~OnlineBackup ,data=churn)) 
churn_dummy_DeviceProtection  <- data.frame(model.matrix( ~DeviceProtection ,data=churn))
churn_dummy_TechSupport  <- data.frame(model.matrix( ~TechSupport ,data=churn))
churn_dummy_StreamingTV  <- data.frame(model.matrix( ~StreamingTV ,data=churn))    
churn_dummy_StreamingMovies  <- data.frame(model.matrix( ~StreamingMovies ,data=churn))



churn_dummy <- churn
churn_dummy <- cbind(churn_dummy,
                     churn_dummy_gender,
                     churn_dummy_SeniorCitizen  ,
                     churn_dummy_Partner  ,
                     churn_dummy_Dependents  ,
                     churn_dummy_PhoneService  ,
                     churn_dummy_Contract  ,
                     churn_dummy_PaperlessBilling  ,
                     churn_dummy_PaymentMethod   ,
                     churn_dummy_MultipleLines  ,
                     churn_dummy_InternetService  ,
                     churn_dummy_OnlineSecurity  ,
                     churn_dummy_OnlineBackup  ,
                     churn_dummy_DeviceProtection  ,
                     churn_dummy_TechSupport  ,
                     churn_dummy_StreamingTV  ,
                     churn_dummy_StreamingMovies 
)

churn_dummy$tenure <- scale(churn_dummy$tenure)
churn_dummy$MonthlyCharges <- scale(churn_dummy$MonthlyCharges)
churn_dummy$TotalCharges <- scale(churn_dummy$TotalCharges)

remove(churn_dummy_gender,
       churn_dummy_SeniorCitizen  ,
       churn_dummy_Partner  ,
       churn_dummy_Dependents  ,
       churn_dummy_PhoneService  ,
       churn_dummy_Contract  ,
       churn_dummy_PaperlessBilling  ,
       churn_dummy_PaymentMethod   ,
       churn_dummy_MultipleLines  ,
       churn_dummy_InternetService  ,
       churn_dummy_OnlineSecurity  ,
       churn_dummy_OnlineBackup  ,
       churn_dummy_DeviceProtection  ,
       churn_dummy_TechSupport  ,
       churn_dummy_StreamingTV  ,
       churn_dummy_StreamingMovies 
)

churn_dummy <- subset(x=churn_dummy, select = -c(customerID, gender, SeniorCitizen ,
                                                 Partner ,
                                                 Dependents ,
                                                 PhoneService ,
                                                 Contract ,
                                                 PaperlessBilling ,
                                                 PaymentMethod  ,
                                                 MultipleLines ,
                                                 InternetService ,
                                                 OnlineSecurity ,
                                                 OnlineBackup ,
                                                 DeviceProtection ,
                                                 TechSupport ,
                                                 StreamingTV ,
                                                 StreamingMovies))



set.seed(100)
# training data contains 70% of the data
#testing data contains 30% of the data
#split_data=sample(1:nrow(churn_dummy),0.7*nrow(churn_dummy))
#data_train=churn_dummy[split_data,]
#data_test=churn_dummy[-split_data,]


split_data <-  sample.split(churn_dummy$Churn, SplitRatio = 0.7)
table(split_data)
data_train <-  churn_dummy[split_data,]
data_test <- churn_dummy[!(split_data),]

remove(split_data)

#--Perform outlier treatment if necessary
boxplot(churn$tenure)$out
boxplot(churn$MonthlyCharges)$out
boxplot(churn$TotalCharges)$out

boxplot(churn$tenure)
boxplot(churn$MonthlyCharges)
boxplot(churn$TotalCharges)
#quantile(churn$TotalCharges,probs = c(0.9,0.91,0.95,0.975, 0.98, 0.985, 0.99 , 1    ))
#####################################################################
#-- Checkpoint 4: Model Building : Model - K-NN
#####################################################################

#-- Model - K-NN:

#Using the train() command to find the best K.
timestamp()
KNNmodel <- train(
  Churn~., 
  data=data_train,
  method='knn',
  tuneGrid=expand.grid(.k=seq(11,121,6)),
  metric='Accuracy',
  trControl=trainControl(method='repeatedcv', number=10,repeats=11)  
  )
timestamp()
plot(KNNmodel, main=" Optimal K")


# True class labels of training data
cl <- data_train[, c("Churn")]

#Optimal  k = 49 better sensitivity
kval=49
#KNN -  Nearest Neighbours Training set
knnTestResult <- knn(subset(data_train, select=-c(Churn)), subset(data_test, select=-c(Churn)), cl=cl, k = kval, prob = T)

table(knnTestResult, data_test[, c("Churn")] )
confusionMatrix(knnTestResult, data_test[, c("Churn")], positive = "Yes")

knn_prob <- attr(knnTestResult, "prob")
knn_prob <- 49*ifelse(knnTestResult == "No", 1-knn_prob, knn_prob) - 48

#calculating the values for ROC curve
pred_knn <- prediction(knn_prob, data_test$Churn)
perf_knn <- performance(pred_knn, measure = "tpr", x.measure = "fpr")
# plotting the ROC curve
plot(perf_knn, col="black",lty=3, lwd=3, main="KNN performance")
# calculating AUC
knnauc <- performance(pred_knn, "auc")


######################################################################

#####################################################################
#-- Checkpoint 4: Model Building : Model - Logistic Regression:
#####################################################################
# Model with all variables
data_train_reg <- data_train
data_test_reg <- data_test

initial_model <-  glm(Churn ~ ., data = data_train_reg, family = "binomial")
summary(initial_model)
#-- Null deviance: 5678.9  on 4914  degrees of freedom
#-- Residual deviance: 4099.2  on 4891  degrees of freedom
#-- AIC: 4147.2

#step <- stepAIC(initial_model, direction="both")
step <- step(initial_model, direction="both")
step
print(step$call)

# Stepwise selection of variables

#making model 2
model_2 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                DependentsYes + PhoneServiceYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo + OnlineBackupYes + 
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = binomial(), data = data_train_reg)
summary(model_2)
vif(model_2)

#VIF is more than 2, removing OnlineBackupYes as the pvalue is high 0.079501

#making model 3
model_3 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                DependentsYes + PhoneServiceYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo  +    
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = binomial(), data = data_train_reg)
summary(model_3)
vif(model_3)

#VIF is more than 2, removing PhoneServiceYes as the pvalue is high 0.128765
model_4 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                DependentsYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo  +    
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = binomial(), data = data_train_reg)
summary(model_4)
vif(model_4)

#VIF is more than 2, removing DeviceProtectionYes as the pvalue is high 0.135756
model_5 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                DependentsYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo  +    
                StreamingTVYes + StreamingMoviesYes, 
              family = binomial(), data = data_train_reg)
summary(model_5)
vif(model_5)

# Model 5 - Residual deviance: 4109.1  on 4901  degrees of freedom
#VIF is high , removing DependentsYes as the pvalue is high 0.013326

final_model <- model_5
summary(final_model)
str(final_model)
#-- 

#
# Model Evaluation
# 

str(data_test)
str(data_train)

logreg_test_predicted <- predict(final_model, newdata = data_test_reg , type = "response")
logreg_test_prediction <- prediction(logreg_test_predicted, data_test_reg$Churn)
logreg_performance_measures <- performance(logreg_test_prediction, measure = "tpr", x.measure = "fpr")
plot(logreg_performance_measures, main="Logistic Regression ROC")
auc <- performance(test_prediction, measure = "auc")
#auc value 0.8512656


library(Hmisc)
#C Index for test --- 8.512656e-01
rcorr.cens(logreg_test_predicted, data_test_reg$Churn)

#KS statistics
logreg_ks_table <- attr(logreg_performance_measures,"y.values")[[1]] - attr(logreg_performance_measures,"x.values")[[1]]
ks_stats <- max(logreg_ks_table) 
ks_stats  #0.5433482


opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    # Find the minimum distance from the top left of elbow [(x=0,y=1)]
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

print(opt.cut(logreg_performance_measures, logreg_test_prediction))
#-- sensitivity 0.7809695
#-- specificity 0.7553260
#-- cutoff      0.3040141

# Checkpoint 6: Threshold value
#model_final Confusion Matrix
#Thresold value is Keep the probability threshold as 0.3, 0.5 and 0.7 respectively.

confusionMatrix(as.factor(as.numeric(logreg_test_predicted > 0.3)), as.factor(as.numeric(data_test_reg$Churn == "Yes")), positive = "1") 
confusionMatrix(as.factor(as.numeric(logreg_test_predicted > 0.5)), as.factor(as.numeric(data_test_reg$Churn == "Yes")), positive = "1") 
confusionMatrix(as.factor(as.numeric(logreg_test_predicted > 0.7)), as.factor(as.numeric(data_test_reg$Churn == "Yes")), positive = "1") 

#####################################################################
#-- Checkpoint 4: Model Building : Model - SVM:
#####################################################################

svm_tune <- tune.svm(Churn~., data = data_train, cost = 2^(1:8),  kernel = "linear") 
#-- sampling method: 10-fold cross validation 
#-- best parameters:
#-- cost 2

svm_model <- svm(Churn ~. , data_train, cost = 2, scale = F, kernel = "linear", probability = T)
#summary and details of model
summary(svm_model)

#fitted results
svm_fitted.results <- predict(svm_model, newdata = data_test, probability = T)

#error
svm_misClasificError <- mean(svm_fitted.results != data_test$Churn)

#Accuracy of model
print(paste('Accuracy',1-misClasificError))

table(svm_fitted.results, data_test$Churn)
confusionMatrix(svm_fitted.results, data_test$Churn, positive = "Yes")

plot(svm_model, data_test, main= " SVM ROC" )

auc <- performance(pred_naive,"auc")
auc #0.7243041


#####################################################################
#-- Checkpoint 4: Model Building : Model - Naive Bayes:
#####################################################################

churn_AllFactors <- subset(churn, select = -c(customerID))
churn_AllFactors$TotalCharges <- cut(churn_AllFactors$TotalCharges,10,labels = c(1:10)) 
churn_AllFactors$MonthlyCharges <- cut(churn_AllFactors$MonthlyCharges,10,labels = c(1:10)) 
churn_AllFactors$tenure <- as.factor(churn_AllFactors$tenure)

str(churn_AllFactors)

split_data <-  sample.split(churn_AllFactors$Churn, SplitRatio = 0.7)
table(split_data)
data_train_factors <-  churn_dummy[split_data,]
data_test_factors <- churn_dummy[!(split_data),]
remove(split_data)

model_naive <- naiveBayes(Churn ~. , data = data_train_factors )
#Naivepred <- predict(model_naive,  subset(data_test_factors, select=-c(Churn))  )
Naivepred <- predict(model_naive,  subset(data_test_factors, select=-c(Churn)) , type = "raw" )
NaiveProb  <- Naivepred[,c("Yes")]


table(Naivepred, data_test_factors$Churn)
confusionMatrix(Naivepred, data_test_factors$Churn, positive = "Yes")

Naivepred1 <- as.numeric(Naivepred)

library(ROCR)
#calculating the values for ROC curve
pred_naive <- prediction(Naivepred, data_test_factors$Churn )
Naiveperf <- performance(pred_naive,"tpr","fpr" )

# plotting the ROC curve
plot(Naiveperf,col="black",lty=3, lwd=3, main = "Naive Bayes ROC" )

# calculating AUC
auc <- performance(pred_naive,"auc")
auc #0.7303841

