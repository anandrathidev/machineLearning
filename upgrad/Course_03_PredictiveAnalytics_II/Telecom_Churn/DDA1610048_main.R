#####################################################################################################
# PGDDA Course-3 
# Course 3 Assignment 1: Telecom Churn
#----------------------------------------------------------------------------------------------------
#
# The goal of this assignment:
# ==================================================================================================
# The company wants to understand the driving factors behind churn and wants to build a model 
# which would predict future churn.  The company can utilise this knowledge for churn prevention. 
# Specifically, the company wants to determine which driver variables are having the most influence 
# on the tendency of churning.
#
# You are required to develop predictive models using each of the 4 models namely
#   - K-NN
#   - Naive Bayes
#   - Logistic Regression 
#   - SVM.
#####################################################################################################
#
#--Setup environment-----------------------------------------------------------
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
library(ROCR)
library(Hmisc)
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(class)

###############################################################################
#--Checkpoint-1: Business Understanding and Preparation of Master File.
###############################################################################
# Customer churn can depend on a lot of internal and external factors. 
# External factors are the ones you might have no control or information about, 
#  for example, the launch of Reliance Jio will lead to churn across all 
#  telecom companies and predicting the churn for such cases gets extremely 
#  difficult. However, 
# Internal factors such as the demographic information, the number of connections 
#  taken by a customer, personal information, billing information, information 
#  on services availed etc. can be used to predict the churn of customers. 
###############################################################################
#--Checkpoint-2: Data Understanding and Preparation of Master File.
###############################################################################
#--Load the 3 files given in separate data frames.
#--Collate the data together in one single file.
cust_data <- read.csv("customer_data.csv")
churn_data <- read.csv("churn_data.csv") 
internet_data <- read.csv("internet_data.csv")

#--Initial Data Exploration----------------------------------------------------
#--Customer Data
str(cust_data)
summary(cust_data)
#-->The customer data has 7043 observations(rows) and 5 variables(columns) 
#-->Customer ID is the primary/unique Key 
#-->The rest are all boolean factors except Senior Citizen which clearly
#   should also be boolean as Yes/No. LEt us fix thsicolumn
cust_data$customerID <- as.character(cust_data$customerID)
cust_data$SeniorCitizen <- factor(ifelse(cust_data$SeniorCitizen == 1, "Yes", "No"))
#-->let us now again analyse our data
str(cust_data)
summary(cust_data)

#--Churn Data
str(churn_data)
summary(churn_data)
#-->The curn data has 7043 observations(rows) and 9 variables(columns)
#-->Customer ID is the primary/unique key
#-->All columns seem apropriate and we observe TotalCharges has 11 NA's
#   We have NA values but let us retain them for now untill we merge the data
churn_data$customerID <- as.character(churn_data$customerID)
#churn_data <- churn_data[!is.na(churn_data$TotalCharges),]
#--let us have alook again at the data
str(churn_data)
summary(churn_data)

#--Internet Data
str(internet_data)
summary(internet_data)
#-->The internet data has 7043 observations(rows) and 9 variables(columns)
#-->Customer ID is the primary/unique key
#-->All columns seem apropriate and we observe no NA's
internet_data$customerID <- as.character(internet_data$customerID)
#--let us have alook again at the data
str(internet_data)
summary(internet_data)

#--Merging the data to form a single dataset-----------------------------------
#--We create a generic function to make this mor efficient
dfMerge <- function(x, y) {
  df <- merge(x, y, by= "customerID", all.x= TRUE, all.y= TRUE)
  return(df)
}
churn <- Reduce(dfMerge, list(cust_data, churn_data, internet_data))
remove(cust_data, churn_data, internet_data)

#--Let us have an initial look at the new churn data
str(churn)
summary(churn)
#-->We have 7043 observations(rows) and 21 variables(columns)
#-->We have Customer ID as the primary key
#-->We have 11 NA values for Total Cahrges as previously observed
###############################################################################
#-- Checkpoint -4: Exploratory Data Analysis
###############################################################################
#-- Make bar charts displaying the relationship between the 
#-- target variable and various other features and report them.
factor_plots_1 <- ggplot(churn, aes(x=Churn,  group = gender)) + 
 geom_bar(aes(y = ..count.., fill = factor(gender)), stat="count")

factor_plots_2 <- ggplot(churn, aes(x=Churn,  group = SeniorCitizen)) + 
  geom_bar(aes(y = ..count.., fill = factor(SeniorCitizen)), stat="count")  

factor_plots_3 <- ggplot(churn, aes(x=Churn,  group = Partner)) + 
  geom_bar(aes(y = ..count.., fill = factor(Partner)), stat="count")  

factor_plots_4 <- ggplot(churn, aes(x=Churn,  group = Dependents)) + 
  geom_bar(aes(y = ..count.., fill = factor(Dependents)), stat="count") 

factor_plots_5 <- ggplot(churn, aes(x=Churn,  group = gender)) + 
  geom_bar(aes(y = ..count.., fill = factor(gender)), stat="count") 

factor_plots_6 <- ggplot(churn, aes(x=Churn,  group = PhoneService)) + 
  geom_bar(aes(y = ..count.., fill = factor(PhoneService)), stat="count") 

factor_plots_7 <- ggplot(churn, aes(x=Churn,  group = Contract)) + 
  geom_bar(aes(y = ..count.., fill = factor(Contract)), stat="count") 

factor_plots_8 <- ggplot(churn, aes(x=Churn,  group = PaperlessBilling)) + 
  geom_bar(aes(y = ..count.., fill = factor(PaperlessBilling)), stat="count") 

factor_plots_9 <- ggplot(churn, aes(x=Churn,  group = PaymentMethod)) + 
  geom_bar(aes(y = ..count.., fill = factor(PaymentMethod)), stat="count") 

factor_plots_10 <- ggplot(churn, aes(x=Churn,  group = InternetService)) + 
  geom_bar(aes(y = ..count.., fill = factor(InternetService)), stat="count") 

factor_plots_11 <- ggplot(churn, aes(x=Churn,  group = OnlineSecurity)) + 
  geom_bar(aes(y = ..count.., fill = factor(OnlineSecurity)), stat="count") 

factor_plots_12 <- ggplot(churn, aes(x=Churn,  group = OnlineBackup)) + 
  geom_bar(aes(y = ..count.., fill = factor(OnlineBackup)), stat="count")

factor_plots_13 <- ggplot(churn, aes(x=Churn,  group = DeviceProtection)) + 
  geom_bar(aes(y = ..count.., fill = factor(DeviceProtection)), stat="count") 

factor_plots_14 <- ggplot(churn, aes(x=Churn,  group = TechSupport)) + 
  geom_bar(aes(y = ..count.., fill = factor(TechSupport)), stat="count")

factor_plots_15 <- ggplot(churn, aes(x=Churn,  group = StreamingTV)) + 
  geom_bar(aes(y = ..count.., fill = factor(StreamingTV)), stat="count") 

factor_plots_16 <- ggplot(churn, aes(x=Churn,  group = StreamingMovies)) + 
  geom_bar(aes(y = ..count.., fill = factor(StreamingMovies)), stat="count") 

# 
# # is the following working?
 grid.arrange(factor_plots_1,
              factor_plots_2,
              factor_plots_3,
              factor_plots_4,
              ncol = 2, top = "Univariate Analysis Factors")

grid.arrange(factor_plots_5,
              factor_plots_6,
              factor_plots_7,
              factor_plots_8,
              ncol = 2, top = "Univariate Analysis Factors ")

grid.arrange(factor_plots_9,
             factor_plots_10,
             factor_plots_11,
             factor_plots_12,
             ncol = 2, top = "Univariate Analysis Factors ")

grid.arrange(factor_plots_13,
             factor_plots_14,
             factor_plots_15,
             factor_plots_16,
             ncol = 2, top = "Univariate Analysis Factors ")

 
#--Tenure
histogram(~churn$tenure | factor(churn$Churn), data = churn,
          xlab = "Tenure", ylab = "Count")
#--MonthlyCharges
histogram(~churn$MonthlyCharges | factor(churn$Churn), data = churn,
          xlab = "Monthly Charges", ylab = "Count")
#--Tenure
histogram(~churn$TotalCharges | factor(churn$Churn), data = churn,
          xlab = "Total Charges", ylab = "Count")


###############################################################################
#-- Checkpoint -5: Data Preparation
###############################################################################
#--Duplicate Data--------------------------------------------------------------
#--Check for duplicate data removing the customer id
#-->There are 22 duplicated rows. Let us get rid of them
sum(duplicated(churn[,-1]))
churn <- churn[!duplicated(churn[,-1]), ]
#--Let us have a look at our data again
str(churn)
summary(churn)
#-->We now have 7021 observations(rows) and 21 variables(cloumns)

#--Missing Values-------------------------------------------------------------
#--We also still retain the NA values for Total Charge. Let us get rid of them
churn <- churn[!is.na(churn$TotalCharges),]
#--Let us have a look at our data again
str(churn)
summary(churn)
#-->We now have 7010 observations(rows) and 21 variables(cloumns)
#--No NA values

#--Outlier Treatment-----------------------------------------------------------
#--Let us perform outlier treatment on teh dataset now
#--We will use 2 methods to check for outliers - boxplot,stats and quantile
boxplot(churn$tenure)
boxplot.stats(churn$tenure)
quantile(churn$tenure, seq(0,1,0.01))

boxplot(churn$MonthlyCharges)
boxplot.stats(churn$MonthlyCharges)
quantile(churn$MonthlyCharges, seq(0,1,0.01))

boxplot(churn$TotalCharges)
boxplot.stats(churn$TotalCharges)
quantile(churn$TotalCharges, seq(0,1,0.01))

###############################################################################
#-- Checkpoint 6: Model Building 
###############################################################################
#--Preperation of Data---------------------------------------------------------
#--Let us have a look at our data again before we begin variable preperation
str(churn)
summary(churn)
#
#--Feature Preperation---------------------------------------------------------
#--Convert all categorical variables to numerical/dummy form
#--Making use of model.matrix function
churn_dummy_gender  <- data.frame(model.matrix( ~gender ,data=churn))
churn_dummy_SeniorCitizen  <- data.frame(model.matrix( ~SeniorCitizen ,data=churn))
churn_dummy_Partner  <- data.frame(model.matrix( ~Partner ,data=churn))
churn_dummy_Dependents  <- data.frame(model.matrix( ~Dependents ,data=churn))
churn_dummy_PhoneService  <- data.frame(model.matrix( ~PhoneService ,data=churn))
churn_dummy_Contract  <- data.frame(model.matrix( ~Contract ,data=churn))
churn_dummy_PaperlessBilling  <- data.frame(model.matrix( ~PaperlessBilling ,data=churn))
churn_dummy_PaymentMethod   <- data.frame(model.matrix( ~PaymentMethod ,data=churn))
churn_dummy_MultipleLines  <- data.frame(model.matrix( ~MultipleLines ,data=churn))
churn_dummy_InternetService  <- data.frame(model.matrix( ~InternetService ,data=churn)) 
churn_dummy_OnlineSecurity  <- data.frame(model.matrix( ~OnlineSecurity ,data=churn))
churn_dummy_OnlineBackup  <- data.frame(model.matrix( ~OnlineBackup ,data=churn)) 
churn_dummy_DeviceProtection  <- data.frame(model.matrix( ~DeviceProtection ,data=churn))
churn_dummy_TechSupport  <- data.frame(model.matrix( ~TechSupport ,data=churn))
churn_dummy_StreamingTV  <- data.frame(model.matrix( ~StreamingTV ,data=churn))    
churn_dummy_StreamingMovies  <- data.frame(model.matrix( ~StreamingMovies ,data=churn))

#--Let us create a new dataset to work with for KNN predictive analysis
churn_model_data <- churn
#--Let us bind all our dummy variables to create a new dataset
churn_model_data <- cbind(churn_model_data,
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
                          churn_dummy_StreamingMovies)

#--Scaling the data
churn_model_data$tenure <- scale(churn_model_data$tenure)
churn_model_data$MonthlyCharges <- scale(churn_model_data$MonthlyCharges)
churn_model_data$TotalCharges <- scale(churn_model_data$TotalCharges)

#--Removing the dummy variables from memory
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
#--removing the duplicated data variables 
churn_model_data <- subset(x=churn_model_data, select = -c(customerID,gender,SeniorCitizen,
                                                           Partner ,Dependents,PhoneService,
                                                           Contract,PaperlessBilling,PaymentMethod,
                                                           MultipleLines,InternetService,OnlineSecurity,
                                                           OnlineBackup,DeviceProtection,TechSupport,
                                                           StreamingTV,StreamingMovies))


#--Let us look at our dataset after transformation
str(churn_model_data)
summary(churn_model_data)

#--Preparing model data -------------------------------------------------------
#--Let us split data into Training and Test datasets
set.seed(100)
split_data <-  sample.split(churn_model_data$Churn, SplitRatio = 0.7)
table(split_data)
data_train <-  churn_model_data[split_data,]
data_test <- churn_model_data[!(split_data),]
#--removing the index data
remove(split_data)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#--K-NN Model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
#--Using the train() command to find the best K--------------------------------
timestamp()
KNNmodel <- train(
  Churn~., 
  data=data_train,
  method='knn',
  tuneGrid=expand.grid(.k=seq(1,55,3)),
  metric='Accuracy',
  trControl=trainControl(method='repeatedcv', number=10,repeats=11)  
)
timestamp()
plot(KNNmodel)

#--Observing the model---------------------------------------------------------
KNNmodel
kval <- KNNmodel$bestTune[[1]]
#..............................................................................
#-->Accuracy was used to select the optimal model using  the largest value.
#   The final value used for the model was k = 52
#..............................................................................
#--Getting the True class labels of training data
cl <- data_train[, c("Churn")]

#--KNN-Nearest Neighbours Training set
knnTestResult <- knn(subset(data_train, select=-c(Churn)), 
                     subset(data_test, select=-c(Churn)), cl=cl, k = kval, prob = T)

#--Evaluating the model--------------------------------------------------------
table(knnTestResult, data_test[, c("Churn")] )
confusionMatrix(knnTestResult, data_test[, c("Churn")], positive = "Yes")

#--
knn_prob <- attr(knnTestResult, "prob")
knn_prob <- kval*ifelse(knnTestResult == "No", 1-knn_prob, knn_prob) - (kval-1)

#--calculating the values for ROC curve
pred_knn <- prediction(knn_prob, data_test$Churn)
perf_knn <- performance(pred_knn, measure = "tpr", x.measure = "fpr")

#--plotting the ROC curve
plot(perf_knn, col="black",lty=3, lwd=3, main="KNN performance")

#--calculating AUC
knnauc <- performance(pred_knn, "auc")

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-- Checkpoint 4: Model Building : Model - Naive Bayes:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
timestamp()
#
#--Taking the original dataset as we do not require dummy creation for Bayesian analysis
#--Removing the CustomerID
churn_AllFactors <- subset(churn, select = -c(customerID))
#--Binning the contnious data
churn_AllFactors$TotalCharges <- cut(churn_AllFactors$TotalCharges,10,labels = c(1:10)) 
churn_AllFactors$MonthlyCharges <- cut(churn_AllFactors$MonthlyCharges,10,labels = c(1:10)) 
churn_AllFactors$tenure <- as.factor(churn_AllFactors$tenure)

#--LEt us observe the dataset
str(churn_AllFactors)

#--Splitting the data into training and testing sets---------------------------
split_data <-  sample.split(churn_AllFactors$Churn, SplitRatio = 0.7)
table(split_data)
data_train_factors <-  churn_AllFactors[split_data,]
data_test_factors <- churn_AllFactors[!(split_data),]
remove(split_data)

#--Creating the model----------------------------------------------------------
naiveBayesModel <- naiveBayes(Churn~.,
                              data = data_train_factors)
naiveBayesPred <- predict(naiveBayesModel, subset(data_test_factors, select=-c(Churn)))

confusionMatrix(naiveBayesPred, data_test_factors$Churn)

#--Plotting an ROC curve-------------------------------------------------------
naiveBayesRawPred <- predict(naiveBayesModel, subset(data_test_factors, 
                                                     select=-c(Churn)), type = 'raw')
naiveBayesPredProb <- naiveBayesRawPred[,1]

churnRealVec <- ifelse(data_test_factors$Churn == "No", 1, 0)
naiveBayesPr <- prediction(naiveBayesPredProb, churnRealVec)
naiveBayesPrf <- performance(naiveBayesPr, "tpr", "fpr")

#--ROC curve
plot(naiveBayesPrf,col="black",lty=3, lwd=3, main = "Naive Bayes ROC" )

#--calculating AUC-------------------------------------------------------------
auc <- performance(naiveBayesPr,"auc")
auc #0.8275361

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-- Checkpoint 4: Model Building : Model - Logistic Regression:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#--Model with all variables
data_train_reg <- data_train
data_test_reg <- data_test

#--Initial model
initial_model <-  glm(Churn ~ ., data = data_train_reg, family = "binomial")
summary(initial_model)
#..............................................................................
#-->Null deviance: 5673.9  on 4906  degrees of freedom
#-->Residual deviance: 4057.5  on 4883  degrees of freedom
#-->AIC: 4105.5
#..............................................................................
#--USing stepwise AIC to select the best model
step <- step(initial_model, direction="both")
step
print(step$call)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4061.0  on 4890  degrees of freedom
# AIC: 4095
#--
#--Manual Stepwise selection of variables
#--Model 2
#--removing SniorCitzenYes as pvalue is 0.013055
model_2 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                PhoneServiceYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo + OnlineBackupYes + 
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_2)
vif(model_2)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4067.1  on 4891  degrees of freedom
# AIC: 4099.1
#--
#--Model 3
#--Removing OnlineBackupYes as the pvalue is high 0.006619
model_3 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                PhoneServiceYes + ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo +  
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_3)
vif(model_3)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4074.5  on 4892  degrees of freedom
# AIC: 4104.5
#--
#--Model 4
#--Removing OnlineBackupYes as the pvalue is high 0.006619
model_4 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo +  
                DeviceProtectionYes + StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_4)
vif(model_4)
#-->We also see a marked drop in the high VIF values such as tnure, monthlycharges etc.
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4081.3  on 4893  degrees of freedom
# AIC: 4109.3
#--
#--Model 5
#--Removing OnlineBackupYes as the pvalue is high 0.013810
model_5 <-glm(formula = Churn ~ tenure + MonthlyCharges + TotalCharges + 
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo +  
                StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_5)
vif(model_5)
#-->We do not see any drop in VIF values also all the remaining variables 
#   appear as significant. But many of the values have high VIF and seem to be slightly co-related
#   for example Monthly and Total charges are clearly related from logic and the same goes for tenure
#   so we take the higher p-value and remove it.
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4087.4  on 4894  degrees of freedom
# AIC: 4113.4
#--
#--Model 6
#--Removing TotalCjharges as it has higher pvalue is 0.000272
model_6 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceFiber.optic + InternetServiceNo +  
                StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_6)
vif(model_6)
#-->As per our expectations this has reduced the tenure but from the decrase it is 
#   clearly more co-related than Monthly charges
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4101.2  on 4895  degrees of freedom
# AIC: 4125.2
#--
#--Model 7
#--Retaining MonthlyCahrges as it is an important factor from a business point
model_7 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceNo +  
                StreamingTVYes + StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_7)
vif(model_7)
#-->This has reduced the Monthly Charges also which is understandable as such features
#   generally incur more cost. This though has related is some insignificant variables.
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4183.5  on 4896  degrees of freedom
# AIC: 4205.5
#--
#--Model 8
#--Removing StreamingTVYes with high PValue 0.11223
model_8 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceNo +  
                StreamingMoviesYes, 
              family = "binomial", data = data_train_reg)
summary(model_8)
vif(model_8)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4186.1  on 4897  degrees of freedom
# AIC: 4206.1
#--
#--Model 9
#--Removing StreamingMoviesYes with high PValue 0.01472
model_9 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes + 
                InternetServiceNo, 
              family = "binomial", data = data_train_reg)
summary(model_9)
vif(model_9)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4192.0  on 4898  degrees of freedom
# AIC: 4210
#--
#--Model 10
#--Removing InternetServiceNo with high PValue 0.0142
model_10 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                ContractOne.year + ContractTwo.year + 
                PaperlessBillingYes + PaymentMethodElectronic.check + MultipleLinesYes, 
              family = "binomial", data = data_train_reg)
summary(model_10)
vif(model_10)
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4198.1  on 4899  degrees of freedom
# AIC: 4214.1
#--
#--Model 11
#--Removing MultipleLinesYes with high PValue 0.0329
model_11 <-glm(formula = Churn ~ tenure + MonthlyCharges +  
                 ContractOne.year + ContractTwo.year + 
                 PaperlessBillingYes + PaymentMethodElectronic.check, 
               family = "binomial", data = data_train_reg)
summary(model_11)
vif(model_11)
#-->With this model we have all VIF values below threshold of 2 and
#   all insignificant variable removed
# Null deviance: 5673.9  on 4906  degrees of freedom
# Residual deviance: 4202.7  on 4900  degrees of freedom
# AIC: 4216.7

#--Taking Model 11 as the finalised model
final_model <- model_11
summary(final_model)
str(final_model)
#-- 
#--Model Evaluation-------------------------------------------------------------
#
#--lets have a look at the training/test dataset
# str(data_test)
# str(data_train)
#
#--ROC and AUC values----------------------------------------------------------
#--Train
#--Ploting an ROC curve and determining the AUC value
logreg_train_predicted <- predict(final_model, newdata = data_train_reg , type = "response")
logreg_train_prediction <- prediction(logreg_train_predicted, data_train_reg$Churn)
logreg_performance_measures_train <- performance(logreg_train_prediction, measure = "tpr", x.measure = "fpr")
plot(logreg_performance_measures_train, main="Logistic Regression ROC")
auc_train <- performance(logreg_train_prediction, measure = "auc")
#-->The AUC value 0.8338867
#   We have a clean ROC graph and the area under the curve is at 83% which is a good value

#--Test
#--Ploting an ROC curve and determining the AUC value
logreg_test_predicted <- predict(final_model, newdata = data_test_reg , type = "response")
logreg_test_prediction <- prediction(logreg_test_predicted, data_test_reg$Churn)
logreg_performance_measures <- performance(logreg_test_prediction, measure = "tpr", x.measure = "fpr")
plot(logreg_performance_measures, main="Logistic Regression ROC")
auc <- performance(logreg_test_prediction, measure = "auc")
#-->The AUC value 0.8323861
#   We have a clean ROC graph and the area under the curve is at 83% which is a good value

#--Calculating C-Index values--------------------------------------------------
#--Train data
rcorr.cens(logreg_train_predicted, data_train_reg$Churn)
#-->C Index for Train data --- 8.338867e-01

#--Test data
rcorr.cens(logreg_test_predicted, data_test_reg$Churn)
#--> C Index for test data --- 8.323861e-01

#-->The values of C-Index are very close to each other and have a good value

#--KS statistics---------------------------------------------------------------
#--Train Data
logreg_ks_table_train <- attr(logreg_performance_measures_train,"y.values")[[1]] - attr(logreg_performance_measures_train,"x.values")[[1]]
ks_stats_train <- max(logreg_ks_table_train) 
ks_stats_train  
#--> KS Stats value 0.5165529

#--Test Data
logreg_ks_table <- attr(logreg_performance_measures,"y.values")[[1]] - attr(logreg_performance_measures,"x.values")[[1]]
ks_stats <- max(logreg_ks_table) 
ks_stats  
#--> KS Stats value 0.5324693

#-->The values of KS-Stats are very close to each other and have a good value

#--Optimal value---------------------------------------------------------------
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    # Find the minimum distance from the top left of elbow [(x=0,y=1)]
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}

print(opt.cut(logreg_performance_measures, logreg_test_prediction))
#-->sensitivity 0.8168761
#-->specificity 0.7095731
#-->cutoff      0.2808579

#--Threshold value-------------------------------------------------------------
#--Taking 0.5 as the cost values as we want an equal emphasis on the Positve and Negative
confusionMatrix(as.factor(as.numeric(logreg_test_predicted > 0.5)), 
                as.factor(as.numeric(data_test_reg$Churn == "Yes")), positive = "1")
#..............................................................................
#      Accuracy : 0.7908 
#   Sensitivity : 0.5045          
#   Specificity : 0.8939
#..............................................................................

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-- Checkpoint 6: Model Building : Model - SVM:
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#-Creating SVM models with various 
#
timestamp()
tune.svm <- tune(svm, Churn~., data = data_train, kernel = "linear", 
                 ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 100)))
timestamp()
#
summary(tune.svm)
#..............................................................................
# -- best parameters: cost 0.1
# - best performance: 0.1999152 
# 
# - Detailed performance results:
#   cost     error dispersion
# 1 1e-03 0.2649237 0.02589409
# 2 1e-02 0.2001189 0.01975207
# 3 1e-01 0.1999152 0.01963013
# 4 5e-01 0.2001189 0.01940211
# 5 1e+00 0.2001185 0.01915922
# 6 1e+01 0.2009331 0.01954427
# 7 1e+02 0.2011368 0.01886735
#..............................................................................
#--> The best performance is for model with cost 0.1
churn.svm.model <- tune.svm$best.model
summary(churn.svm.model)
#..............................................................................
# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  linear 
#       cost:  0.1 
#      gamma:  0.02173913 
# 
# Number of Support Vectors:  2271 ( 1139 1132 )
# Number of Classes:  2 
# Levels: No Yes
#..............................................................................

svmpred <- predict(churn.svm.model, data_test)
table(predicted = svmpred, truth = data_test$Churn)
#..............................................................................
#             truth
# predicted   No   Yes
#        No  1383  260
#       Yes  163   297
#..............................................................................
confusionMatrix(svmpred, data_test$Churn, positive = "Yes")
#..............................................................................
# Confusion Matrix and Statistics
# -------------------------------
#             Reference
# predicted   No   Yes
#        No  1383  260
#       Yes  163   297
# 
#            Accuracy : 0.7989          
#              95% CI : (0.7772, 0.8121)
# 
#          Sensitivity : 0.8946          
#          Specificity : 0.5332
# 
# 'Positive' Class : No
#..............................................................................

###############################################################################
#-- END
###############################################################################