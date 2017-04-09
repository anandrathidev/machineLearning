
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
library(ggplot2)
library(grid)
library(gridExtra)

#######################################################################
#Checkpoint 1: Data Understanding and Data Exploration
#######################################################################

#--TASK: You have to just predict whether the customer will default or not.
#setwd(dir = "C:\\Users\\rb117\\Documents\\Personal\\Upgrad\\LogisticRegression\\")
setwd(dir = "C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\Supervised_Classification_II\\Assignment\\")
german_credit <- read.csv("german.csv")

#--Understand Data
summary(german_credit)

sum(is.null(german_credit))==0

#--(Certain variables which should be ideally of factor type might be of 
#-- character or integer type. 
#-- you need to first convert such variables into factor variables using "as.factor()")
#--Check if any thing needs to be in factors
str(german_credit)


#######################################################################
# Checkpoint 2: Data Cleaning and Transformation
#######################################################################
# Data prpeapartion and feature transformation

#-- In this step, you first need to identify the missing values
sum(is.null(german_credit))==0

#-- And outliers for each variable and impute them accordingly. OR REMOVE
#-- I belive in removing single entry of month  == 72 
#-- reason is I am not sure how it will impact co-related varianbles if imputed
#-- I am not touching other outliers like > 48 because they are in large numbers 
# -- and any imputation might change the business logic as they mey affect co-related varianbles

#-- Any outliers are going to decrease model performance
outlier_plots_1 <- ggplot(german_credit, aes(x = "Duration.in.month", y = Duration.in.month )) +  geom_boxplot()
quantile(german_credit$Duration.in.month, c(0.95,0.96,0.97,0.98,0.99,1))

#-- Ger rid of that outlier
#german_credit <- subset(german_credit, Duration.in.month< 72 )

#-- does Duration.in.month looks like a factor? No , its atleast interval if not ratio 
#-- I keep it as number
german_credit <- subset(german_credit, ! Duration.in.month %in% boxplot(german_credit$Duration.in.month)$out  )


quantile(german_credit$Credit.amount, c(0.95,0.96,0.97,0.98,0.99,1))
#boxplot(german_credit$Credit.amount)
outlier_plots_2 <- ggplot(german_credit, aes(x = "Credit.amount", y = Credit.amount )) +  geom_boxplot()
german_credit <- subset(german_credit, ! Credit.amount %in% boxplot(german_credit$Credit.amount)$out  )

#-- does Present.residence.since looks like a fctor? No 
#-- I keep it as number
quantile(german_credit$Installment.rate.in.percentage.of.disposable.income, c(0.95,0.96,0.97,0.98,0.99,1))
outlier_plots_3 <- ggplot(german_credit, aes(x = "Installment.rate.in.percentage.of.disposable.income", y = Installment.rate.in.percentage.of.disposable.income )) +  geom_boxplot()

#-- does Present.residence.since looks like a fctor? No ,  since its more looks like atleast ordinal if not interval 
#-- I keep it as number
quantile(german_credit$Present.residence.since, c(0.95,0.96,0.97,0.98,0.99,1))
outlier_plots_4 <- ggplot(german_credit, aes(x = "Present.residence.since", y = Present.residence.since )) +  geom_boxplot()

#-- does Age looks like a fctor? No ,  since its more looks like atleast ordinal if not interval 
#--I keep it as number
quantile(german_credit$Age.in.Years, c(0.95,0.96,0.97,0.98,0.99,1))
outlier_plots_4 <- ggplot(german_credit, aes(x = "Age.in.Years", y = Age.in.Years )) +  geom_boxplot()

quantile(german_credit$Number.of.existing.credits.at.this.bank., c(0.95,0.96,0.97,0.98,0.99,1))
boxplot(german_credit$Number.of.existing.credits.at.this.bank.)
outlier_plots_5 <- ggplot(german_credit, aes(x = "Number.of.existing.credits.at.this.bank.", y = Number.of.existing.credits.at.this.bank. )) +  geom_boxplot()
german_credit <- subset(german_credit, ! Number.of.existing.credits.at.this.bank. %in% boxplot(german_credit$Number.of.existing.credits.at.this.bank.)$out  )

grid.arrange(outlier_plots_1,
             outlier_plots_2 ,
             outlier_plots_3,
             outlier_plots_4 ,
             outlier_plots_5,
             ncol = 2, top = "Box Plots")

#--german_credit$Number.of.people.being.liable.to.provide.maintenance.for. 
#-- does it looks like a fctor? No ,  since its more looks like atleast ordinal if not interval 
#--I keep it as number



#------------------------------------------------------------------------
# Exploratory Data Analysis
#------------------------------------------------------------------------

#--Explore Default Vs Credit.amount
#--create bins
bins<-10
cutpoints<-quantile(german_credit$Credit.amount,(0:bins)/bins)
german_credit$binned_Credit.amount  <- cut(german_credit$Credit.amount, cutpoints, include.lowest=TRUE, labels = paste0(seq(10,100,10), '% ' ))
#--Plot Continues Variables
library(scales) #-- percentage
univariate_plots_0 <- ggplot(data=german_credit, aes(x=binned_Credit.amount,  fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
univariate_plots_1 <- ggplot(german_credit, aes(x=Duration.in.month, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
univariate_plots_2 <- ggplot(german_credit, aes(x=Installment.rate.in.percentage.of.disposable.income, fill=factor(Default_status))) + geom_bar()  + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5) 
univariate_plots_3 <- ggplot(german_credit, aes(x=Present.residence.since, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
univariate_plots_4 <- ggplot(german_credit, aes(x=Age.in.Years, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
univariate_plots_5 <- ggplot(german_credit, aes(x=Number.of.existing.credits.at.this.bank., fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)

grid.arrange(univariate_plots_0, 
             univariate_plots_1,
             ncol = 1, top = "Univariate Analysis")

grid.arrange(univariate_plots_2 ,
             univariate_plots_3,
             ncol = 2, top = "Univariate Analysis")

grid.arrange(univariate_plots_4 ,
             univariate_plots_5,
             ncol = 1, top = "Univariate Analysis")


factor_plots_1 <- ggplot(german_credit, aes(x=Status.of.existing.checking.account, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_2 <- ggplot(german_credit, aes(x=Credit.history, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_3 <- ggplot(german_credit, aes(x=Purpose, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_4 <- ggplot(german_credit, aes(x=Savings.account.bonds, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_5 <- ggplot(german_credit, aes(x=Present.employment.since., fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_6 <- ggplot(german_credit, aes(x=Personal.status.and.sex, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_7 <- ggplot(german_credit, aes(x=Other.debtors...guarantors, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_8 <- ggplot(german_credit, aes(x=Property, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_9 <- ggplot(german_credit, aes(x=Other.installment.plans, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_10 <- ggplot(german_credit, aes(x=Housing., fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_11 <- ggplot(german_credit, aes(x=Number.of.existing.credits.at.this.bank., fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_12 <- ggplot(german_credit, aes(x=Telephone., fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)
factor_plots_13 <- ggplot(german_credit, aes(x=foreign.worker, fill=factor(Default_status))) + geom_bar() + stat_count(geom = "text", aes(label = paste(round((..count..)/sum(..count..)*100), "%")), vjust = 5)

grid.arrange(factor_plots_1,
             factor_plots_2,
             factor_plots_3,
             factor_plots_4,
             factor_plots_5,
             factor_plots_6,
             ncol = 2, top = "Univariate Analysis Factors 1")
grid.arrange(factor_plots_7,
             factor_plots_8,
             factor_plots_9,
             factor_plots_10,
             factor_plots_11,
             factor_plots_12,
             factor_plots_13,
             ncol = 2, top = "Univariate Analysis Factors 2")



#--Generate dummy variables for factor variables. 
Status_of_existing_checking_account_Dummy <- data.frame(model.matrix( ~Status.of.existing.checking.account, data = german_credit))
Status_of_existing_checking_account_Dummy <- Status_of_existing_checking_account_Dummy[,-1]

Credit_history_Dummy <- data.frame(model.matrix( ~Credit.history, data = german_credit))
Credit_history_Dummy <-Credit_history_Dummy[,-1]

Purpose_Dummy  <- data.frame(model.matrix( ~Purpose, data = german_credit))
Purpose_Dummy  <- Purpose_Dummy[,-1]

Savings_account_bonds_Dummy <- data.frame(model.matrix( ~Savings.account.bonds, data = german_credit))
Savings_account_bonds_Dummy <- Savings_account_bonds_Dummy[,-1]

Present_employment_since_Dummy <- data.frame(model.matrix( ~Present.employment.since., data = german_credit))
Present_employment_since_Dummy <- Present_employment_since_Dummy[,-1]

Personal_status_and_sex_Dummy <- data.frame(model.matrix( ~Personal.status.and.sex, data = german_credit))
Personal_status_and_sex_Dummy <- Personal_status_and_sex_Dummy[,-1]

Other_debtors_guarantors_Dummy <- data.frame(model.matrix( ~Other.debtors...guarantors, data = german_credit))
Other_debtors_guarantors_Dummy <- Other_debtors_guarantors_Dummy[,-1]

Property_Dummy <- data.frame(model.matrix( ~Property, data = german_credit))
Property_Dummy <- Property_Dummy[,-1]

Other_installment_plans_Dummy <- data.frame(model.matrix( ~Other.installment.plans, data = german_credit))
Other_installment_plans_Dummy <- Other_installment_plans_Dummy[,-1]

Housing_Dummy <- data.frame(model.matrix( ~Housing., data = german_credit))
Housing_Dummy <- Housing_Dummy[,-1]

Number_of_existing_credits_at_this_bank_Dummy <- data.frame(model.matrix( ~Number.of.existing.credits.at.this.bank., data = german_credit))
Number_of_existing_credits_at_this_bank_Dummy <- data.frame(Number_of_existing_credits_at_this_bank_Dummy[,-1])

Telephone_Dummy <- data.frame(model.matrix( ~Telephone., data = german_credit))
Telephone_Dummy <- data.frame(Telephone_Dummy[,-1])

foreign_worker_Dummy <- data.frame(model.matrix( ~foreign.worker, data = german_credit))
foreign_worker_Dummy <- data.frame(foreign_worker_Dummy[,-1])

german_credit_dummy <- subset(german_credit, select = -c(Status.of.existing.checking.account,
                                                         Credit.history,
                                                         Purpose,
                                                         Savings.account.bonds,
                                                         Present.employment.since.,
                                                         Personal.status.and.sex,
                                                         Other.debtors...guarantors,
                                                         Property,
                                                         Other.installment.plans,
                                                         Housing.,
                                                         Number.of.existing.credits.at.this.bank.,
                                                         Telephone.,
                                                         foreign.worker))

german_credit_dummy <- cbind(german_credit_dummy, Status_of_existing_checking_account_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Credit_history_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Purpose_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Savings_account_bonds_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Present_employment_since_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Personal_status_and_sex_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Other_debtors_guarantors_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Property_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Other_installment_plans_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Housing._Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Number_of_existing_credits_at_this_bank_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Telephone_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, foreign_worker_Dummy )


#######################################################################
# Checkpoint 3: Splitting the Dataset into train and test
#######################################################################

set.seed(100)
split_german_credit = sample.split(german_credit_dummy$Default_status, SplitRatio = 0.7)
table(split_german_credit)
german_credit_train = german_credit_dummy[split_german_credit,]
german_credit_test = german_credit_dummy[!(split_german_credit),]



#######################################################################
# Checkpoint 4:Modeling
#######################################################################

#--In this step, you will be actually building the logistic regression model on the data set.
#--Make your initial model including all the variables and then select variables using step-wise function in R
#--The model from the step-wise procedure should be checked using for multicollinearity using VIF in R
#--(use a VIF threshold of 3).
#--Report the AIC value and Null deviance and Residual Deviance of the final model
#--(Although in the lectures we discussed the importance of subject matter knowledge for selection of variables, for this exercise selection of variables has to be done only on the basis of step-wise selection and VIF procedure).

# Model with all variables
initial_model = glm(Default_status ~ ., data = german_credit_train, family = "binomial")
summary(initial_model)
#-- Null deviance: 691.21  on 598  degrees of freedom
#-- Residual deviance: 512.39  on 543  degrees of freedom
#-- AIC: 624.39      
# Stepwise selection of variables

better_model = step(initial_model, direction = "both")

summary(better_model)

#-- Model after STEP
#glm(formula = Default_status ~ Duration.in.month + Installment.rate.in.percentage.of.disposable.income + 
#      Age.in.Years + Status.of.existing.checking.accountA12 + Status.of.existing.checking.accountA13 + 
#      Status.of.existing.checking.accountA14 + Credit.historyA32 + 
#      Credit.historyA34 + PurposeA41 + PurposeA42 + PurposeA43 + 
#      PurposeA48 + PurposeA49 + Savings.account.bondsA65 + Present.employment.since.A74 + 
#      Personal.status.and.sexA93 + Other.debtors...guarantorsA103 + 
#      Other.installment.plansA143 + foreign_worker_Dummy....1., 
#    family = "binomial", data = german_credit_train)
#Null deviance: 691.21  on 598  degrees of freedom
#Residual deviance: 540.94  on 579  degrees of freedom
#AIC: 580.94

vif(better_model) #-- All VIF < 3

#Since all VIF are < 3 but still model is quite large 
# I can get still rid of features with  p > 0.1
# Drop 
#-- PurposeA48                                          -1.73407    1.22254  -1.418 0.156068    
#-- Other.debtors...guarantorsA103                      -0.77607    0.52850  -1.468 0.141985    
#-- foreign_worker_Dummy....1.                          -1.38549    1.06985  -1.295 0.195308   


better_model_1 <- glm(formula = Default_status ~ Duration.in.month + Installment.rate.in.percentage.of.disposable.income + 
            Age.in.Years + Status.of.existing.checking.accountA12 + Status.of.existing.checking.accountA13 + 
            Status.of.existing.checking.accountA14 + Credit.historyA32 + 
            Credit.historyA34 + PurposeA41 + PurposeA42 + PurposeA43 + 
            PurposeA49 + Savings.account.bondsA65 + Present.employment.since.A74 + 
            Personal.status.and.sexA93 +  Other.installment.plansA143 , 
           family = "binomial", data = german_credit_train)
vif(better_model_1)
summary(better_model_1)
#-- Null deviance: 691.21  on 598  degrees of freedom
#-- Residual deviance: 548.28  on 582  degrees of freedom << slight increase
#-- AIC: 582.28 << slight increase 


# Drop 
#-  Installment.rate.in.percentage.of.disposable.income  0.15796    0.10195   1.549 0.121278
#-  Other.installment.plansA143                         -0.40624    0.26002  -1.562 0.118208    

better_model_2 <- glm(formula = Default_status ~ Duration.in.month + 
                        Age.in.Years + Status.of.existing.checking.accountA12 + 
                        Status.of.existing.checking.accountA13 + 
                        Status.of.existing.checking.accountA14 + Credit.historyA32 + 
                        Credit.historyA34 + PurposeA41 + PurposeA42 + PurposeA43 + 
                        PurposeA49 + Savings.account.bondsA65 + Present.employment.since.A74 + 
                        Personal.status.and.sexA93 , 
                      family = "binomial", data = german_credit_train)
vif(better_model_2)
summary(better_model_2)
#-- Null deviance: 691.21  on 598  degrees of freedom
#-- Residual deviance: 553.02  on 584  degrees of freedom
#-- AIC: 583.02

final_model <- better_model
#
#######################################################################
# Checkpoint 5: Model Evaluation
#######################################################################
#-- evaluate the model using C-statistic and KS-statistic for both train and test data. 
#-- Based on the values of C-statistic and KS-statistic, determine whether your model has good accuracy or not.

## C-statistic
#install.packages("Hmisc")
library(Hmisc)
german_credit_test$predicted_prob = predict(final_model, newdata = german_credit_test, type = "response")
german_credit_train$predicted_prob = predict(final_model, newdata = german_credit_train, type = "response")
cstat_test <- rcorr.cens(german_credit_test$predicted_prob, german_credit_test$Default_status)
cstat_train <- rcorr.cens(german_credit_train$predicted_prob, german_credit_train$Default_status)
#-- Test  C Index 0.8035325
#-- Train C Index 0.7968799
#KS-statistic
library(ROCR)
model_score <- prediction(german_credit_test$predicted_prob,german_credit_test$Default_status)
model_score_train <- prediction(german_credit_train$predicted_prob,german_credit_train$Default_status)
model_perf <- performance(model_score, "tpr", "fpr")
model_perf_train <- performance(model_score_train, "tpr", "fpr")
ks_table <- attr(model_perf, "y.values")[[1]] - (attr(model_perf, "x.values")[[1]])
ks_table_train <- attr(model_perf_train, "y.values")[[1]] - (attr(model_perf_train, "x.values")[[1]])

ks = max(ks_table) 
#-- Get where D is max  0.498366 

ks_train = max(ks_table_train) 
#-- Get where D is max  0.47

MaxRowNum <- which(ks_table == ks)
KSBin <- MaxRowNum/length(ks_table) 
#-- which bin this belongs to 0.3622047 ~ 4th decile



#######################################################################
# Checkpoint 6: Threshold value
#######################################################################

#-- After model evaluation, determine the threshold value of probability using ROC curve. 
plot(model_perf, col = "red", lab = c(10,10,10))
#model.perf = performance(model_score, measure ="acc")
#plot(model.perf)

cutoffs <- data.frame(cut=model_perf@alpha.values[[1]], fpr=model_perf@x.values[[1]], 
                      tpr=model_perf@y.values[[1]])

cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]

cutoffs <- data.frame(cut=model_perf@alpha.values[[1]], fpr=model_perf@x.values[[1]], 
                      tpr=model_perf@y.values[[1]])


opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    # Find the minimum distance from the top left of elbow [(x=0,y=1)]
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
}
print(opt.cut(model_perf, model_score))
#--sensitivity 0.7205882
#--specificity 0.7777778
#--cutoff      0.3505962

#-- Once the optimal value of threshold value is determined, generate misclassification 
#-- table for both train and test data and report the following:
#--Sensitivity
#--Specificity
#--Overall Accuracy

confusionMatrix(as.numeric(german_credit_train$predicted_prob > 0.35),german_credit_train$Default_status, positive = "1")
#--Overall Training Accuracy=0.7746
#--Training  Sensitivity=0.6329
#--Training Specificity=0.8254

confusionMatrix(as.numeric(german_credit_test$predicted_prob > 0.35),german_credit_test$Default_status, positive = "1")
#-- Overall Testing Accuracy=0.7626
#-- Testing Sensitivity=0.7206
#-- Testing Specificity=0.7778

