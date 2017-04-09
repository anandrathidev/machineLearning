
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

#--And outliers for each variable and impute them accordingly.
quantile(german_credit$Duration.in.month, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Credit.amount, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Installment.rate.in.percentage.of.disposable.income, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Present.residence.since, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Age.in.Years, c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Number.of.existing.credits.at.this.bank., c(0.95,0.96,0.97,0.98,0.99,1))
quantile(german_credit$Number.of.people.being.liable.to.provide.maintenance.for., c(0.95,0.96,0.97,0.98,0.99,1))

#------------------------------------------------------------------------
#-- No Outliers Found
#------------------------------------------------------------------------

# Exploratory Data Analysis

Credit_amount_plot <- ggplot(data=german_credit, aes(x = german_credit$Default_status, x=Credit.amount)) 
Credit_amount_plot + geom_bar()

Credit_amount_plot <- Credit_amount_plot + scale_size_area() + xlab("Credit.amount") + ylab("Count") 
Credit_amount_plot <- Credit_amount_plot + ggtitle("Credit amount ")


ggplot(german_credit, aes(x=Default_status, y=Credit.amount, fill=Purpose))+
  geom_point()+
  facet_wrap(~Status.of.existing.checking.account+ Credit.history+  Savings.account.bonds + Present.employment.since. + Personal.status.and.sex)
facet_wrap(~german_credit+ Status.of.existing.checking.account+ Credit.history+  Savings.account.bonds + Present.employment.since. + Personal.status.and.sex)

grade_plot <- ggplot(loan_analysis_data, aes(x=grade)) 
grade_plot <- grade_plot + geom_bar() 
grade_plot <- grade_plot + scale_size_area() + xlab("LC Grade") + ylab("Count") 
grade_plot <- grade_plot + ggtitle("LC Grade")

emp_len_plot <- ggplot(loan_analysis_data, aes(x=emp_len_grp)) 
emp_len_plot <- emp_len_plot + geom_bar() 
emp_len_plot <- emp_len_plot + scale_size_area() + xlab("Employment Length (yrs)") + ylab("Count") 
emp_len_plot <- emp_len_plot + ggtitle("Employment Length")

ownership_plot <- ggplot(loan_analysis_data, aes(x=home_ownership)) 
ownership_plot <- ownership_plot + geom_bar() 
ownership_plot <- ownership_plot + scale_size_area() + xlab("Home ownership") + ylab("Count") 
ownership_plot <- ownership_plot + ggtitle("Home Ownership Status")

loan_status_1_plot <- ggplot(loan_analysis_data, aes(x=loan_status_1)) 
loan_status_1_plot <- loan_status_1_plot + geom_bar()  
loan_status_1_plot <- loan_status_1_plot + scale_size_area() + xlab("Loan Status") + ylab("Count") 
loan_status_1_plot <- loan_status_1_plot + ggtitle("Loan Status")


grid.arrange(int_rate_plot,
             grade_plot ,
             emp_len_plot,
             ownership_plot ,
             loan_status_1_plot,
             ncol = 2, top = "Categorical Data Plots")



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

german_credit_dummy <- cbind(german_credit_dummy, Status.of.existing.checking.account_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Credit.history_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Purpose_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Savings.account.bonds_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Present.employment.since._Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Personal.status.and.sex_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Other.debtors...guarantors_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Property_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Other.installment.plans_Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Housing._Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Number.of.existing.credits.at.this.bank._Dummy )
german_credit_dummy <- cbind(german_credit_dummy, Telephone._Dummy )
german_credit_dummy <- cbind(german_credit_dummy, foreign.worker_Dummy )


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

# Stepwise selection of variables
better_model = step(initial_model,direction = "both")
summary(better_model)
# AIC: 817.93

vif(better_model)
# Remove Credit.amount vif = 2.108649 high p val 
better_model_1 <- glm(formula = Default_status ~ Duration.in.month +  
      Installment.rate.in.percentage.of.disposable.income + Age.in.Years + 
      PurposeA41 + PurposeA410 + PurposeA42 + PurposeA43 + PurposeA49 + 
      PropertyA124, family = "binomial", data = german_credit_train)

# AIC: 801.01
summary(better_model_1)
vif(better_model_1)
#-- Null deviance: 855.21  on 699  degrees of freedom
#-- Residual deviance: 781.01  on 690  degrees of freedom

better_model_2 <- glm(formula = Default_status ~ Duration.in.month +  
                        +       Installment.rate.in.percentage.of.disposable.income + Age.in.Years + 
                        +       PurposeA41 + PurposeA42 + PurposeA43 + PurposeA49 + 
                        +       PropertyA124, family = "binomial", data = german_credit_train)
summary(better_model_2)
#-- Null deviance: 855.21  on 699  degrees of freedom
#-- Residual deviance: 783.27  on 691  degrees of freedom
#-- AIC: 801.27

better_model_3 <- glm(formula = Default_status ~ Duration.in.month +  
                        +       Age.in.Years + 
                        +       PurposeA41 + PurposeA42 + PurposeA43 + PurposeA49 + 
                        +       PropertyA124, family = "binomial", data = german_credit_train)
summary(better_model_3)
#-- Null deviance: 855.21  on 699  degrees of freedom
#-- Residual deviance: 785.88  on 692  degrees of freedom
#-- AIC: 801.88

better_model_4 <- glm(formula = Default_status ~ Duration.in.month +  
                        +       Age.in.Years + 
                        +       PurposeA41 +  PurposeA43 + PurposeA49 + 
                        +       PropertyA124, family = "binomial", data = german_credit_train)
summary(better_model_4)

#-- Null deviance: 855.21  on 699  degrees of freedom
#-- Residual deviance: 788.31  on 693  degrees of freedom
#-- AIC: 802.31

better_model_5 <- glm(formula = Default_status ~ Duration.in.month +  
                        +       Age.in.Years + 
                        +       PurposeA41 +  PurposeA43  + 
                        +       PropertyA124, family = "binomial", data = german_credit_train)

summary(better_model_5)

#######################################################################
# Checkpoint 5: Model Evaluation
#######################################################################
#--evaluate the model using C-statistic and KS-statistic for both train and test data. Based on the values of C-statistic and KS-statistic, determine whether your model has good accuracy or not.

#-- Null deviance: 855.21  on 699  degrees of freedom
#-- Residual deviance: 790.07  on 694  degrees of freedom
#-- AIC: 802.07

#######################################################################
# Checkpoint 6: Threshold value
#######################################################################

#--After model evaluation, determine the threshold value of probability using ROC curve. Once the optimal value of threshold value is determined, generate misclassification table for both train and test data and report the following:
#--Sensitivity
#--Specificity
#--Overall Accuracy
