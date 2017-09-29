install.packages("MASS")
install.packages("car")
install.packages("e1071")
install.packages("ROCR")
install.packages("caret")

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

#setwd("C:\\Users\\Vishesh Sakshi\\Documents\\Upgrad assignments\\BFSI domain\\Assignment")
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_06_elective\\ecommerce\\Assingment\\")

bank_m_data <- read.csv("bank_marketing.csv", header = TRUE, stringsAsFactors = F)

str(bank_m_data)
summary(bank_m_data)
sum(is.na(bank_m_data))

###############################################################################
#-- Checkpoint : Exploratory Data Analysis
###############################################################################

ggplot(bank_m_data, aes(age, fill=bank_m_data$age)) + geom_bar()
ggplot(bank_m_data, aes(marital, fill=bank_m_data$marital)) + geom_bar()
ggplot(bank_m_data, aes(job, fill=bank_m_data$job)) + geom_bar()
ggplot(bank_m_data, aes(education, fill=bank_m_data$education)) + geom_bar()
ggplot(bank_m_data, aes(housing, fill=bank_m_data$housing)) + geom_bar()
ggplot(bank_m_data, aes(loan, fill=bank_m_data$loan)) + geom_bar()
ggplot(bank_m_data, aes(month, fill=bank_m_data$month)) + geom_bar()
ggplot(bank_m_data, aes(contact, fill=bank_m_data$contact)) + geom_bar()
ggplot(bank_m_data, aes(day_of_week, fill=bank_m_data$day_of_week)) + geom_bar()
ggplot(bank_m_data, aes(poutcome, fill=bank_m_data$poutcome)) + geom_bar()

###############################################################################
#-- Checkpoint : removing column "duration" and adding unqiueID column
###############################################################################
bank_m_data$uid <- 1:nrow(bank_m_data)
bank_m_data_1 <- bank_m_data[,-11]
View(bank_m_data_1)

#Checking for outlier in Age.in.Years
boxplot.stats(bank_m_data_1$age)
quantile(bank_m_data_1$age, seq(0,1,0.01))

# we got ouliers in Age 99% 71 and 100% 98 - capping outlier at 71
bank_m_data_1 <- subset(bank_m_data_1,bank_m_data_1$age <=71)

# Binning the variable Age
#bank_m_data_1$Age_Group [bank_m_data_1$age >16 & bank_m_data_1$age <= 40] <- "Young_Age"
#bank_m_data_1$Age_Group [bank_m_data_1$age >40 & bank_m_data_1$age <= 60] <- "Mid_Age"
#bank_m_data_1$Age_Group [bank_m_data_1$age >60] <- "Old_Age"

str(bank_m_data_1)
View(bank_m_data_1)
#Converting respone variable to numeric
bank_m_data_1$response_num <- as.integer(ifelse(bank_m_data_1$response == "yes", "1", "0"))

# Generate dummy variables for factor variables. 
summary(bank_m_data_1)
str(bank_m_data_1)

bank_m_data2 <- bank_m_data_1
str(bank_m_data2)

bank_fact <- bank_m_data2[,-c(1,11,12,13,15,16,17,18,19,20,21,22)] 
str(bank_fact)
bank_factor<- as.data.frame( sapply(bank_fact, function(x) as.factor(x)))
str(bank_factor)

dummy1 <- data.frame(model.matrix(~ job -1 , data = bank_factor)[ ,-1] )
dummy2 <- data.frame(model.matrix(~ marital -1 , data = bank_factor)[ ,-1] )
dummy3 <- data.frame(model.matrix(~ education -1 , data = bank_factor)[ ,-1] )
dummy4 <- data.frame(model.matrix(~ default -1 , data = bank_factor)[ ,-1] )
dummy5 <- data.frame(model.matrix(~ housing -1 , data = bank_factor)[ ,-1] )
dummy6 <- data.frame(model.matrix(~ loan -1 , data = bank_factor)[ ,-1] )
dummy7 <- data.frame(model.matrix(~ contact -1 , data = bank_factor)[ ,-1] )
dummy8 <- data.frame(model.matrix(~ month -1 , data = bank_factor)[ ,-1] )
dummy9 <- data.frame(model.matrix(~ day_of_week -1 , data = bank_factor)[ ,-1] )
dummy10 <- data.frame(model.matrix(~ poutcome -1 , data = bank_factor)[ ,-1] )


bank_final <- cbind(dummy1,dummy2,dummy3,dummy4,dummy5,dummy6,dummy7,dummy8,dummy9,dummy10,bank_m_data2[,c(1,11,12,13,15,16,17,18,19,22)])
str(bank_final)
View(bank_final)

## Scaling the dataset
#range01 <- function(x){(x-min(x))/(max(x)-min(x))}
#head(range01(bank_final[, -53]))
#scaled_data1 <- range01(bank_final[, -53])
#str(scaled_data1)
#View(scaled_data1)
#scaled_data <- cbind(scaled_data1,bank_fact$response_num)

set.seed(100)
indices = sample(1:nrow(bank_final), 0.7*nrow(bank_final))
train_data = bank_final[indices,]
test_data = bank_final[-indices,]

initial_model <-  glm(response_num ~ ., data = train_data, family = "binomial")
summary(initial_model)

step<-stepAIC(initial_model,direction = "both")
print(step$call)

final_model <-  glm(response_num ~ jobretired + jobstudent + maritalmarried + 
                        educationprofessional.course + educationuniversity.degree + 
                        defaultunknown + model.matrix..contact...1..data...bank_factor.....1. + 
                        monthaug + monthdec + monthjun + monthmar + monthmay + monthnov + 
                        monthoct + monthsep + day_of_weekmon + day_of_weekthu + day_of_weekwed + 
                        poutcomenonexistent + poutcomesuccess + campaign + pdays + 
                        previous + emp.var.rate + cons.price.idx + cons.conf.idx + 
                        euribor3m + nr.employed
                      , data = train_data, family = "binomial")
predict(final_model, test_data, type="response")