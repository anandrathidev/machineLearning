# Simple Linear regression 
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/SimpleLinearRegression/")
# Load advertising dataset 

advertising <- read.csv("advertising.csv")

# structure of dataset

str(advertising)

# Make the linear regression model 

model<-lm(Sales~Advertising_budget,advertising)

# summary of model 

summary(model)

# Predict the sales value at 400 and 500.

Predict_1 <- predict(model,data.frame(Advertising_budget=c(400,500)))

Predict_1

