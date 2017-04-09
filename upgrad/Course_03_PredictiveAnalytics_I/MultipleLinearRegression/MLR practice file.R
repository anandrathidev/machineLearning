
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/MultipleLinearRegression//")

#---------------------Marketing Dataset-----------------------------------
#-------------------------------------------------------------------------

# load Advertising dataset

advertisement<-read.csv("Advertising.csv")


#-------------------------------------------------------------------------

# Check the structure of Dataset 

str(advertisement)

#-------------------------------------------------------------------------
# Let's build the first model with all the parameters

model_1 <- lm(Sales~.,data=advertisement)


#-------------------------------------------------------------------------
# check the summary of model

summary(model_1)

#-------------------------------------------------------------------------
#Now, check the P value of variables 

# Remove Newspaper variable from the model 

model_2 <- lm(Sales~.-Newspaper,data=advertisement)
summary(model_2)

#-------------------------------------------------------------------------
