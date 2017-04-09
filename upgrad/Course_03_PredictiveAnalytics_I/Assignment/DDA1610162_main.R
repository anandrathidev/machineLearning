#-----------------------Assignment- Linear Regression---------------------------------------
#-------------------------------------------------------------------------------------------
# Install packages 
#for StepAIC
#install.packages("MASS") 

#for VIF
#install.packages("car") 

library(MASS)
library(car)
library(ggplot2)
#-------------------------------------------------------------------------------------------
# Set the Working directory first 
#setwd("C:/Users/rb117/Documents/Personal/Upgrad/Course3_assingment")
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/Assignment/")
carmileage <- read.csv("carMPG.csv")

#-------------------------------------------------------------------------------------------
#-----------------------Checkpoit 1 Business Understanding---------------------------------------
# predict the city-cycle fuel consumption in miles per gallon, 
# in terms of 3 multivalued discrete and 5 continuous attributes. 
# multivalued discrete attributes = (Cylinders, Model.year, Origin)
# continuous attributes = (Displacement, Horsepower, Weight, Acceleration )
# Prediction variable = (mpg) 
# develop a predictive model which can follow these three constraints thoroughly: 

# The model should not contain more than 5 variables.
# According to the business needs, set the VIF to 2. 
# The model should be highly predictive in nature i.e it should show 80% (R squared) of accuracy.
#-------------------------------------------------------------------------------------------
# Understand data 
summary(carmileage)
str(carmileage)


#-------------------------------------------------------------------------------------------
#-----------------------Checkpoint 2: Data Cleaning and Preparation---------------------------------------
# predict the city-cycle fuel consumption in miles per gallon, 
# in terms of 3 multivalued discrete and 5 continuous attributes. 
# multivalued discrete attributes = (Cylinders, Model_year, Origin)
# continuous attributes = (Displacement, Horsepower, Weight, Acceleration )
# Prediction variable = (mpg) 
#-------------------------------------------------------------------------------------------
# First stage: Variables Formatting  
carmileage_clean <- carmileage
#remove "?" 6 of 398 = 1.5% removals wont hurt 
carmileage_clean <- carmileage_clean[-which(carmileage_clean$Horsepower=="?"),]

# Will factor these to build matrix 
carmileage_clean$Car_Name <- as.character(carmileage_clean$Car_Name)
carmileage_clean$Horsepower  <- as.numeric(levels(carmileage_clean$Horsepower))[carmileage_clean$Horsepower] 
#Extract Brand names 
carmileage_clean$Brand_Name <-  sapply(strsplit(carmileage_clean$Car_Name,  split = c(" ")), FUN = function(x) { x[1] }) 

# since Model_year, Cylinders, can be directly converted to Numeric as they have meaning full interval vars
# like year 2010 > 2009 etc also cyl 8 > cyl 6 
carmileage_clean$Model_year <-  as.numeric(as.character(carmileage_clean$Model_year))
carmileage_clean$Cylinders <-  as.numeric(as.character(carmileage_clean$Cylinders))

# Origin is an ordinal vars , however there is no meaning ful interval between them so we will create dummies
carmileage_clean$Origin <-  as.factor(as.character(carmileage_clean$Origin))

#Replace synonyms 
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="vw")]<- "volkswagen"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="vokswagen")]<- "volkswagen"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="toyouta")]<- "toyota"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="mercedes-benz")]<- "mercedes"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="maxda")]<- "mazda"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="chevy")]<- "chevroelt"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="chevroelt")]<- "chevrolet"

#Remove that single brands: hi,capri,nissan,triumph 4/388 = 1% So wont make diff
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="hi"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="capri"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="nissan"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="triumph"),]

#Brand_Name should be  factor 
carmileage_clean$Brand_Name <- as.factor(carmileage_clean$Brand_Name)


# TO be used later on OPTIONAL
carmileage_clean$Model_year_factor <-  as.factor(as.character(carmileage_clean$Model_year))
carmileage_clean$Cylinders_factor <-  as.factor(as.character(carmileage_clean$Cylinders))

#-------------------------------------------------------------------------------------------
#Third Stage: Variables Transformation
# Create the dummy variables

# For Brand_Name  
Brand_Name <- data.frame(model.matrix( ~Brand_Name, data = carmileage_clean))
Brand_Name<-Brand_Name[,-1]

Origin_factor  <- data.frame(model.matrix( ~Origin, data = carmileage_clean))
Origin_factor <- Origin_factor[,-1]


# To be used later // optional for more insigts, which year/s
Model_year_factor <- data.frame(model.matrix( ~Model_year_factor, data = carmileage_clean))
Model_year_factor <- Model_year_factor[,-1]


# As per Instruction Store the dataset into "carmileage" object.
carmileage <- cbind(carmileage_clean[ c("MPG", "Displacement", "Horsepower", "Weight", "Acceleration", "Cylinders", "Model_year")], Brand_Name, Origin_factor, Model_year_factor)
model_1 <-lm(MPG~.,data=carmileage)

if(length(boxplot.stats(carmileage$MPG)$out)==0) {
  print("No Outliers in MPG")
}else {
  print("Outliers in MPG !!!!!!!!!!!!!!!!!!!!!!!!!!")
}


#-------------------------------------------------------------------------------------------
# 
# Divide you data in 70:30 

set.seed(100)
indices= sample(1:nrow(carmileage), 0.7*nrow(carmileage))

train = carmileage[indices,]
test = carmileage[-indices,]


#-------------------------------------------------------------------------------------------
# Checkpoint 3: Model Development
#------------------------------Multiple Linear regression-----------------------------------
#-------------------------------------------------------------------------------------------
# Develop the first model 
model_1 <-lm(MPG~Displacement+Horsepower+Weight+Cylinders+Origin2+Origin3+Model_year+
Brand_Nameaudi+Brand_Namecadillac+Brand_Namebuick+Brand_Namecadillac+Brand_Namechevrolet+Brand_Namechrysler+Brand_Namedatsun+Brand_Namedodge+Brand_Namefiat+Brand_Nameford+Brand_Namehonda+Brand_Namemazda+Brand_Namemercedes+Brand_Namemercury+Brand_Nameoldsmobile+Brand_Nameopel+Brand_Namepeugeot+Brand_Nameplymouth+Brand_Nametoyota+Brand_Namepontiac+Brand_Namerenault+
Brand_Namesaab+Brand_Namevolkswagen+Brand_Namevolvo,data=train)
summary(model_1)

#-------------------------------------------------------------------------------------------
# Apply the stepwise approach
step <- stepAIC(model_1, direction="both")

#-------------------------------------------------------------------------------------------
# Run the step object
step
step$anova
#-------------------------------------------------------------------------------------------
# create a new model_2 after stepwise method
# from  step$anova
#Final Model:
#  MPG ~ Displacement + Horsepower + Weight + Model_year + Brand_Nameaudi + 
#   Brand_Namedatsun + Brand_Namefiat + Brand_Namehonda + Brand_Namemazda + 
#   Brand_Nameplymouth + Brand_Namepontiac + Brand_Namerenault + 
#   Brand_Namesaab + Brand_Nametoyota + Brand_Namevolkswagen

model_2 <- lm(formula = MPG ~ Displacement + Horsepower + Weight + Origin3 + Model_year + 
                Brand_Nameaudi + Brand_Namedatsun + Brand_Namefiat + Brand_Nameplymouth + 
                Brand_Namepontiac + Brand_Namerenault + Brand_Namesaab + 
                Brand_Namevolkswagen, data = train)

# let's check the summary first
summary(model_2)
# Again calculate VIF of model_2
vif(model_2)

#-------------------------------------------------------------------------------------------
# Now the variable "Displacement" is more collinear So we will  
# remove this variable now and develop the new model_3

model_3 <- lm(formula =  MPG ~  Horsepower + Weight + Origin3 + Model_year + 
                Brand_Nameaudi + Brand_Namedatsun + Brand_Namefiat + Brand_Nameplymouth + 
                Brand_Namepontiac + Brand_Namerenault + Brand_Namesaab + 
                Brand_Namevolkswagen, data = train)

summary(model_3)
# Again calculate VIF of model_3
vif(model_3)

#-------------------------------------------------------------------------------------------
# Now the variable "Horsepower" is more collinear and not so relevant So we will  
# remove this variable now and develop the new model_4

model_4 <- lm(formula = MPG ~  Weight + Origin3 + Model_year + 
                Brand_Nameaudi + Brand_Namedatsun + Brand_Namefiat + Brand_Nameplymouth + 
                Brand_Namepontiac + Brand_Namerenault + Brand_Namesaab + 
                Brand_Namevolkswagen, data = train)

summary(model_4)
# Again calculate VIF of model_4
vif(model_4)
#-------------------------------------------------------------------------------------------
## Now we have all the variables with VIF less than 2
## According to the business needs, set the VIF to 2. 

#-------------------------------------------------------------------------------------------
# Now Keep only relevant variables 
# Weight + Model_year  +Origin3
# Brand_Namedatsun  + Brand_Namevolkswagen
# develop the new model_5
model_5 <- lm(formula = MPG ~ Weight + Model_year + Origin3 +
                Brand_Namedatsun  + Brand_Namevolkswagen, data = train)
summary(model_5)


#-------------------------------------------------------------------------------------------
## We keep TOP Most effective features and check our model
model_final <- lm(formula = MPG ~ Weight + Model_year + Brand_Namevolkswagen,
                  data = train)
summary(model_final)
#-------------------------------------------------------------------------------------------
# Final Features in Model
# MPG, Weight, Model_year, Brand_Namevolkswagen
# Finally we get Multiple R-squared:  0.8063,	Adjusted R-squared:  0.8041
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#  Checkpoint 3: End
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
#  Checkpoint 4: Model Evaluation and Testing 
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Test the model on test dataset

# Add a new column "test_MPG" into the test dataset
test$test_MPG <- predict(model_final,test)
#-------------------------------------------------------------------------------------------
# calculate the test R2 
cor(test$MPG,test$test_MPG)
PredictedR2 <- cor(test$MPG,test$test_MPG)^2
print(paste( "PredictedR2=" , as.character(PredictedR2)))
# PredictedR2= 0.82


#-------------------------------------------------------------------------------------------
#  Checkpoint 5: Model acceptance or Rejection
#-------------------------------------------------------------------------------------------

# The model should not contain more than 5 variables.
# Ans: MPG + 3 is enough
# MPG, Weight, Model_year, Brand_Namevolkswagen

# The model should be highly predictive in nature i.e it should show 80% (R squared) of accuracy.
# Ans: model_final Finally we get Multiple R-squared:  0.80,	Adjusted R-squared:  0.799
summary(model_final)

# The model should give high accuracy (test R-squared ) when tested it on the test dataset.
# test R-squared ==  0.81 > 0.8
print( paste("test R-squared ==", PredictedR2))

#-------------------------------Summary-----------------------------------------------
#-------------------------------------------------------------------------------------
# For good MPG prediction use one of the following model
# At 95 % confidence 
# Weight, Model_year, Brand_Namevolkswagen
# MPG = 1639.8 - 0.006*Wieght - 0.79*Year +  2.34 * Brand_Namevolkswagen + 3.394

###########################################################################
#########################   OPTIONAL  #####################################
###########################################################################


# Dig deeper into year  ## Additional insights
model_2_1 <- lm(formula = MPG ~  Weight +  Model_year_factor2004 + Model_year_factor2005 +
                      Model_year_factor2006 + Model_year_factor2007 + Model_year_factor2008 + Model_year_factor2009 +
                      Model_year_factor2010 + Model_year_factor2011 +  Model_year_factor2012 + 
                      Model_year_factor2012 + Model_year_factor2013 +  Model_year_factor2014 +  
                      Model_year_factor2015 , data = train)

step2 <- stepAIC(model_2_1, direction="both")

#-------------------------------------------------------------------------------------------
# Run the step object
step2
step2$anova

# step removed Model_year_factor2004
model_2_2 <- lm(formula = MPG ~ Weight + Model_year_factor2005 + Model_year_factor2006 + 
                        Model_year_factor2007 + Model_year_factor2008 + Model_year_factor2009 + 
                        Model_year_factor2010 + Model_year_factor2011 + Model_year_factor2012 + 
                        Model_year_factor2013 + Model_year_factor2014 + Model_year_factor2015, data = train)
summary(model_2_2)
# From summary remove less influencial years
# Remove 2005, 2006 

model_2_3 <- lm(formula = MPG ~ Weight +
                        Model_year_factor2007 + Model_year_factor2008 + Model_year_factor2009 + 
                        Model_year_factor2010 + Model_year_factor2011 + Model_year_factor2012 + 
                        Model_year_factor2013 + Model_year_factor2014 + Model_year_factor2015 ,
                        data = train)
# As per this Model top MPG are 
# Weight, Model year > 2007
 
# Multiple R-squared:  0.8276,	Adjusted R-squared:  0.821
model_final_2 = model_2_3
summary(model_final_2)

# calculate the test R2 
# Add a new column "test_MPG_2" into the test dataset
test$test_MPG_2  <- predict(model_final_2, test)

cor(test$MPG,test$test_MPG_2)
PredictedR2_2 <- cor(test$MPG,test$test_MPG_2)^2
print(paste( "PredictedR2=" , as.character(PredictedR2_2)))
# PredictedR2_2= 0.811


# Double check 
# Only Year > 2007
train2 <- train[which(train$Model_year>=2007),]
test2 <- test[which(test$Model_year>=2007),]
model_final_3 <- lm(formula = MPG ~ Weight +  Model_year ,
                      data = train2)

summary(model_final_3)

# calculate the test R2 
# Add a new column "test_MPG_3" into the test dataset
test2$test_MPG_3 <- predict(model_final_3,test2)

cor(test2$MPG, test2$test_MPG_3)
PredictedR2_3 <- cor(test2$MPG,test2$test_MPG_3)^2
print(paste( "PredictedR2=" , as.character(PredictedR2_3)))
# PredictedR2= 0.811

#-------------------------------Summary-----------------------------------------------
#-------------------------------------------------------------------------------------
# For good MPG prediction use one of the following model
# At 95 % confidence 
# 1] Weight, Model_year, Brand_Namevolkswagen
# MPG = 1639.8 - 0.006*Wieght - 0.79*Year +  2.34 * Brand_Namevolkswagen + 3.394
# Or 
# 2] model with Weight, For Model_year > 2007 
#  MPG_formula = 850.78 - 0.00577*Weight - 0.40389*Model_year
#-------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------the End

ggplot(test, alpha = 0.02) + 
  geom_jitter(aes(x=MPG,y=test_MPG), colour="blue") + geom_smooth(aes(x=MPG,y=test_MPG), method=lm, se=FALSE) +
  geom_jitter(aes(x=MPG,y=test_MPG_2), colour="green") + geom_smooth(aes(x=MPG,y=test_MPG_2), method=lm, se=FALSE, colour="green") + 
  geom_text(x=30, y=10, label="MPG = 1639.8 - 0.006*Wieght - 0.79*Year +  2.34 Brand_Namevolkswagen + 3.394",colour="blue") +
  geom_text(x=30, y=13, label="MPG = 850.78 - 0.00577*Weight - 0.40389*Model_year + 2.68",colour="green") +
  labs(x = "Actual MPG", y = "Predicted MPG")
