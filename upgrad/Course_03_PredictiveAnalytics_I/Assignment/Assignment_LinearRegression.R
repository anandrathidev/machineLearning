#-----------------------Assignment- Linear Regression---------------------------------------

#-------------------------------------------------------------------------------------------
# Install packages 


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
#-------------------------------------------------------------------------------------------
# Understand data 
summary(carmileage)
str(carmileage)


#-------------------------------------------------------------------------------------------
#-----------------------Checkpoint 2: Data Cleaning and Preparation---------------------------------------
#-------------------------------------------------------------------------------------------
# First stage: Variables Formatting  
carmileage_clean <- carmileage
#remove "?"
carmileage_clean <- carmileage_clean[-which(carmileage_clean$Horsepower=="?"),]

# Will factor these to build matrix 
carmileage_clean$Car_Name <- as.character(carmileage_clean$Car_Name)
carmileage_clean$Horsepower  <- as.numeric(levels(carmileage_clean$Horsepower))[carmileage_clean$Horsepower] 
#Extract Brand names 
carmileage_clean$Brand_Name <-  sapply(strsplit(carmileage_clean$Car_Name,  split = c(" ")), FUN = function(x) { x[1] }) 

# since Model_year, Cylinders, Origin can be directly converted to Numeric we will

carmileage_clean$Model_year <-  as.numeric(as.character(carmileage_clean$Model_year))
carmileage_clean$Cylinders <-  as.numeric(as.character(carmileage_clean$Cylinders))
carmileage_clean$Origin <-  as.numeric(as.character(carmileage_clean$Origin))

#Replace synonyms 
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="vw")]<- "volkswagen"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="vokswagen")]<- "volkswagen"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="toyouta")]<- "toyota"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="mercedes-benz")]<- "mercedes"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="maxda")]<- "mazda"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="chevy")]<- "chevroelt"
carmileage_clean$Brand_Name[which(carmileage_clean$Brand_Name=="chevroelt")]<- "chevrolet"

#Remove that single brands: hi,capri,nissan,triumph
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="hi"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="capri"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="nissan"),]
carmileage_clean <-  carmileage_clean[which(carmileage_clean$Brand_Name!="triumph"),]

#Brand_Name should be  factor 
carmileage_clean$Brand_Name <- as.factor(carmileage_clean$Brand_Name)

#-------------------------------------------------------------------------------------------
#Third Stage: Variables Transformation
# Create the dummy variables

# For Brand_Name  
Brand_Name <- data.frame(model.matrix( ~Brand_Name, data = carmileage_clean))
Brand_Name<-Brand_Name[,-1]

# As per Instruction Store the dataset into "carmileage" object.
carmileage <- cbind(carmileage_clean[ c("MPG", "Displacement", "Horsepower", "Weight", "Acceleration", "Cylinders", "Origin", "Model_year")], Brand_Name)
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
model_1 <-lm(price~.,data=train[,-1])
summary(model_1)

#-------------------------------------------------------------------------------------------
# Apply the stepwise approach
step <- stepAIC(model_1, direction="both")

#-------------------------------------------------------------------------------------------
# Run the step object
step

#-------------------------------------------------------------------------------------------
# create a new model_2 after stepwise method
# from  step$call
print(step$call)
model_2 <- lm(formula = MPG ~ Displacement + Horsepower + Weight + Cylinders + 
                Model_year + Origin + Brand_Nameaudi + Brand_Namecadillac + 
                Brand_Namedatsun + Brand_Namedodge + Brand_Namefiat + Brand_Namehonda + 
                Brand_Nameoldsmobile + Brand_Nameplymouth + Brand_Namepontiac + 
                Brand_Namevolkswagen, data = carmileage)

# let's check the summary first
summary(model_2)
# Again calculate VIF of model_2
vif(model_2)

#-------------------------------------------------------------------------------------------
# Now the variable "Displacement" is more collinear So we will  
# remove this variable now and develop the new model_3

model_3 <- lm(formula = MPG ~ Horsepower + Weight + Cylinders + 
                   Model_year + Origin + Brand_Nameaudi + Brand_Namecadillac + 
                   Brand_Namedatsun + Brand_Namedodge + Brand_Namefiat + Brand_Namehonda + 
                   Brand_Nameoldsmobile + Brand_Nameplymouth + Brand_Namepontiac + 
                   Brand_Namevolkswagen, data = carmileage)


summary(model_3)
# Again calculate VIF of model_3
vif(model_3)

#-------------------------------------------------------------------------------------------
# Now the variable "Cylinders" is more collinear and not so relevant So we will  
# remove this variable now and develop the new model_4

model_4 <- lm(formula = MPG ~ Horsepower + Weight + 
                Model_year + Origin + Brand_Nameaudi + Brand_Namecadillac + 
                Brand_Namedatsun + Brand_Namedodge + Brand_Namefiat + Brand_Namehonda + 
                Brand_Nameoldsmobile + Brand_Nameplymouth + Brand_Namepontiac + 
                Brand_Namevolkswagen, data = carmileage)


summary(model_4)
# Again calculate VIF of model_3
vif(model_4)

#-------------------------------------------------------------------------------------------
#2nd high vif but high p val 
# Now the variable "Horsepower" is more collinear So we will  
# remove this variable now and develop the new model_5

model_5 <- lm(formula = MPG ~  Weight + 
             Model_year + Origin + Brand_Nameaudi + Brand_Namecadillac + 
             Brand_Namedatsun + Brand_Namedodge + Brand_Namefiat + Brand_Namehonda + 
             Brand_Nameoldsmobile + Brand_Nameplymouth + Brand_Namepontiac + 
             Brand_Namevolkswagen, data = carmileage)


summary(model_5)
# Again calculate VIF of model_3
vif(model_5)

#-------------------------------------------------------------------------------------------
# Now the variable "Origin" is more collinear So we will  
# remove this variable now and develop the new model_5

model_6 <- lm(formula = MPG ~  Weight + 
                Model_year + Brand_Nameaudi + Brand_Namecadillac + 
                Brand_Namedatsun + Brand_Namedodge + Brand_Namefiat + Brand_Namehonda + 
                Brand_Nameoldsmobile + Brand_Nameplymouth + Brand_Namepontiac + 
                Brand_Namevolkswagen, data = carmileage)

summary(model_6)
# Again calculate VIF of model_6
vif(model_6)

#-------------------------------------------------------------------------------------------
## Now we have all the variables with VIF less than 2
## According to the business needs, set the VIF to 2. 

#Weight           Model_year       Brand_Nameaudi   Brand_Namecadillac     Brand_Namedatsun 
#1.466773             1.149322             1.022579             1.018842             1.111041 
#Brand_Namedodge       Brand_Namefiat      Brand_Namehonda Brand_Nameoldsmobile   Brand_Nameplymouth 
#1.048883             1.061790             1.098317             1.038099             1.048592 
#Brand_Namepontiac Brand_Namevolkswagen 
#1.050577             1.145061 


#-------------------------------------------------------------------------------------------
## We keep TOP 4 Most effective features and check our model
model_final <- lm(formula = MPG ~  Weight + 
                Model_year +  Brand_Namedatsun + Brand_Namevolkswagen , data = carmileage)
summary(model_final)
#-------------------------------------------------------------------------------------------
# Final Features in Model
# MPG, Weight, Model_year, Brand_Namedatsun, Brand_Namevolkswagen
# Finally we get Multiple R-squared:  0.8147,	Adjusted R-squared:  0.8128 
#-------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------
#  Checkpoint 3: End
#-------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------
#  Checkpoint 4: Model Evaluation and Testing 
#-------------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------
# Test the model on test dataset
Predict_1 <- predict(model_final,test)

#-------------------------------------------------------------------------------------------

# Add a new column "test_MPG" into the test dataset
test$test_MPG <- Predict_1
#-------------------------------------------------------------------------------------------

# calculate the test R2 

cor(test$MPG,test$test_MPG)
PredictedR2 <- cor(test$MPG,test$test_MPG)^2


#-------------------------------------------------------------------------------------------
#  Checkpoint 5: Model acceptance or Rejection
#-------------------------------------------------------------------------------------------

# The model should not contain more than 5 variables.
# Ans: MPG, + 4 
# Weight, Model_year, Brand_Namedatsun, Brand_Namevolkswagen

# The model should be highly predictive in nature i.e it should show 80% (R squared) of accuracy.
# Ans: model_final Finally we get Multiple R-squared:  0.8147,	Adjusted R-squared:  0.8128
summary(model_final)

# The model should give high accuracy (test R-squared ) when tested it on the test dataset.
# PredictedR2 ==  0.8325276 > 0.8


#-------------------------------------------------------------------------------------the End

