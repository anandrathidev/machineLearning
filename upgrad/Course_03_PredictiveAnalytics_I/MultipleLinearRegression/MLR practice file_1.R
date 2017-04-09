
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/MultipleLinearRegression//")

#-----------------------Multiple Linear regression------------------------------------------
#-------------------------------------------------------------------------------------------

# Set the Working directory first 

#setwd("~/Linear Regression")
setwd("C:/Users/anandrathi/Documents/ramiyampersonal/Personal/Upgrad/Course_03/MultipleLinearRegression//")

#-------------------------------------------------------------------------------------------

# Load the dataset 

carPrice <- read.csv("carPrice.csv")

#-------------------------------------------------------------------------------------------

# Create the dummy variables

# For carCompany
dummy_1 <- data.frame(model.matrix( ~carCompany, data = carPrice))
dummy_1<-dummy_1[,-1]

#-------------------------------------------------------------------------------------------

# For carbody
dummy_2 <- data.frame(model.matrix( ~carbody, data = carPrice))
dummy_2<-dummy_2[,-1]

#-------------------------------------------------------------------------------------------

# Drivewheel 
dummy_3 <- data.frame(model.matrix( ~drivewheel, data = carPrice))
dummy_3<-dummy_3[,-1]

#-------------------------------------------------------------------------------------------

#Engine type
dummy_4 <- data.frame(model.matrix( ~enginetype, data = carPrice))
dummy_4<-dummy_4[,-1]

#-------------------------------------------------------------------------------------------

#cylindernumber
dummy_5 <- data.frame(model.matrix( ~cylindernumber, data = carPrice))
dummy_5<-dummy_5[,-1]

#-------------------------------------------------------------------------------------------

# Fuelsystem
dummy_6 <- data.frame(model.matrix( ~fuelsystem, data = carPrice))
dummy_6<-dummy_6[,-1]

#-------------------------------------------------------------------------------------------

# Variable having 2 levels.

# for fueltype
levels(carPrice$fueltype)<-c(1,0)
carPrice$fueltype<- as.numeric(levels(carPrice$fueltype))[carPrice$fueltype]

#-------------------------------------------------------------------------------------------

# for aspiration
levels(carPrice$aspiration)<-c(1,0)
carPrice$aspiration <- as.numeric(levels(carPrice$aspiration))[carPrice$aspiration]

#-------------------------------------------------------------------------------------------

# For doornumber
levels(carPrice$doornumber)<-c(1,0)
carPrice$doornumber<- as.numeric(levels(carPrice$doornumber))[carPrice$doornumber]

#-------------------------------------------------------------------------------------------

# Enginelocation
levels(carPrice$enginelocation)<-c(1,0)
carPrice$enginelocation<- as.numeric(levels(carPrice$enginelocation))[carPrice$enginelocation]

#-------------------------------------------------------------------------------------------

# Combine the dummy variables and the numeric columns of carPrice dataset.

carPrice_1 <- cbind(carPrice[ , c(1,2,4:6,9:14,17,19:26)], dummy_1,dummy_2,dummy_3,dummy_4,dummy_5,dummy_6)

#-------------------------------------------------------------------------------------------

# View the new dataset carPrice_1

View(carPrice_1)

#-------------------------------------------------------------------------------------------

# Divide you data in 70:30 

set.seed(101)
indices= sample(1:nrow(carPrice_1), 0.7*nrow(carPrice_1))

train=carPrice_1[indices,]
test = carPrice_1[-indices,]

#-------------------------------------------------------------------------------------------