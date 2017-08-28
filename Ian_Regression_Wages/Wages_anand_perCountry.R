#############################################################
#### Model per country 
# for IAN , syntagium
#  by anandrathi.dev@gmail.com
require(xlsx)
require(ggplot2)
require(car)
require(MASS)

set.seed(123)

# Load the data 
wages_data <-read.xlsx("C:/Users/rb117/Documents/personal/IAN/TimeseriesData.xlsx", sheetName = "Sheet2")

# Understand data
str(wages_data)

# Good data . no na's 
summary(wages_data)

# Explore the data .. some plotting
p <- ggplot(data=wages_data, aes(x=TIME, y=Wages, group=LOCATION, colour =LOCATION)) +   geom_line()+   geom_point()
p + geom_text(data = wages_data[wages_data$LOCATION == "AUS",  ], aes(label = LOCATION), hjust = 0.7, vjust = 1)

# create dummy for LOCATION
dummyFy <- function(inDF) {
  xtrain_dummy <- as.data.frame( model.matrix( ~. , inDF ))
  xtrain_dummy <- subset(xtrain_dummy, select=-`(Intercept)`)
  return(xtrain_dummy)
}

# Split data in train & test 
SamplTrainTest <- function(x, perc = 0.7) {
  smp_size <- floor(perc * nrow(x))
  ## set the seed to make your partition reproductible
  train_ind <- sample(seq_len(nrow(x)), size = smp_size)
  
  train <- x[train_ind, ]
  test <- x[-train_ind, ]
  return (list(train,test))
}


table(wages_data$LOCATION)

X <- split(wages_data, wages_data$LOCATION)
str(X)
Y <- lapply(seq_along(X), function(x) subset(as.data.frame(X[[x]]), select=-c(LOCATION) ) )
#Assign the dataframes in the list Y to individual objects
AUS <- Y[[1]]
AUT <- Y[[2]]
CAN <- Y[[3]]
DEU <- Y[[4]]
JPN <- Y[[5]]
KOR <- Y[[6]]
MEX <- Y[[7]]
NZL <- Y[[8]]
USA <- Y[[9]]

CountryList <- list( "AUS"= AUS, "AUT"=AUT , "CAN"=CAN, "DEU"=DEU, "JPN"=JPN, "KOR"=KOR, "MEX"=MEX, "NZL"=NZL, "USA"=USA)

corrplot::corrplot(cor(wages_data))
lapply( seq_along(CountryList), function(x) corrplot::corrplot(cor(CountryList[[x]]))  ) 

createModels  <- function(wages_x, name) {
  # First Model 
  #wagesModel  <- lm(Wages ~  . + sin(.), wages_x)
  print( name )
  wagesModel  <- lm(Wages ~  TIME+ CPI + GDP + TT + Unemp +  WorkingPop + sin(TIME), wages_x)

  fit <- auto.arima(residuals(wagesModel),stepwise=FALSE, approximation=FALSE) 
  print(summary(fit))
  acf(residuals(wagesModel), main=name) 
  #acf(residuals(arima(residuals(wagesModel),order = c(2,0,2)) ))
  #acf(residuals(arima(residuals(wagesModel),order = c(2,0,2)) ), type="partial")
  acf(residuals(wagesModel), type="partial", main=name)

  #print(summary(wagesModel))
  #wagesModel2 <- stepAIC(wagesModel, trace = FALSE)
  #print(summary(wagesModel2))
  return(wagesModel)
}

modelList <- lapply( seq_along(CountryList), function(x) createModels(CountryList[[x]], names(CountryList[x])))  

modelList[[1]]



