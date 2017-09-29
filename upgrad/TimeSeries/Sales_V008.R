setwd("C:\\Users\\Vishesh Sakshi\\Documents\\Upgrad assignments\\Course 4\\Retail-Giant Sales Forecasting group study")
#####################################################################################################
# PGDDA Course-4 
# Session 1 Group Assignment 1: Retail-Giant Sales Forecasting
#----------------------------------------------------------------------------------------------------
#
# The goal of this assignment:
# ==================================================================================================
# You are required to develop a time series predictive model that will
#  forecast the sales and the demand for the next 6 months, 
#  that would help to manage the revenue and inventory accordingly.
#####################################################################################################
#
#--Setup environment-----------------------------------------------------------
# install.packages("ggplot2")
# install.packages("corrplot")
# install.packages("forecast")
# install.packages("lubridate")
# install.packages("raster")
# install.packages("zoo")
# install.packages("tseries")
# install.packages("caret")

library(ggplot2)
library(corrplot)
library(forecast)
library(lubridate)
library(raster)
library(zoo)
library(tseries)
library(caret)
library(stats)
library(MASS)

###############################################################################
#--Checkpoint 1: (Data Understanding & Data Preparation)
###############################################################################
#-"Global Mart" is an online store super giant having worldwide operations. 
# It takes orders and delivers across the globe and deals with 
# all the major product categories - consumer, corporate & home office.
#-The store caters to 7 different market segments and in 3 major categories. 
# You want to forecast at this granular level, 
# so you subset your data into 21 (7*3) buckets before analysing these data.
#-Not all of these 21 market buckets are important from the store's point of view. 
# So you need to find out 5 most profitable (and consistent) 
# segment from these 21 and forecast the sales and demand for these segments. 
###############################################################################
#--Checkpoint-1: Part 1 Make the subsets of the complete dataset using the 7 factor levels of "Market" and 3 factor levels of "Segment".
###############################################################################
#--Load the data--------------------------------------------------------------
sales_data <- read.csv("Global Superstore.csv", stringsAsFactors = T)

#--Let us have an initial look at the sales data
str(sales_data)
summary(sales_data)
nrow(sales_data)

#==============================================================================
#--There are 51290 rows/observations and 24 columns/attributes
#--There are 4 ID or PK columns
#--There are 3 columns with high intrensic values
#       - Customer Name
#       - Postal Code
#-- There are some continious data attributes 
#       - Tenure
#       - MonthlyCharges
#       - TotalCharges
#-- We have only NA or missing values for TotalCharges
#==============================================================================

#--Initial data cleanup--------------------------------------------------------
#-Before we begin splitting and preperation of data we will try to do an initial cleanup
# We do not need the ID cloumsn or the high intrensic data
sales_data_clean0 <- sales_data[,-c(1,2,6,7,12,15)]

#--LEt us now check for de-duplication
sum(duplicated(sales_data_clean0))

#--LEt us now check for NA
sum(is.na(sales_data_clean0)) 

sales_data_clean0$Order.Date <- as.Date(sales_data_clean0$Order.Date, "%d-%m-%Y")
sales_data_clean0$Ship.Date <- as.Date(sales_data_clean0$Ship.Date, "%d-%m-%Y")

Order.Date.YrMonth <- format(sales_data_clean0$Order.Date, "%Y-%m")
Ship.Date.YrMonth <- format(sales_data_clean0$Ship.Date, "%Y-%m")

sales_data_clean0 <- cbind(sales_data_clean0, Order.Date.YrMonth, Ship.Date.YrMonth)

#--Let us have an initial look at the sales data
str(sales_data_clean0)
summary(sales_data_clean0)
nrow(sales_data_clean0)

###############################################################################
#--Checkpoint-1: Part 1 Make the subsets of the complete dataset using the 7 factor levels of "Market" and 3 factor levels of "Segment".
###############################################################################
#-We have noticed that there are 4 main sets of data that are merged
#       - Product
#       - Order
#       - Sales
#       - Customer
#-Out of the above we have a time series data fro Sales and Order
# We will need to split the data and aggregate it to make it more meaningfull

#-Subsetting based on Market
summary(sales_data_clean0$Market)
str(sales_data_clean0)

#-Filter out the requitred columns
col.required <- c(1,19,4,8,13,14,16)

#-Subset by the Market
sales_market_Africa <- sales_data_clean0[sales_data_clean0$Market == "Africa", col.required]
sales_market_APAC <- sales_data_clean0[sales_data_clean0$Market == "APAC",col.required]
sales_market_Canada <- sales_data_clean0[sales_data_clean0$Market == "Canada",col.required]
sales_market_EMEA <- sales_data_clean0[sales_data_clean0$Market == "EMEA",col.required]
sales_market_EU <- sales_data_clean0[sales_data_clean0$Market == "EU",col.required]
sales_market_LATAM <- sales_data_clean0[sales_data_clean0$Market == "LATAM",col.required]
sales_market_US <- sales_data_clean0[sales_data_clean0$Market == "US",col.required]

#-Subsetting further on Segment
sales_Africa_Consumer <- sales_market_Africa[sales_market_Africa$Segment == "Consumer",]
sales_Africa_Corporate <- sales_market_Africa[sales_market_Africa$Segment == "Corporate",]
sales_Africa_HomeOffice <- sales_market_Africa[sales_market_Africa$Segment == "Home Office",]

sales_APAC_Consumer <- sales_market_APAC[sales_market_APAC$Segment == "Consumer",]
sales_APAC_Corporate <- sales_market_APAC[sales_market_APAC$Segment == "Corporate",]
sales_APAC_HomeOffice <- sales_market_APAC[sales_market_APAC$Segment == "Home Office",]

sales_Canada_Consumer <- sales_market_Canada[sales_market_Canada$Segment == "Consumer",]
sales_Canada_Corporate <- sales_market_Canada[sales_market_Canada$Segment == "Corporate",]
sales_Canada_HomeOffice <- sales_market_Canada[sales_market_Canada$Segment == "Home Office",]

sales_EMEA_Consumer <- sales_market_EMEA[sales_market_EMEA$Segment == "Consumer",]
sales_EMEA_Corporate <- sales_market_EMEA[sales_market_EMEA$Segment == "Corporate",]
sales_EMEA_HomeOffice <- sales_market_EMEA[sales_market_EMEA$Segment == "Home Office",]

sales_EU_Consumer <- sales_market_EU[sales_market_EU$Segment == "Consumer",]
sales_EU_Corporate <- sales_market_EU[sales_market_EU$Segment == "Corporate",]
sales_EU_HomeOffice <- sales_market_EU[sales_market_EU$Segment == "Home Office",]

sales_US_Consumer <- sales_market_US[sales_market_US$Segment == "Consumer",]
sales_US_Corporate <- sales_market_US[sales_market_US$Segment == "Corporate",]
sales_US_HomeOffice <- sales_market_US[sales_market_US$Segment == "Home Office",]

sales_LATAM_Consumer <- sales_market_LATAM[sales_market_LATAM$Segment == "Consumer",]
sales_LATAM_Corporate <- sales_market_LATAM[sales_market_LATAM$Segment == "Corporate",]
sales_LATAM_HomeOffice <- sales_market_LATAM[sales_market_LATAM$Segment == "Home Office",]


#--Checkpoint-1: Part 2 Aggregate the transaction level data to the month level time series using the "Sales", "Quantity" and the "Profit".
#--Checkpoint-1: Part 4 For each time series that you are forecasting, separate out last 6 months data for out of sample testing.

#-We have 21 sets of data that we will now aggregate in terms of monthly Sales,Quantity and Profit
func.Create_ts <- function(x){
  Sales.x.YrMonth <- aggregate(x$Sales, by=list(x$Order.Date.YrMonth), FUN = sum)
  Quantity.x.YrMonth <- aggregate(x$Quantity, by=list(x$Order.Date.YrMonth), FUN = sum)
  Profit.x.YrMonth <- aggregate(x$Profit, by=list(x$Order.Date.YrMonth), FUN = sum)
  
  x.ts <- cbind(Sales.x.YrMonth, Quantity.x.YrMonth$x,
                Profit.x.YrMonth$x)
  colnames(x.ts) <- c("Year.Month", "Sales","Quantity","Profit")
  
  return(x.ts)
}

#------------------------------------------------------------------------------
#--Africa
#------------------------------------------------------------------------------
#--Consumer Segment
Africa.Consumer.ts <- func.Create_ts(sales_Africa_Consumer)
Africa.Consumer.ts.train <- Africa.Consumer.ts[1:42,]
Africa.Consumer.ts.test <- Africa.Consumer.ts[43:48,]

Africa.Consumer.cv <- cv(Africa.Consumer.ts$Profit)
Africa.Consumer.cv
#--Corporate Segment
Africa.Corporate.ts <- func.Create_ts(sales_Africa_Corporate)
Africa.Corporate.ts.train <- Africa.Corporate.ts[1:42,]
Africa.Corporate.ts.test <- Africa.Corporate.ts[43:48,]

Africa.Corporate.cv <- cv(Africa.Corporate.ts$Profit)
Africa.Corporate.cv
#--Home Office Segment
Africa.HomeOffice.ts <- func.Create_ts(sales_Africa_HomeOffice)
Africa.HomeOffice.ts.train <- Africa.HomeOffice.ts[1:42,]
Africa.HomeOffice.ts.test <- Africa.HomeOffice.ts[43:48,]

Africa.HomeOffice.cv <- cv(Africa.HomeOffice.ts$Profit)
Africa.HomeOffice.cv

Africa.cv <- c(Africa.Consumer.cv,Africa.Corporate.cv,Africa.HomeOffice.cv)
#------------------------------------------------------------------------------
#--APAC
#------------------------------------------------------------------------------
#--Consumer Segment
APAC.Consumer.ts <- func.Create_ts(sales_APAC_Consumer)
APAC.Consumer.ts.train <- APAC.Consumer.ts[1:42,]
APAC.Consumer.ts.test <- APAC.Consumer.ts[43:48,]

APAC.Consumer.cv <- cv(APAC.Consumer.ts$Profit)
APAC.Consumer.cv
#--Corporate Segment
APAC.Corporate.ts <- func.Create_ts(sales_APAC_Corporate)
APAC.Corporate.ts.train <- APAC.Corporate.ts[1:42,]
APAC.Corporate.ts.test <- APAC.Corporate.ts[43:48,]

APAC.Corporate.cv <- cv(APAC.Corporate.ts$Profit)
APAC.Corporate.cv
#--Home Office Segment
APAC.HomeOffice.ts <- func.Create_ts(sales_APAC_HomeOffice)
APAC.HomeOffice.ts.train <- APAC.HomeOffice.ts[1:42,]
APAC.HomeOffice.ts.test <- APAC.HomeOffice.ts[43:48,]

APAC.HomeOffice.cv <- cv(APAC.HomeOffice.ts$Profit)
APAC.HomeOffice.cv

APAC.cv <- c(APAC.Consumer.cv,APAC.Corporate.cv,APAC.HomeOffice.cv)
#------------------------------------------------------------------------------
#--Canada
#------------------------------------------------------------------------------
#--Consumer Segment
Canada.Consumer.ts <- func.Create_ts(sales_Canada_Consumer)
Canada.Consumer.ts.train <- Canada.Consumer.ts[1:42,]
Canada.Consumer.ts.test <- Canada.Consumer.ts[43:48,]


Canada.Consumer.cv <- cv(Canada.Consumer.ts$Profit)
Canada.Consumer.cv
#--Corporate Segment
Canada.Corporate.ts <- func.Create_ts(sales_Canada_Corporate)
Canada.Corporate.ts.train <- Canada.Corporate.ts[1:42,]
Canada.Corporate.ts.test <- Canada.Corporate.ts[43:48,]

Canada.Corporate.cv <- cv(Canada.Corporate.ts$Profit)
Canada.Corporate.cv
#--Home Office Segment
Canada.HomeOffice.ts <- func.Create_ts(sales_Canada_HomeOffice)
Canada.HomeOffice.ts.train <- Canada.HomeOffice.ts[1:42,]
Canada.HomeOffice.ts.test <- Canada.HomeOffice.ts[43:48,]

Canada.HomeOffice.cv <- cv(Canada.HomeOffice.ts$Profit)
Canada.HomeOffice.cv

Canada.cv <- c(Canada.Consumer.cv,Canada.Corporate.cv,Canada.HomeOffice.cv)
#------------------------------------------------------------------------------
#--EMEA
#------------------------------------------------------------------------------
#--Consumer Segment
EMEA.Consumer.ts <- func.Create_ts(sales_EMEA_Consumer)
EMEA.Consumer.ts.train <- EMEA.Consumer.ts[1:42,]
EMEA.Consumer.ts.test <- EMEA.Consumer.ts[43:48,]

EMEA.Consumer.cv <- cv(EMEA.Consumer.ts$Profit)
EMEA.Consumer.cv
#--Corporate Segment
EMEA.Corporate.ts <- func.Create_ts(sales_EMEA_Corporate)
EMEA.Corporate.ts.train <- EMEA.Corporate.ts[1:42,]
EMEA.Corporate.ts.test <- EMEA.Corporate.ts[43:48,]

EMEA.Corporate.cv <- cv(EMEA.Corporate.ts$Profit)
EMEA.Corporate.cv
#--Home Office Segment
EMEA.HomeOffice.ts <- func.Create_ts(sales_EMEA_HomeOffice)
EMEA.HomeOffice.ts.train <- EMEA.HomeOffice.ts[1:42,]
EMEA.HomeOffice.ts.test <- EMEA.HomeOffice.ts[43:48,]

EMEA.HomeOffice.cv <- cv(EMEA.HomeOffice.ts$Profit)
EMEA.HomeOffice.cv

EMEA.cv <- c(EMEA.Consumer.cv,EMEA.Corporate.cv,EMEA.HomeOffice.cv)
#------------------------------------------------------------------------------
#--EU
#------------------------------------------------------------------------------
#--Consumer Segment
EU.Consumer.ts <- func.Create_ts(sales_EU_Consumer)
EU.Consumer.ts.train <- EU.Consumer.ts[1:42,]
EU.Consumer.ts.test <- EU.Consumer.ts[43:48,]

EU.Consumer.cv <- cv(EU.Consumer.ts$Profit)
EU.Consumer.cv
#--Corporate Segment
EU.Corporate.ts <- func.Create_ts(sales_EU_Corporate)
EU.Corporate.ts.train <- EU.Corporate.ts[1:42,]
EU.Corporate.ts.test <- EU.Corporate.ts[43:48,]

EU.Corporate.cv <- cv(EU.Corporate.ts$Profit)
EU.Corporate.cv
#--Home Office Segment
EU.HomeOffice.ts <- func.Create_ts(sales_EU_HomeOffice)
EU.HomeOffice.ts.train <- EU.HomeOffice.ts[1:42,]
EU.HomeOffice.ts.test <- EU.HomeOffice.ts[43:48,]

EU.HomeOffice.cv <- cv(EU.HomeOffice.ts$Profit)
EU.HomeOffice.cv

EU.cv <- c(EU.Consumer.cv,EU.Corporate.cv,EU.HomeOffice.cv)
#------------------------------------------------------------------------------
#--LATAM
#------------------------------------------------------------------------------
#--Consumer Segment
LATAM.Consumer.ts <- func.Create_ts(sales_LATAM_Consumer)
LATAM.Consumer.ts.train <- LATAM.Consumer.ts[1:42,]
LATAM.Consumer.ts.test <- LATAM.Consumer.ts[43:48,]

LATAM.Consumer.cv <- cv(LATAM.Consumer.ts$Profit)
LATAM.Consumer.cv
#--Corporate Segment
LATAM.Corporate.ts <- func.Create_ts(sales_LATAM_Corporate)
LATAM.Corporate.ts.train <- LATAM.Corporate.ts[1:42,]
LATAM.Corporate.ts.test <- LATAM.Corporate.ts[43:48,]

LATAM.Corporate.cv <- cv(LATAM.Corporate.ts$Profit)
LATAM.Corporate.cv
#--Home Office Segment
LATAM.HomeOffice.ts <- func.Create_ts(sales_LATAM_HomeOffice)
LATAM.HomeOffice.ts.train <- LATAM.HomeOffice.ts[1:42,]
LATAM.HomeOffice.ts.test <- LATAM.HomeOffice.ts[43:48,]

LATAM.HomeOffice.cv <- cv(LATAM.HomeOffice.ts$Profit)
LATAM.HomeOffice.cv

LATAM.cv <- c(LATAM.Consumer.cv,LATAM.Corporate.cv,LATAM.HomeOffice.cv)
#------------------------------------------------------------------------------
#--US
#------------------------------------------------------------------------------
#--Consumer Segment
US.Consumer.ts <- func.Create_ts(sales_US_Consumer)
US.Consumer.ts.train <- US.Consumer.ts[1:42,]
US.Consumer.ts.test <- US.Consumer.ts[43:48,]

US.Consumer.cv <- cv(US.Consumer.ts$Profit)
US.Consumer.cv
#--Corporate Segment
US.Corporate.ts <- func.Create_ts(sales_US_Corporate)
US.Corporate.ts.train <- US.Corporate.ts[1:42,]
US.Corporate.ts.test <- US.Corporate.ts[43:48,]

US.Corporate.cv <- cv(US.Corporate.ts$Profit)
US.Corporate.cv
#--Home Office Segment
US.HomeOffice.ts <- func.Create_ts(sales_US_HomeOffice)
US.HomeOffice.ts.train <- US.HomeOffice.ts[1:42,]
US.HomeOffice.ts.test <- US.HomeOffice.ts[43:48,]

US.HomeOffice.cv <- cv(US.HomeOffice.ts$Profit)
US.HomeOffice.cv

US.cv <- c(US.Consumer.cv,US.Corporate.cv,US.HomeOffice.cv)

#--Checkpoint-1: Part 3 For each of the 21 segments, find the profitability index. Find the top 5 profitable segment. (you can use Coefficient of Variation as the measure for this)


#------------------------------------------------------------------------------
Sales.cv <- rbind(Africa.cv,APAC.cv,Canada.cv,EMEA.cv,EU.cv,LATAM.cv,US.cv)
colnames(Sales.cv) <- c("Consumer.cv","Corporate.cv","HomeOffice.cv")

Sales.cv

#==============================================================================
#--We will now choose the lowest 5 CV------------------------------------------
#  1.    EU + Consumer = 62.43052
#  2.  APAC + Consumer = 63.21323
#  3. LATAM + Consumer = 66.14828
#  4. APAC + Corporate = 69.80869
#  5.   EU + Corporate = 76.38072
#==============================================================================
#--Outlier Test for data ------------------------------------------------------
boxplot.stats(EU.Consumer.ts.train$Sales) #--no
boxplot.stats(EU.Consumer.ts.train$Quantity) #--no

boxplot.stats(APAC.Consumer.ts.train$Sales) #--no
boxplot.stats(APAC.Consumer.ts.train$Quantity) #--no

boxplot.stats(LATAM.Consumer.ts.train$Sales)
boxplot.stats(LATAM.Consumer.ts.train$Quantity)

boxplot.stats(APAC.Corporate.ts.train$Sales)
boxplot.stats(APAC.Corporate.ts.train$Quantity)

boxplot.stats(EU.Corporate.ts.train$Sales)
boxplot.stats(EU.Corporate.ts.train$Quantity)
###############################################################################
#--Checkpoint-4: Model Building
###############################################################################
#--Aim to forecast the sales and quantity for the next 6 months.
#------------------------------------------------------------------------------
#--Convert all required models to time series
EU.Consumer.ts.train$Year.Month <- as.yearmon(EU.Consumer.ts.train$Year.Month)
EU.Consumer.ts.train <- as.ts(read.zoo(EU.Consumer.ts.train))
#-
EU.Consumer.ts.test$Year.Month <- as.yearmon(EU.Consumer.ts.test$Year.Month)
EU.Consumer.ts.test <- as.ts(read.zoo(EU.Consumer.ts.test))

APAC.Consumer.ts.train$Year.Month <- as.yearmon(APAC.Consumer.ts.train$Year.Month)
APAC.Consumer.ts.train <- as.ts(read.zoo(APAC.Consumer.ts.train))
#-
APAC.Consumer.ts.test$Year.Month <- as.yearmon(APAC.Consumer.ts.test$Year.Month)
APAC.Consumer.ts.test <- as.ts(read.zoo(APAC.Consumer.ts.test))

LATAM.Consumer.ts.train$Year.Month <- as.yearmon(LATAM.Consumer.ts.train$Year.Month)
LATAM.Consumer.ts.train <- as.ts(read.zoo(LATAM.Consumer.ts.train))
#-
LATAM.Consumer.ts.test$Year.Month <- as.yearmon(LATAM.Consumer.ts.test$Year.Month)
LATAM.Consumer.ts.test <- as.ts(read.zoo(LATAM.Consumer.ts.test))

APAC.Corporate.ts.train$Year.Month <- as.yearmon(APAC.Corporate.ts.train$Year.Month)
APAC.Corporate.ts.train <- as.ts(read.zoo(APAC.Corporate.ts.train))
#-
APAC.Corporate.ts.test$Year.Month <- as.yearmon(APAC.Corporate.ts.test$Year.Month)
APAC.Corporate.ts.test <- as.ts(read.zoo(APAC.Corporate.ts.test))

EU.Corporate.ts.train$Year.Month <- as.yearmon(EU.Corporate.ts.train$Year.Month)
EU.Corporate.ts.train <- as.ts(read.zoo(EU.Corporate.ts.train))
#-
EU.Corporate.ts.test$Year.Month <- as.yearmon(EU.Corporate.ts.test$Year.Month)
EU.Corporate.ts.test <- as.ts(read.zoo(EU.Corporate.ts.test))

#--LEt us validate our data sets
class(EU.Consumer.ts.train)
class(APAC.Consumer.ts.train)
class(LATAM.Consumer.ts.train)
class(APAC.Corporate.ts.train)
class(EU.Corporate.ts.train)

#-- Checkpoint 2: (Time series modelling) Part:1 Plot and smoothen the time series.

#==============================================================================
#--We see that the datasets are
#     - "mts" : monthly time series
#     - "ts"  : time series dataset
#     - matrix: as we have sales/quantity/profit
#-Note: We will not use profit but continue with Sales/Quantity
#==============================================================================

#--Plot the time series for sales/quantity.
#--Sales + Quantity
par("mar")
par(mar=c(2,2,2,2))
par(mfrow=c(5,2))
#--
plot(EU.Consumer.ts.train[,2],main="Quantity EU Consumer Market", ylab = "Quantity")
abline(reg=lm(EU.Consumer.ts.train[,2]~time(EU.Consumer.ts.train)),col = "red")
plot(EU.Consumer.ts.train[,3],main="Sales EU Consumer Market", ylab = "Sales")
abline(reg=lm(EU.Consumer.ts.train[,3]~time(EU.Consumer.ts.train)),col = "red")

plot(APAC.Consumer.ts.train[,2],main="Quantity APAC Consumer Market", ylab = "Quantity")
abline(reg=lm(APAC.Consumer.ts.train[,2]~time(APAC.Consumer.ts.train)),col = "red")
plot(APAC.Consumer.ts.train[,3],main="Sales APAC Consumer Market", ylab = "Sales")
abline(reg=lm(APAC.Consumer.ts.train[,3]~time(APAC.Consumer.ts.train)),col = "red")

plot(LATAM.Consumer.ts.train[,2],main="Quantity LATAM Consumer Market", ylab = "Quantity")
abline(reg=lm(LATAM.Consumer.ts.train[,2]~time(LATAM.Consumer.ts.train)),col = "red")
plot(LATAM.Consumer.ts.train[,3],main="Sales LATAM Consumer Market", ylab = "Sales")
abline(reg=lm(LATAM.Consumer.ts.train[,3]~time(LATAM.Consumer.ts.train)),col = "red")


plot(APAC.Corporate.ts.train[,2],main="Quantity APAC Corporate Market", ylab = "Quantity")
abline(reg=lm(APAC.Corporate.ts.train[,2]~time(APAC.Corporate.ts.train)),col = "red")
plot(APAC.Corporate.ts.train[,3],main="Sales APAC Corporate Market", ylab = "Sales")
abline(reg=lm(APAC.Corporate.ts.train[,3]~time(APAC.Corporate.ts.train)),col = "red")


plot(EU.Corporate.ts.train[,2],main="Quantity EU Corporate Market", ylab = "Quantity")
abline(reg=lm(EU.Corporate.ts.train[,2]~time(EU.Corporate.ts.train)),col = "red")
plot(EU.Corporate.ts.train[,3],main="Sales EU Corporate Market", ylab = "Sales")
abline(reg=lm(EU.Corporate.ts.train[,3]~time(EU.Corporate.ts.train)),col = "red")

#------------------------------------------------------------------------------
par(mfrow = c(1,1))
plot(aggregate(EU.Consumer.ts.train,FUN=mean))
boxplot(EU.Consumer.ts.train~cycle(EU.Consumer.ts.train))

plot(aggregate(APAC.Consumer.ts.train,FUN=mean))
boxplot(APAC.Consumer.ts.train~cycle(APAC.Consumer.ts.train))

plot(aggregate(LATAM.Consumer.ts.train,FUN=mean))
boxplot(LATAM.Consumer.ts.train~cycle(LATAM.Consumer.ts.train))

plot(aggregate(APAC.Corporate.ts.train,FUN=mean))
boxplot(APAC.Corporate.ts.train~cycle(APAC.Corporate.ts.train))

plot(aggregate(EU.Corporate.ts.train,FUN=mean))
boxplot(EU.Corporate.ts.train~cycle(EU.Corporate.ts.train))

###############################################################################
plotForecastErrors <- function(forecasterrors)
{
  # make a histogram of the forecast errors:
  mybinsize <- IQR(forecasterrors)/4
  mysd   <- sd(forecasterrors)
  mymin  <- min(forecasterrors) - mysd*5
  mymax  <- max(forecasterrors) + mysd*3
  # generate normally distributed data with mean 0 and standard deviation mysd
  mynorm <- rnorm(10000, mean=0, sd=mysd)
  mymin2 <- min(mynorm)
  mymax2 <- max(mynorm)
  if (mymin2 < mymin) { mymin <- mymin2 }
  if (mymax2 > mymax) { mymax <- mymax2 }
  # make a red histogram of the forecast errors, with the normally distributed data overlaid:
  mybins <- seq(mymin, mymax, mybinsize)
  hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
  # freq=FALSE ensures the area under the histogram = 1
  # generate normally distributed data with mean 0 and standard deviation mysd
  myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
  # plot the normal curve as a blue line on top of the histogram of forecast errors:
  points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
}

#--
tryArma <- function(delta, guessp, guessq, timeseries) {
  df <- data.frame()
  # generate all possible ARMA models
  for (p in max(0,(guessp-delta)):(guessp+delta)) {
    for (q in max(0,(guessq-delta)):(guessq+delta)) {
      order <- c(p,0,q)
      # Fit a maximum likelihood ARMA(p,q) model
      armafit <- Arima(timeseries, order=order, method="ML")
      # Add the results to the dataframe
      df <- rbind(df, c(p, q, armafit$loglik, armafit$aic,
                        armafit$aicc, armafit$bic))
    }
  }
  names(df) <- c('p',"q","log.likelihood", "AIC", "AICc", "BIC")
  return(df)
}

#-- Checkpoint 2: Use feature engineering to come up with the best regression fit. 
#--Creates a manually derived autoregression model
#  And returns the auto fitted arima model back
chkRegFit <- function(ts_ds, plotlabel="") {
  tsdf <- as.data.frame(ts_ds)
  tsdf$date <- as.yearmon(time(ts_ds))
  
  
  lmfit <- lm(x ~ sin(25*(time(date))) * poly(time(date),2,raw=TRUE) + 
                cos(25*(time(date))) * poly(time(date),2,raw=TRUE) + 
                poly(time(date),2,raw=TRUE) + time(date) , data=tsdf)
  
  trend <- predict(lmfit, newdata = tsdf)
  
  
  tsdf$trend <- trend
  tsdf$diff <- tsdf$x - trend
  
  print(ggplot(data=tsdf, aes(x=date, y = x)) + 
    geom_line() + 
    geom_line(aes(y=trend, col="red") ) +
    ggtitle(plotlabel))
  
  ts_dsf <- as.ts(read.zoo(tsdf[,c(2,4)]))
  final_arima <- auto.arima(ts_dsf)
  return(final_arima)
}
###############################################################################
#==============================================================================
#-From the above diagrams we can clearly observe the slight increase in sales/Quantity 
# with time clearly signifying a time series component that needs to be removed
#-homoscedasticity - There is also observed elements of variance with time
#-In the following graphs,we also notice the spread becomes closer as the time increases.
#--Most of the non stationary elements are mild in nature but are still evident
#==============================================================================
#############################################################################################
#--Sales--
#############################################################################################
#--EU.Consumer.Sales-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component
#--EU Consumer Quantity-

decompose(EU.Consumer.ts.train[,1])
plot(decompose(EU.Consumer.ts.train[,1]))

#--Using feature engineering to come up with the best regression fit
EU.Consumer.man.arima <- chkRegFit(EU.Consumer.ts.train[,1], "EU.Consumer.Sales")
#--Display the ARIMA model from the above
EU.Consumer.man.arima

#--Checkpoint 2: Part:4 Find the optimal value of p,d,q for ARIMA modelling 

#--Fnd the optimal value of p,d,q for ARIMA modelling
par(mfrow = c(1,3))
plot(EU.Consumer.ts.train[,1])
plot(diff(EU.Consumer.ts.train[,1], differences = 1))
plot(diff(log(EU.Consumer.ts.train[,1]), differences = 1))

EU.Consumer.ts.train.stat <- diff(log(EU.Consumer.ts.train[,1]), differences = 1)
# EU.Consumer.ts.train.stat <- filter(EU.Consumer.ts.train.stat, filter=rep(1/width, width), # -- Filter smoothing
#                                     method="convolution", sides=2, circular = T) 

plot(decompose(EU.Consumer.ts.train.stat))

adf.test(EU.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(EU.Consumer.ts.train.stat)

par(mfrow = c(1,2))
acf(EU.Consumer.ts.train.stat)
pacf(EU.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(EU.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,EU.Consumer.ts.train.stat)
df

auto.arima(EU.Consumer.ts.train.stat)

EU.Consumer.Sales.arima <- arima(EU.Consumer.ts.train.stat, order = c(1,1,3)) # fit an ARIMA(0,1,1) model
EU.Consumer.Sales.arima

EU.Consumer.Sales.forcast <- forecast.Arima(EU.Consumer.Sales.arima, h=6, level=c(99))
EU.Consumer.Sales.forcast_12 <- forecast.Arima(EU.Consumer.Sales.arima, h=12, level=c(99))

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(EU.Consumer.Sales.forcast, main = "EU.Consumer.Sales.forcast")

summary(EU.Consumer.Sales.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(EU.Consumer.Sales.forcast$residuals, lag.max=20)
Box.test(EU.Consumer.Sales.forcast$residuals, lag=20, type="Ljung-Box")


par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(EU.Consumer.Sales.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(EU.Consumer.Sales.forcast$residuals) # make a histogram
qqnorm(EU.Consumer.Sales.forcast$residuals)
abline(0,1, col="red")

EU.Consumer.ts.test[,1]
fitted(Arima(EU.Consumer.ts.test[,1], model = EU.Consumer.Sales.arima))

#-- Checkpoint 3: (Model Evaluation)

#--MAPE
mape <- mean(((EU.Consumer.ts.test[,1] - fitted(Arima(EU.Consumer.ts.test[,1], model = EU.Consumer.Sales.arima)))/EU.Consumer.ts.test[,1]) *100)
abs(mape)

#==============================================================================
#-MAPE: 6.788415
#-p-value [Box-Ljung test] = 0.13
#==============================================================================


#--APAC.Consumer.Sales-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(APAC.Consumer.ts.train[,1])
plot(decompose(APAC.Consumer.ts.train[,1]))

#--Using feature engineering to come up with the best regression fit
APAC.Consumer.man.arima <- chkRegFit(APAC.Consumer.ts.train[,1], "APAC.Consumer.Sales")
#--Display the ARIMA model from the above
APAC.Consumer.man.arima

par(mfrow = c(1,2))
plot(APAC.Consumer.ts.train[,1])
plot(diff(log(APAC.Consumer.ts.train[,1]), differences = 1))

APAC.Consumer.ts.train.stat <- diff(log(APAC.Consumer.ts.train[,1]), differences = 2)


adf.test(APAC.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(APAC.Consumer.ts.train.stat)

plot(decompose(APAC.Consumer.ts.train.stat))
par(mfrow = c(1,2))
acf(APAC.Consumer.ts.train.stat)
pacf(APAC.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(APAC.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,APAC.Consumer.ts.train.stat)
df

auto.arima(APAC.Consumer.ts.train.stat)

APAC.Consumer.Sales.arima <- arima(APAC.Consumer.ts.train.stat, order = c(2,2,3)) # fit an ARIMA(0,1,1) model
APAC.Consumer.Sales.arima

APAC.Consumer.Sales.forcast <- forecast.Arima(APAC.Consumer.Sales.arima, h=6)
APAC.Consumer.Sales.forcast_12 <- forecast.Arima(APAC.Consumer.Sales.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(APAC.Consumer.Sales.forcast, main="APAC.Consumer.Sales.forcast")

summary(APAC.Consumer.Sales.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(APAC.Consumer.Sales.forcast$residuals, lag.max=20)
Box.test(APAC.Consumer.Sales.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(APAC.Consumer.Sales.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(APAC.Consumer.Sales.forcast$residuals) # make a histogram
qqnorm(APAC.Consumer.Sales.forcast$residuals)
abline(0,1, col="red")

APAC.Consumer.ts.test[,1]
fitted(Arima(APAC.Consumer.ts.test[,1], model = APAC.Consumer.Sales.arima))

#--MAPE
mape <- mean(((APAC.Consumer.ts.test[,1] - fitted(Arima(APAC.Consumer.ts.test[,1], model = APAC.Consumer.Sales.arima)))/APAC.Consumer.ts.test[,1]) *100)
abs(mape)

#==============================================================================
#-MAPE: 4.228637
#-p-value [Box-Ljung test] = 0.172
#==============================================================================


#--LATAM.Consumer.Sales-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(LATAM.Consumer.ts.train[,1])
plot(decompose(LATAM.Consumer.ts.train[,1]))

#--Using feature engineering to come up with the best regression fit
LATAM.Consumer.man.arima <- chkRegFit(LATAM.Consumer.ts.train[,1],"LATAM.Consumer.Sales")
#--Display the ARIMA model from the above
LATAM.Consumer.man.arima

par(mfrow = c(1,2))
plot(LATAM.Consumer.ts.train[,1])
plot(diff(log(LATAM.Consumer.ts.train[,1]), differences = 1))

LATAM.Consumer.ts.train.stat <- diff(log(LATAM.Consumer.ts.train[,1]), differences = 2)

adf.test(LATAM.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(LATAM.Consumer.ts.train.stat)

plot(decompose(LATAM.Consumer.ts.train.stat))
par(mfrow = c(1,2))
acf(LATAM.Consumer.ts.train.stat)
pacf(LATAM.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(LATAM.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,LATAM.Consumer.ts.train.stat)
df

auto.arima(LATAM.Consumer.ts.train.stat)

LATAM.Consumer.Sales.arima <- arima(LATAM.Consumer.ts.train.stat, order = c(3,2,0)) # fit an ARIMA(0,1,1) model
LATAM.Consumer.Sales.arima

LATAM.Consumer.Sales.forcast <- forecast.Arima(LATAM.Consumer.Sales.arima, h=6)
LATAM.Consumer.Sales.forcast_12 <- forecast.Arima(LATAM.Consumer.Sales.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(LATAM.Consumer.Sales.forcast, main="LATAM.Consumer.Sales.forcast")

summary(LATAM.Consumer.Sales.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(LATAM.Consumer.Sales.forcast$residuals, lag.max=20)
Box.test(LATAM.Consumer.Sales.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(LATAM.Consumer.Sales.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(LATAM.Consumer.Sales.forcast$residuals) # make a histogram
qqnorm(LATAM.Consumer.Sales.forcast$residuals)
abline(0,1, col="red")

LATAM.Consumer.ts.test[,1]
fitted(Arima(LATAM.Consumer.ts.test[,1], model = LATAM.Consumer.Sales.arima))

#--MAPE
mape <- mean(((LATAM.Consumer.ts.test[,1] - fitted(Arima(LATAM.Consumer.ts.test[,1], model = LATAM.Consumer.Sales.arima)))/LATAM.Consumer.ts.test[,1]) *100)
abs(mape)
#==============================================================================
#-MAPE: 9.010047
#-p-value [Box-Ljung test] = 0.06643
#==============================================================================
#--APAC.Corporate.Sales-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(APAC.Corporate.ts.train[,1])
plot(decompose(APAC.Corporate.ts.train[,1]))

#--Using feature engineering to come up with the best regression fit
APAC.Corporate.man.arima <- chkRegFit(APAC.Corporate.ts.train[,1],"APAC.Corporate.Sales")
#--Display the ARIMA model from the above
APAC.Corporate.man.arima

par(mfrow = c(1,2))
plot(APAC.Corporate.ts.train[,1])
plot(diff(log(APAC.Corporate.ts.train[,1]), differences = 1))

APAC.Corporate.ts.train.stat <- diff(log(APAC.Corporate.ts.train[,1]), differences = 2)


adf.test(APAC.Corporate.ts.train.stat, alternative="stationary", k=0)
kpss.test(APAC.Corporate.ts.train.stat)

plot(decompose(APAC.Corporate.ts.train.stat))
par(mfrow = c(1,2))
acf(APAC.Corporate.ts.train.stat)
pacf(APAC.Corporate.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(APAC.Corporate.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,APAC.Corporate.ts.train.stat)
df

auto.arima(APAC.Corporate.ts.train.stat)

APAC.Corporate.Sales.arima <- arima(APAC.Corporate.ts.train.stat, order = c(2,2,3)) # fit an ARIMA(0,1,1) model
APAC.Corporate.Sales.arima

APAC.Corporate.Sales.forcast <- forecast.Arima(APAC.Corporate.Sales.arima, h=6)
APAC.Corporate.Sales.forcast_12 <- forecast.Arima(APAC.Corporate.Sales.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(APAC.Corporate.Sales.forcast, main="APAC.Corporate.Sales.forcast")

summary(APAC.Corporate.Sales.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(APAC.Corporate.Sales.forcast$residuals, lag.max=20)
Box.test(APAC.Corporate.Sales.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(APAC.Corporate.Sales.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(APAC.Corporate.Sales.forcast$residuals) # make a histogram
qqnorm(APAC.Corporate.Sales.forcast$residuals)
abline(0,1, col="red")

APAC.Corporate.ts.test[,1]
fitted(Arima(APAC.Corporate.ts.test[,1], model = APAC.Corporate.Sales.arima))

mape <- mean(((APAC.Corporate.ts.test[,1] - fitted(Arima(APAC.Corporate.ts.test[,1], model = APAC.Corporate.Sales.arima)))/APAC.Corporate.ts.test[,1]) *100)
abs(mape)
#==============================================================================
#-MAPE: 5.640266
#-p-value [Box-Ljung test] = 0.07388
#==============================================================================
#--EU.Corporate.Sales-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(EU.Corporate.ts.train[,1])
plot(decompose(EU.Corporate.ts.train[,1]))

#--Using feature engineering to come up with the best regression fit
EU.Corporate.man.arima <- chkRegFit(EU.Corporate.ts.train[,1],"EU.Corporate.Sales")
#--Display the ARIMA model from the above
EU.Corporate.man.arima

par(mfrow = c(1,2))
plot(EU.Corporate.ts.train[,1])
plot(diff(log(EU.Corporate.ts.train[,1]), differences = 1))

EU.Corporate.ts.train.stat <- diff(log(EU.Corporate.ts.train[,1]), differences = 2)


adf.test(EU.Corporate.ts.train.stat, alternative="stationary", k=0)
kpss.test(EU.Corporate.ts.train.stat)

plot(decompose(EU.Corporate.ts.train.stat))
par(mfrow = c(1,2))
acf(EU.Corporate.ts.train.stat)
pacf(EU.Corporate.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(EU.Corporate.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,EU.Corporate.ts.train.stat)
df

auto.arima(EU.Corporate.ts.train.stat)

EU.Corporate.Sales.arima <- arima(EU.Corporate.ts.train.stat, order = c(3,1,3)) # fit an ARIMA(0,1,1) model
EU.Corporate.Sales.arima

EU.Corporate.Sales.forcast <- forecast.Arima(EU.Corporate.Sales.arima, h=6)
EU.Corporate.Sales.forcast_12 <- forecast.Arima(EU.Corporate.Sales.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(EU.Corporate.Sales.forcast, main="EU.Corporate.Sales.forcast")

summary(APAC.Corporate.Sales.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(EU.Corporate.Sales.forcast$residuals, lag.max=20)
Box.test(EU.Corporate.Sales.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(EU.Corporate.Sales.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(EU.Corporate.Sales.forcast$residuals) # make a histogram
qqnorm(EU.Corporate.Sales.forcast$residuals)
abline(0,1, col="red")

EU.Corporate.ts.test[,1]
fitted(Arima(EU.Corporate.ts.test[,1], model = EU.Corporate.Sales.arima))
mape <- mean(((EU.Corporate.ts.test[,1] - fitted(Arima(EU.Corporate.ts.test[,1], model = EU.Corporate.Sales.arima)))/EU.Corporate.ts.test[,1]) *100)
abs(mape)
#==============================================================================
#-MAPE: 5.392183
#-p-value [Box-Ljung test] = 0.8367
#==============================================================================
#############################################################################################
#--Quantity--
#############################################################################################
#--EU.Consumer.Quantity-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component
#--EU Consumer Quantity-

decompose(EU.Consumer.ts.train[,2])
plot(decompose(EU.Consumer.ts.train[,2]))

#--Using feature engineering to come up with the best regression fit
EU.Consumer.man.arima <- chkRegFit(EU.Consumer.ts.train[,1],"EU.Consumer.Quantity")
#--Display the ARIMA model from the above
EU.Consumer.man.arima


par(mfrow = c(1,2))
plot(EU.Consumer.ts.train[,2])
plot(diff(log(EU.Consumer.ts.train[,2]), differences = 1))

EU.Consumer.ts.train.stat <- diff(log(EU.Consumer.ts.train[,2]), differences = 1)

adf.test(EU.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(EU.Consumer.ts.train.stat)

par(mfrow = c(1,2))
acf(EU.Consumer.ts.train.stat)
pacf(EU.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(EU.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,EU.Consumer.ts.train.stat)
df

auto.arima(EU.Consumer.ts.train.stat)

EU.Consumer.Quantity.arima <- arima(EU.Consumer.ts.train.stat, order = c(2,2,3)) # fit an ARIMA(0,1,1) model
EU.Consumer.Quantity.arima

EU.Consumer.Quantity.forcast <- forecast.Arima(EU.Consumer.Quantity.arima, h=6)
EU.Consumer.Quantity.forcast_12 <- forecast.Arima(EU.Consumer.Quantity.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(EU.Consumer.Quantity.forcast, main="EU.Consumer.Quantity.forcast")

summary(EU.Consumer.Quantity.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(EU.Consumer.Quantity.forcast$residuals, lag.max=20)
Box.test(EU.Consumer.Quantity.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(EU.Consumer.Quantity.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(EU.Consumer.Quantity.forcast$residuals) # make a histogram
qqnorm(EU.Consumer.Quantity.forcast$residuals)
abline(0,1, col="red")

EU.Consumer.ts.test[,2]
fitted(Arima(EU.Consumer.ts.test[,2], model = EU.Consumer.Quantity.arima))

mape <- mean(((EU.Consumer.ts.test[,2] - fitted(Arima(EU.Consumer.ts.test[,2], model = EU.Consumer.Quantity.arima)))/EU.Consumer.ts.test[,2]) *100)
abs(mape)
#==============================================================================
#-MAPE: 5.832784
#-p-value [Box-Ljung test] = 0.2066
#==============================================================================
#--APAC.Consumer.Quantity-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(APAC.Consumer.ts.train[,2])
plot(decompose(APAC.Consumer.ts.train[,2]))

#--Using feature engineering to come up with the best regression fit
APAC.Consumer.man.arima <- chkRegFit(APAC.Consumer.ts.train[,1],"APAC.Consumer.Quantity")
#--Display the ARIMA model from the above
APAC.Consumer.man.arima

par(mfrow = c(1,2))
plot(APAC.Consumer.ts.train[,2])
plot(diff(log(APAC.Consumer.ts.train[,2]), differences = 1))

APAC.Consumer.ts.train.stat <- diff(log(APAC.Consumer.ts.train[,2]), differences = 2)


adf.test(APAC.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(APAC.Consumer.ts.train.stat)

plot(decompose(APAC.Consumer.ts.train.stat))
par(mfrow = c(1,2))
acf(APAC.Consumer.ts.train.stat)
pacf(APAC.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(APAC.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,APAC.Consumer.ts.train.stat)
df

auto.arima(APAC.Consumer.ts.train.stat)

APAC.Consumer.Quantity.arima <- arima(APAC.Consumer.ts.train.stat, order = c(2,2,5)) # fit an ARIMA(0,1,1) model
APAC.Consumer.Quantity.arima

APAC.Consumer.Quantity.forcast <- forecast.Arima(APAC.Consumer.Quantity.arima, h=6)
APAC.Consumer.Quantity.forcast_12 <- forecast.Arima(APAC.Consumer.Quantity.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(APAC.Consumer.Quantity.forcast, main="APAC.Consumer.Quantity.forcast")

summary(APAC.Consumer.Quantity.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(APAC.Consumer.Quantity.forcast$residuals, lag.max=20)
Box.test(APAC.Consumer.Quantity.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(APAC.Consumer.Quantity.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(APAC.Consumer.Quantity.forcast$residuals) # make a histogram
qqnorm(APAC.Consumer.Quantity.forcast$residuals)
abline(0,1, col="red")

APAC.Consumer.ts.test[,2]
fitted(Arima(APAC.Consumer.ts.test[,2], model = APAC.Consumer.Quantity.arima))

mape <- mean(((APAC.Consumer.ts.test[,2] - fitted(Arima(APAC.Consumer.ts.test[,2], model = APAC.Consumer.Quantity.arima)))/APAC.Consumer.ts.test[,2]) *100)
abs(mape)
#==============================================================================
#-MAPE: 6.898358
#-p-value [Box-Ljung test] = 0.3505
#==============================================================================

#--LATAM.Consumer.Quantity-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(LATAM.Consumer.ts.train[,2])
plot(decompose(LATAM.Consumer.ts.train[,2]))

#--Using feature engineering to come up with the best regression fit
LATAM.Consumer.man.arima <- chkRegFit(LATAM.Consumer.ts.train[,1],"LATAM.Consumer.Quantity")
#--Display the ARIMA model from the above
LATAM.Consumer.man.arima

par(mfrow = c(1,2))
plot(LATAM.Consumer.ts.train[,2])
plot(diff(log(LATAM.Consumer.ts.train[,2]), differences = 1))

LATAM.Consumer.ts.train.stat <- diff(log(LATAM.Consumer.ts.train[,2]), differences = 2)


adf.test(LATAM.Consumer.ts.train.stat, alternative="stationary", k=0)
kpss.test(LATAM.Consumer.ts.train.stat)

plot(decompose(LATAM.Consumer.ts.train.stat))
par(mfrow = c(1,2))
acf(LATAM.Consumer.ts.train.stat)
pacf(LATAM.Consumer.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(LATAM.Consumer.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,LATAM.Consumer.ts.train.stat)
df

auto.arima(LATAM.Consumer.ts.train.stat)

LATAM.Consumer.Quantity.arima <- arima(LATAM.Consumer.ts.train.stat, order = c(2,2,7)) # fit an ARIMA(0,1,1) model
LATAM.Consumer.Quantity.arima

LATAM.Consumer.Quantity.forcast <- forecast.Arima(LATAM.Consumer.Quantity.arima, h=6)
LATAM.Consumer.Quantity.forcast_12 <- forecast.Arima(LATAM.Consumer.Quantity.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(LATAM.Consumer.Quantity.forcast, main="LATAM.Consumer.Quantity.forcast")

summary(LATAM.Consumer.Quantity.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(LATAM.Consumer.Quantity.forcast$residuals, lag.max=20)
Box.test(LATAM.Consumer.Quantity.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(LATAM.Consumer.Quantity.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(LATAM.Consumer.Quantity.forcast$residuals) # make a histogram
qqnorm(LATAM.Consumer.Quantity.forcast$residuals)
abline(0,1, col="red")

LATAM.Consumer.ts.test[,2]
fitted(Arima(LATAM.Consumer.ts.test[,2], model = LATAM.Consumer.Quantity.arima))

mape <- mean(((LATAM.Consumer.ts.test[,2] - fitted(Arima(LATAM.Consumer.ts.test[,2], model = LATAM.Consumer.Quantity.arima)))/LATAM.Consumer.ts.test[,2]) *100)
abs(mape)
#==============================================================================
#-MAPE: 6.714756
#-p-value [Box-Ljung test] = 0.1622
#==============================================================================
#--APAC.Corporate.Quantity-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(APAC.Corporate.ts.train[,2])
plot(decompose(APAC.Corporate.ts.train[,2]))

#--Using feature engineering to come up with the best regression fit
APAC.Corporate.man.arima <- chkRegFit(APAC.Corporate.ts.train[,1],"APAC.Corporate.Quantity")
#--Display the ARIMA model from the above
APAC.Corporate.man.arima

par(mfrow = c(1,2))
plot(APAC.Corporate.ts.train[,2])
plot(diff(log(APAC.Corporate.ts.train[,2]), differences = 1))

APAC.Corporate.ts.train.stat <- diff(log(APAC.Corporate.ts.train[,2]), differences = 2)


adf.test(APAC.Corporate.ts.train.stat, alternative="stationary", k=0)
kpss.test(APAC.Corporate.ts.train.stat)

plot(decompose(APAC.Corporate.ts.train.stat))
par(mfrow = c(1,2))
acf(APAC.Corporate.ts.train.stat)
pacf(APAC.Corporate.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(APAC.Corporate.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,APAC.Corporate.ts.train.stat)
df

auto.arima(APAC.Corporate.ts.train.stat)

APAC.Corporate.Quantity.arima <- arima(APAC.Corporate.ts.train.stat, order = c(3,2,3)) # fit an ARIMA(0,1,1) model
APAC.Corporate.Quantity.arima

APAC.Corporate.Quantity.forcast <- forecast.Arima(APAC.Corporate.Quantity.arima, h=6)
APAC.Corporate.Quantity.forcast_12 <- forecast.Arima(APAC.Corporate.Quantity.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(APAC.Corporate.Quantity.forcast, main = "APAC.Corporate.Quantity.forcast" )

summary(APAC.Corporate.Quantity.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(APAC.Corporate.Quantity.forcast$residuals, lag.max=20)
Box.test(APAC.Corporate.Quantity.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(APAC.Corporate.Quantity.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(APAC.Corporate.Quantity.forcast$residuals) # make a histogram
qqnorm(APAC.Corporate.Quantity.forcast$residuals)
abline(0,1, col="red")

APAC.Corporate.ts.test[,2]
fitted(Arima(APAC.Corporate.ts.test[,2], model = APAC.Corporate.Quantity.arima))

mape <- mean(((APAC.Corporate.ts.test[,2] - fitted(Arima(APAC.Corporate.ts.test[,2], model = APAC.Corporate.Quantity.arima)))/APAC.Corporate.ts.test[,2]) *100)
abs(mape)
#==============================================================================
#-MAPE: 2.978279
#-p-value [Box-Ljung test] =  0.2496
#==============================================================================


#--EU.Corporate.Quantity-----------------------------------------------------------------
#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

decompose(EU.Corporate.ts.train[,2])
plot(decompose(EU.Corporate.ts.train[,2]))

#--Using feature engineering to come up with the best regression fit
EU.Corporate.man.arima <- chkRegFit(EU.Corporate.ts.train[,1],"EU.Corporate.Quantity")
#--Display the ARIMA model from the above
EU.Corporate.man.arima

par(mfrow = c(1,2))
plot(EU.Corporate.ts.train[,2])
plot(diff(log(EU.Corporate.ts.train[,2]), differences = 1))

EU.Corporate.ts.train.stat <- diff(log(EU.Corporate.ts.train[,2]), differences = 2)


adf.test(EU.Corporate.ts.train.stat, alternative="stationary", k=0)
kpss.test(EU.Corporate.ts.train.stat)

plot(decompose(EU.Corporate.ts.train.stat))
par(mfrow = c(1,2))
acf(EU.Corporate.ts.train.stat)
pacf(EU.Corporate.ts.train.stat)

#-Just to validate the model choosen 
auto.arima(EU.Corporate.ts.train.stat, ic="bic")

#==============================================================================
#-The ACF lag value from the graph can be either 0 or 1
#-The PACF lag value from the graph is 0
#-The d value as determined from above is 1
#-Model is ARMA(0,1,0) or ARMA(1,1,0)
#-Therefore, using the principle of parsimony we choose the model with lesser parmaters
#------------- ARMA(0,1,0) ---------------------------------------------------
#==============================================================================
df <- tryArma(1,0,0,EU.Corporate.ts.train.stat)
df

auto.arima(EU.Corporate.ts.train.stat)

EU.Corporate.Quantity.arima <- arima(EU.Corporate.ts.train.stat, order = c(2,1,3)) # fit an ARIMA(0,1,1) model
EU.Corporate.Quantity.arima

EU.Corporate.Quantity.forcast <- forecast.Arima(EU.Corporate.Quantity.arima, h=6)
EU.Corporate.Quantity.forcast_12 <- forecast.Arima(EU.Corporate.Quantity.arima, h=12)

par("mar")
par(mar=c(0,5.1,1,2.1))
par(mfrow = c(1,1))
plot.forecast(EU.Corporate.Quantity.forcast, main = "EU.Corporate.Quantity.forcast")

summary(EU.Corporate.Quantity.forcast)
# As in the case of exponential smoothing models, it is a good idea to investigate 
# whether the forecast errors of an ARIMA model are normally distributed with mean zero 
# and constant variance, and whether the are correlations between successive forecast errors.
acf(EU.Corporate.Quantity.forcast$residuals, lag.max=20)
Box.test(EU.Corporate.Quantity.forcast$residuals, lag=20, type="Ljung-Box")

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,3))
plot.ts(EU.Corporate.Quantity.forcast$residuals)            # make time plot of forecast errors
plotForecastErrors(EU.Corporate.Quantity.forcast$residuals) # make a histogram
qqnorm(EU.Corporate.Quantity.forcast$residuals)
abline(0,1, col="red")

EU.Corporate.ts.test[,2]
fitted(Arima(EU.Corporate.ts.test[,2], model = EU.Corporate.Quantity.arima))

mape <- mean(((EU.Corporate.ts.test[,2] - fitted(Arima(EU.Corporate.ts.test[,2], model = EU.Corporate.Quantity.arima)))/EU.Corporate.ts.test[,2]) *100)
abs(mape)
#==============================================================================
#-MAPE: 0.6372419
#-p-value [Box-Ljung test] =  0.9323
#==============================================================================

#-- Checkpoint 4: (Result Interpretation) 

#-RESULT-Sales-------------------------------------------------------------------
#YrMon <- time(EU.Consumer.ts.test)

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,5))

resDF <- as.data.frame(cbind(YrMon=time(EU.Consumer.ts.test),
              EU.Consumer.ts.test,
              APAC.Consumer.ts.test,
              LATAM.Consumer.ts.test,
              APAC.Corporate.ts.test,
              EU.Corporate.ts.test))


#-- Checkpoint 4: 
#-- Based on the 10 forecasts that you have got, comment on the major source of revenue for the store over the next 6 months.

ggplot(resDF, aes(x = YrMon)) + 
  geom_line(aes(y = EU.Consumer.ts.test.Sales , colour="EU.Consumer"  ))  +
  geom_line(aes(y = APAC.Consumer.ts.test.Sales, colour="APAC.Consumer" )) + 
  geom_line(aes(y = APAC.Corporate.ts.test.Sales, colour="APAC.Corporate" )) + 
  geom_line(aes(y = EU.Corporate.ts.test.Sales, colour="EU.Corporate" )) + 
  geom_line(aes(y = LATAM.Consumer.ts.test.Sales , colour="LATAM.Consumer" )) + 
  ylab(label="Sales/Region") + 
  xlab("Sales/Months")

#-- Checkpoint 4:
#-- comment on the resource allocation that the store should do, based on demand forecast. 
#-- Based on the 10 forecasts that you have got, comment on the major source of revenue for the store over the next 6 months.

#-RESULT-Qty-------------------------------------------------------------------

ggplot(resDF, aes(x = YrMon)) + 
  geom_line(aes(y = EU.Consumer.ts.test.Quantity , colour="EU.Consumer"  ))  +
  geom_line(aes(y = APAC.Consumer.ts.test.Quantity, colour="APAC.Consumer")) + 
  geom_line(aes(y = APAC.Corporate.ts.test.Quantity, colour="APAC.Corporate")) + 
  geom_line(aes(y = EU.Corporate.ts.test.Quantity, colour="EU.Corporate" )) + 
  geom_line(aes(y = LATAM.Consumer.ts.test.Quantity , colour="LATAM.Consumer")) + 
  ylab(label="Quantity/Region ") + 
  xlab("Quantity/Months") +  scale_colour_manual(values=c("red","green","blue", "black", "purple"))


#==============================================================================
#--We will now choose the lowest 5 CV------------------------------------------
#  1.    EU + Consumer = 62.43052
#  2.  APAC + Consumer = 63.21323
#  3. LATAM + Consumer = 66.14828
#  4. APAC + Corporate = 69.80869
#  5.   EU + Corporate = 76.38072
#=================================

par("mar")
par(mar=c(3,5.1,2,2.1))
par(mfrow = c(1,5))

#YrMon_12=time(EU.Consumer.ts.train)


EU.Consumer.Sales.forcast_12_df <-  as.data.frame(EU.Consumer.Sales.forcast_12)
EU.Consumer.Quantity.forcast_12_df <-  as.data.frame(EU.Consumer.Quantity.forcast_12)
EU.Corporate.Sales.forcast_12_df <-  as.data.frame(EU.Corporate.Sales.forcast_12)
EU.Corporate.Quantity.forcast_12_df <-  as.data.frame(EU.Corporate.Quantity.forcast_12)
APAC.Consumer.Sales.forcast_12_df <-  as.data.frame(APAC.Consumer.Sales.forcast_12)
APAC.Consumer.Quantity.forcast_12_df <-  as.data.frame(APAC.Consumer.Quantity.forcast_12)
APAC.Corporate.Sales.forcast_12_df <-  as.data.frame(APAC.Corporate.Sales.forcast_12)
APAC.Corporate.Quantity.forcast_12_df <-  as.data.frame(APAC.Corporate.Quantity.forcast_12)
LATAM.Consumer.Sales.forcast_12_df <-  as.data.frame(LATAM.Consumer.Sales.forcast_12)
LATAM.Consumer.Quantity.forcast_12_df <-  as.data.frame(LATAM.Consumer.Quantity.forcast_12)


resDFWhole <- as.data.frame(cbind(YrMon=rownames(EU.Consumer.Sales.forcast_12_df),
                             EU.Consumer.Sales = as.double(EU.Consumer.Sales.forcast_12_df$`Point Forecast`), 
                             EU.Consumer.Quantity = as.double(EU.Consumer.Quantity.forcast_12_df$`Point Forecast`),
                             EU.Corporate.Sales = as.double(EU.Corporate.Sales.forcast_12_df$`Point Forecast`),
                             EU.Corporate.Quantity = as.double(EU.Corporate.Quantity.forcast_12_df$`Point Forecast`),
                             APAC.Consumer.Sales = as.double(APAC.Consumer.Sales.forcast_12_df$`Point Forecast`), 
                             APAC.Consumer.Quantity = as.double(APAC.Consumer.Quantity.forcast_12_df$`Point Forecast`),
                             APAC.Corporate.Sales = as.double(APAC.Corporate.Sales.forcast_12_df$`Point Forecast`), 
                             APAC.Corporate.Quantity = as.double(APAC.Corporate.Quantity.forcast_12_df$`Point Forecast`),
                             LATAM.Consumer.Sales = as.double(LATAM.Consumer.Sales.forcast_12_df$`Point Forecast`),
                             LATAM.Consumer.Quantity = as.double(LATAM.Consumer.Quantity.forcast_12_df$`Point Forecast`)) 
                            
                            )


resDF <- resDFWhole[which(as.yearmon(resDFWhole$YrMon) > as.yearmon('Dec 2014') ), ]
resDF$EU.Consumer.Sales <-  as.double(resDF$EU.Consumer.Sales)
resDF$EU.Consumer.Quantity <-  as.double(resDF$EU.Consumer.Quantity)
resDF$EU.Corporate.Sales <-  as.double(resDF$EU.Corporate.Sales)
resDF$EU.Corporate.Quantity <-  as.double(resDF$EU.Corporate.Quantity)
resDF$APAC.Consumer.Sales <-  as.double(resDF$APAC.Consumer.Sales)
resDF$APAC.Consumer.Quantity <-  as.double(resDF$APAC.Consumer.Quantity)
resDF$APAC.Corporate.Sales <-  as.double(resDF$APAC.Corporate.Sales)
resDF$APAC.Corporate.Quantity <-  as.double(resDF$APAC.Corporate.Quantity)
resDF$LATAM.Consumer.Sales <-  as.double(resDF$LATAM.Consumer.Sales)
resDF$LATAM.Consumer.Quantity <-  as.double(resDF$LATAM.Consumer.Quantity)



ggplot(resDF, aes(x = YrMon)) + 
  geom_line(aes(y = EU.Consumer.Sales , colour="EU.Consumer Sales"  ))  +
  geom_line(aes(y = APAC.Consumer.Sales, colour="APAC.Consumer" )) + 
  geom_line(aes(y = APAC.Corporate.Sales, colour="APAC.Corporate" )) + 
  geom_line(aes(y = EU.Corporate.Sales, colour="EU.Corporate" )) + 
  geom_line(aes(y = LATAM.Consumer.Sales , colour="LATAM.Consumer" )) + 
  ylab(label="Sales/Region") + 
  xlab("Sales/Months")   


#-- Checkpoint 4:
#-- comment on the resource allocation that the store should do, based on demand forecast. 
#-- Based on the 10 forecasts that you have got, comment on the major source of revenue for the store over the next 6 months.

#-RESULT-Qty-------------------------------------------------------------------
ggplot(resDF, aes(x = YrMon)) + 
  geom_line(aes(y = EU.Consumer.Quantity , colour="EU.Consumer Qty"  ))  +
  geom_line(aes(y = APAC.Consumer.Quantity, colour="APAC.Consumer" )) + 
  geom_line(aes(y = APAC.Corporate.Quantity, colour="APAC.Corporate" )) + 
  geom_line(aes(y = EU.Corporate.Quantity, colour="EU.Corporate" )) + 
  geom_line(aes(y = LATAM.Consumer.Quantity , colour="LATAM.Consumer" )) + 
  ylab(label="Sales/Region") + 
  xlab("Qty/Months")



#####################################################################################################
#------------------------------END-------------------------------------------------------------------
#####################################################################################################
