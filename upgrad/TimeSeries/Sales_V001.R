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

library(ggplot2)
library(corrplot)
library(forecast)
library(lubridate)
library(raster)
library(zoo)
library(tseries)
###############################################################################
#--Checkpoint-1:  (Data Understanding & Data Preparation) 
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
#--Checkpoint-1:Step 1: Data Understanding
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
#--Checkpoint-1:Step-2: Aggregate the transaction level data
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
Africa.Consumer.ts

Africa.Consumer.cv <- cv(Africa.Consumer.ts$Profit)
Africa.Consumer.cv
#--Corporate Segment
Africa.Corporate.ts <- func.Create_ts(sales_Africa_Corporate)
Africa.Corporate.ts

Africa.Corporate.cv <- cv(Africa.Corporate.ts$Profit)
Africa.Corporate.cv
#--Home Office Segment
Africa.HomeOffice.ts <- func.Create_ts(sales_Africa_HomeOffice)
Africa.HomeOffice.ts

Africa.HomeOffice.cv <- cv(Africa.HomeOffice.ts$Profit)
Africa.HomeOffice.cv

Africa.cv <- c(Africa.Consumer.cv,Africa.Corporate.cv,Africa.HomeOffice.cv)
#------------------------------------------------------------------------------
#--APAC
#------------------------------------------------------------------------------
#--Consumer Segment
APAC.Consumer.ts <- func.Create_ts(sales_APAC_Consumer)
APAC.Consumer.ts

APAC.Consumer.cv <- cv(APAC.Consumer.ts$Profit)
APAC.Consumer.cv
#--Corporate Segment
APAC.Corporate.ts <- func.Create_ts(sales_APAC_Corporate)
APAC.Corporate.ts

APAC.Corporate.cv <- cv(APAC.Corporate.ts$Profit)
APAC.Corporate.cv
#--Home Office Segment
APAC.HomeOffice.ts <- func.Create_ts(sales_APAC_HomeOffice)
APAC.HomeOffice.ts

APAC.HomeOffice.cv <- cv(APAC.HomeOffice.ts$Profit)
APAC.HomeOffice.cv

APAC.cv <- c(APAC.Consumer.cv,APAC.Corporate.cv,APAC.HomeOffice.cv)
#------------------------------------------------------------------------------
#--Canada
#------------------------------------------------------------------------------
#--Consumer Segment
Canada.Consumer.ts <- func.Create_ts(sales_Canada_Consumer)
Canada.Consumer.ts

Canada.Consumer.cv <- cv(Canada.Consumer.ts$Profit)
Canada.Consumer.cv
#--Corporate Segment
Canada.Corporate.ts <- func.Create_ts(sales_Canada_Corporate)
Canada.Corporate.ts

Canada.Corporate.cv <- cv(Canada.Corporate.ts$Profit)
Canada.Corporate.cv
#--Home Office Segment
Canada.HomeOffice.ts <- func.Create_ts(sales_Canada_HomeOffice)
Canada.HomeOffice.ts

Canada.HomeOffice.cv <- cv(Canada.HomeOffice.ts$Profit)
Canada.HomeOffice.cv

Canada.cv <- c(Canada.Consumer.cv,Canada.Corporate.cv,Canada.HomeOffice.cv)
#------------------------------------------------------------------------------
#--EMEA
#------------------------------------------------------------------------------
#--Consumer Segment
EMEA.Consumer.ts <- func.Create_ts(sales_EMEA_Consumer)
EMEA.Consumer.ts

EMEA.Consumer.cv <- cv(EMEA.Consumer.ts$Profit)
EMEA.Consumer.cv
#--Corporate Segment
EMEA.Corporate.ts <- func.Create_ts(sales_EMEA_Corporate)
EMEA.Corporate.ts

EMEA.Corporate.cv <- cv(EMEA.Corporate.ts$Profit)
EMEA.Corporate.cv
#--Home Office Segment
EMEA.HomeOffice.ts <- func.Create_ts(sales_EMEA_HomeOffice)
EMEA.HomeOffice.ts

EMEA.HomeOffice.cv <- cv(EMEA.HomeOffice.ts$Profit)
EMEA.HomeOffice.cv

EMEA.cv <- c(EMEA.Consumer.cv,EMEA.Corporate.cv,EMEA.HomeOffice.cv)
#------------------------------------------------------------------------------
#--EU
#------------------------------------------------------------------------------
#--Consumer Segment
EU.Consumer.ts <- func.Create_ts(sales_EU_Consumer)
EU.Consumer.ts

EU.Consumer.cv <- cv(EU.Consumer.ts$Profit)
EU.Consumer.cv
#--Corporate Segment
EU.Corporate.ts <- func.Create_ts(sales_EU_Corporate)
EU.Corporate.ts

EU.Corporate.cv <- cv(EU.Corporate.ts$Profit)
EU.Corporate.cv
#--Home Office Segment
EU.HomeOffice.ts <- func.Create_ts(sales_EU_HomeOffice)
EU.HomeOffice.ts

EU.HomeOffice.cv <- cv(EU.HomeOffice.ts$Profit)
EU.HomeOffice.cv

EU.cv <- c(EU.Consumer.cv,EU.Corporate.cv,EU.HomeOffice.cv)
#------------------------------------------------------------------------------
#--LATAM
#------------------------------------------------------------------------------
#--Consumer Segment
LATAM.Consumer.ts <- func.Create_ts(sales_LATAM_Consumer)
LATAM.Consumer.ts

LATAM.Consumer.cv <- cv(LATAM.Consumer.ts$Profit)
LATAM.Consumer.cv
#--Corporate Segment
LATAM.Corporate.ts <- func.Create_ts(sales_LATAM_Corporate)
LATAM.Corporate.ts

LATAM.Corporate.cv <- cv(LATAM.Corporate.ts$Profit)
LATAM.Corporate.cv
#--Home Office Segment
LATAM.HomeOffice.ts <- func.Create_ts(sales_LATAM_HomeOffice)
LATAM.HomeOffice.ts

LATAM.HomeOffice.cv <- cv(LATAM.HomeOffice.ts$Profit)
LATAM.HomeOffice.cv

LATAM.cv <- c(LATAM.Consumer.cv,LATAM.Corporate.cv,LATAM.HomeOffice.cv)
#------------------------------------------------------------------------------
#--US
#------------------------------------------------------------------------------
#--Consumer Segment
US.Consumer.ts <- func.Create_ts(sales_US_Consumer)
US.Consumer.ts

US.Consumer.cv <- cv(US.Consumer.ts$Profit)
US.Consumer.cv
#--Corporate Segment
US.Corporate.ts <- func.Create_ts(sales_US_Corporate)
US.Corporate.ts

US.Corporate.cv <- cv(US.Corporate.ts$Profit)
US.Corporate.cv
#--Home Office Segment
US.HomeOffice.ts <- func.Create_ts(sales_US_HomeOffice)
US.HomeOffice.ts

US.HomeOffice.cv <- cv(US.HomeOffice.ts$Profit)
US.HomeOffice.cv

US.cv <- c(US.Consumer.cv,US.Corporate.cv,US.HomeOffice.cv)
#------------------------------------------------------------------------------
Sales.cv <- rbind(Africa.cv,APAC.cv,Canada.cv,EMEA.cv,EU.cv,LATAM.cv,US.cv)
colnames(Sales.cv) <- c("Consumer.cv","Corporate.cv","HomeOffice.cv")
###############################################################################
#--Checkpoint-1:Step-3: Find the top 5 profitable segment. 
###############################################################################

Sales.cv
#==============================================================================
#--We will now choose the lowest 5 CV------------------------------------------
#  1.    EU + Consumer = 62.43052
#  2.  APAC + Consumer = 63.21323
#  3. LATAM + Consumer = 66.14828
#  4. APAC + Corporate = 69.80869
#  5.   EU + Corporate = 76.38072
#==============================================================================

###############################################################################
#--Checkpoint 2: (Time series modelling) 
###############################################################################
#--Aim to forecast the sales and quantity for the next 6 months.
#------------------------------------------------------------------------------
#--Convert all required models to time series
EU.Consumer.ts$Year.Month <- as.yearmon(EU.Consumer.ts$Year.Month)
EU.Consumer.ts <- as.ts(read.zoo(EU.Consumer.ts))

APAC.Consumer.ts$Year.Month <- as.yearmon(APAC.Consumer.ts$Year.Month)
APAC.Consumer.ts <- as.ts(read.zoo(APAC.Consumer.ts))

LATAM.Consumer.ts$Year.Month <- as.yearmon(LATAM.Consumer.ts$Year.Month)
LATAM.Consumer.ts <- as.ts(read.zoo(LATAM.Consumer.ts))

APAC.Corporate.ts$Year.Month <- as.yearmon(APAC.Corporate.ts$Year.Month)
APAC.Corporate.ts <- as.ts(read.zoo(APAC.Corporate.ts))

EU.Corporate.ts$Year.Month <- as.yearmon(EU.Corporate.ts$Year.Month)
EU.Corporate.ts <- as.ts(read.zoo(EU.Corporate.ts))

#--LEt us validate our data sets
class(EU.Consumer.ts)
class(APAC.Consumer.ts)
class(LATAM.Consumer.ts)
class(APAC.Corporate.ts)
class(EU.Corporate.ts)

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
plot(EU.Consumer.ts[,2],main="Quantity EU Consumer Market", ylab = "Quantity")
abline(reg=lm(EU.Consumer.ts[,2]~time(EU.Consumer.ts)),col = "red")
plot(EU.Consumer.ts[,3],main="Sales EU Consumer Market", ylab = "Sales")
abline(reg=lm(EU.Consumer.ts[,3]~time(EU.Consumer.ts)),col = "red")

plot(APAC.Consumer.ts[,2],main="Quantity APAC Consumer Market", ylab = "Quantity")
abline(reg=lm(APAC.Consumer.ts[,2]~time(APAC.Consumer.ts)),col = "red")
plot(APAC.Consumer.ts[,3],main="Sales APAC Consumer Market", ylab = "Sales")
abline(reg=lm(APAC.Consumer.ts[,3]~time(APAC.Consumer.ts)),col = "red")

plot(LATAM.Consumer.ts[,2],main="Quantity LATAM Consumer Market", ylab = "Quantity")
abline(reg=lm(LATAM.Consumer.ts[,2]~time(LATAM.Consumer.ts)),col = "red")
plot(LATAM.Consumer.ts[,3],main="Sales LATAM Consumer Market", ylab = "Sales")
abline(reg=lm(LATAM.Consumer.ts[,3]~time(LATAM.Consumer.ts)),col = "red")


plot(APAC.Corporate.ts[,2],main="Quantity APAC Corporate Market", ylab = "Quantity")
abline(reg=lm(APAC.Corporate.ts[,2]~time(APAC.Corporate.ts)),col = "red")
plot(APAC.Corporate.ts[,3],main="Sales APAC Corporate Market", ylab = "Sales")
abline(reg=lm(APAC.Corporate.ts[,3]~time(APAC.Corporate.ts)),col = "red")


plot(EU.Corporate.ts[,2],main="Quantity EU Corporate Market", ylab = "Quantity")
abline(reg=lm(EU.Corporate.ts[,2]~time(EU.Corporate.ts)),col = "red")
plot(EU.Corporate.ts[,3],main="Sales EU Corporate Market", ylab = "Sales")
abline(reg=lm(EU.Corporate.ts[,3]~time(EU.Corporate.ts)),col = "red")

#------------------------------------------------------------------------------
par(mfrow = c(1,1))
plot(aggregate(EU.Consumer.ts,FUN=mean))
boxplot(EU.Consumer.ts~cycle(EU.Consumer.ts))

plot(aggregate(APAC.Consumer.ts,FUN=mean))
boxplot(APAC.Consumer.ts~cycle(APAC.Consumer.ts))

plot(aggregate(LATAM.Consumer.ts,FUN=mean))
boxplot(LATAM.Consumer.ts~cycle(LATAM.Consumer.ts))

plot(aggregate(APAC.Corporate.ts,FUN=mean))
boxplot(APAC.Corporate.ts~cycle(APAC.Corporate.ts))

plot(aggregate(EU.Corporate.ts,FUN=mean))
boxplot(EU.Corporate.ts~cycle(EU.Corporate.ts))
#==============================================================================
#-From the above diagrams we can clearly observe the slight increase in sales/Quantity 
# with time clearly signifying a time series component that needs to be removed
#-homoscedasticity - There is also observed elements of variance with time
#-In the following graphs,we also notice the spread becomes closer as the time increases.
#--Most of the non stationary elements are mild in nature but are still evident
#==============================================================================

#--Smoothen the series using any of the smoothing techniques. 
#  This would help you identify the trend/seasonality component

#--EU Consumer --
#--EU Consumer Sales-
sales  <- EU.Consumer.ts[,1]
plot(decompose(sales))
width <- 2
par(mfrow = c(1,3))
plot(sales) # -- Original
plot(diff(log(sales), differences = 2)) # -- Log smoothing
smoothedseries <- filter(sales, filter=rep(1/width, width), # -- Filter smoothing
                         method="convolution", sides=2)  
plot(smoothedseries)

#--EU Consumer Qty-
qty  <- EU.Consumer.ts[,2]
plot(decompose(qty))
width <- 2
par(mfrow = c(1,3))
plot(qty) # -- Original
plot(diff(log(qty), differences = 2)) # -- Log smoothing
smoothedseries <- filter(qty, filter=rep(1/width, width), # -- Filter smoothing
                         method="convolution", sides=2)  
plot(smoothedseries)

#--EU Consumer Profit-
profit  <- EU.Consumer.ts[,2]
plot(decompose(profit))
width <- 2
par(mfrow = c(1,3))
plot(profit) # -- Original
plot(diff(log(profit), differences = 2)) # -- Log smoothing
smoothedseries <- filter(profit, filter=rep(1/width, width), # -- Filter smoothing
                         method="convolution", sides=2)  
plot(smoothedseries)

#--EU Consumer --
#--EU Consumer --
#--EU Consumer --
#--EU Consumer --
#--EU Consumer --
#--EU Consumer --


par(mfrow = c(1,2))
sales  <- EU.Consumer.ts[,1]
smoothedseries <- filter(sales, filter=rep(1/width, width), # -- Filter smoothing
                         method="convolution", sides=2)  

acf(diff(log(EU.Consumer.ts[,1]), differences = 1))
pacf(diff(log(EU.Consumer.ts[,1]), differences = 1))

auto.arima(diff(log(EU.Consumer.ts[,2]), differences = 1), ic="bic")

adf.test(diff(log(EU.Corporate.ts[,2])), alternative="stationary", k=0)

acf(log(EU.Corporate.ts[,1]))


EU.Corporate.ts.components <- decompose(EU.Corporate.ts)
EU.Corporate.ts.components$seasonal
plot(EU.Corporate.ts.components)

