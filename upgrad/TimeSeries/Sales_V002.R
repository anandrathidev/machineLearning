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
  #MonthSeq <- seq.Date(min(x$Order.Date), max(x$Order.Date), by="month")
  #x.ts <- cbind( MonthSeq, Sales.x.YrMonth, Quantity.x.YrMonth$x,  Profit.x.YrMonth$x )
  x.ts <- cbind( Sales.x.YrMonth, Quantity.x.YrMonth$x,  Profit.x.YrMonth$x )
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

chkRegFit <- function(ts_ds, plotlabel="") {
  tsdf <- as.data.frame(ts_ds)
  tsdf$date <- time(ts_ds) 
  lmfit <- lm(Quantity ~ sin(25*(time(date))) * poly(time(date),2,raw=TRUE) + cos(25*(time(date))) * poly(time(date),2,raw=TRUE) + poly(time(date),2,raw=TRUE) + time(date) , data=tsdf)
  trend <- predict(lmfit, newdata = tsdf)
  tsdf$trend <- trend
  ggplot(data=tsdf, aes(x=date, y = Quantity)) + 
    geom_line() + 
    geom_line(aes(y=trend, col="red") ) +
    ggtitle(paste0("Quantity ",plotlabel)) 
  
  lmfit <- lm(Sales ~ sin(25*(time(date))) * poly(time(date),2,raw=TRUE) + cos(25*(time(date))) * poly(time(date),2,raw=TRUE) + poly(time(date),2,raw=TRUE) + time(date) , data=tsdf)
  trend <- predict(lmfit, newdata = tsdf)
  tsdf$trend <- trend
  ggplot(data=tsdf, aes(x=date, y = Sales)) + 
    geom_line() + 
    geom_line(aes(y=trend, col="red") ) +
    ggtitle(paste0("Sales ",plotlabel)) 
}

chkRegFit(EU.Consumer.ts, "EU.Consumer")
chkRegFit(APAC.Consumer.ts, "APAC.Consumer")
chkRegFit(LATAM.Consumer.ts, "APAC.Consumer")
chkRegFit(APAC.Corporate.ts, "APAC.Consumer")
chkRegFit(EU.Corporate.ts, "APAC.Consumer")



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


func.SmoothPlot = function(fts, plotlabel="Plot Label") {
  plot(decompose(fts) )
  width <- 2
  par(mfrow = c(1,3))
  plot(fts , main= plotlabel ) # -- Original
  plot(diff(log(fts), differences = 2) , main= paste0(plotlabel, " Log smoothing")) # -- Log smoothing
  smoothedseries <- filter(fts, filter=rep(1/width, width), # -- Filter smoothing
                         method="convolution", sides=2)  
  plot(smoothedseries , main= paste0(plotlabel, " smoothing"))
}

#--EU Consumer --
#--EU Consumer Sales-
func.SmoothPlot(fts = EU.Consumer.ts[,"Sales"], plotlabel="EU Consumer Sales ")

#--EU Consumer Qty-
qty  <- EU.Consumer.ts[,"Quantity"]
func.SmoothPlot(fts = qty, plotlabel="EU Consumer Qty ")

#--EU Consumer Profit-
profit  <- EU.Consumer.ts[,c("Profit")]
func.SmoothPlot(fts = profit, plotlabel="EU Consumer Profit ")

#--EU Corporate --
#--EU Corporate Sales-
sales  <- EU.Corporate.ts[,"Sales"]
func.SmoothPlot(fts = sales, plotlabel="EU Corporate Sales ")

#--EU Corporate Qty-
qty  <- EU.Corporate.ts[,"Quantity"]
func.SmoothPlot(fts = qty, plotlabel="EU Corporate Quantity ")

#--EU Consumer Profit-
profit  <- EU.Corporate.ts[,c("Profit")]
func.SmoothPlot(fts = profit, plotlabel="EU Corporate Profit ")

#....................

#--EU Consumer --
#--EU Consumer --
#--EU Consumer --
#--EU Consumer --
#--EU Consumer --


#--EU Consumer Sales-

require(foreign)
#--Check the residual series for White noise.
func.WhiteNoise = function(fts, plotlabel="") {
  w<-2
  smoothedseries <- ts(filter(fts, filter=rep(1/width, width), # -- Filter smoothing
                              method="convolution", sides=2) )
  
  diff3 <- smoothedseries[w+2,"Series 3"] - smoothedseries[w+1,"Series 3"]
  diff2 <- smoothedseries[w+2,"Series 2"] - smoothedseries[w+1,"Series 2"]
  diff1 <- smoothedseries[w+2,"Series 1"] - smoothedseries[w+1,"Series 1"]
  for (i in seq(w,1,-1)) {
    smoothedseries[i,"Series 1"] <- smoothedseries[i+1,"Series 1"] - diff1
    smoothedseries[i,"Series 2"] <- smoothedseries[i+1,"Series 2"] - diff2
    smoothedseries[i,"Series 3"] <- smoothedseries[i+1,"Series 3"] - diff3
  }
  n <- nrow(as.data.frame(fts))
  diff3 <- smoothedseries[n-w,"Series 3"] - smoothedseries[n-w-1,"Series 3"]
  diff2 <- smoothedseries[n-w,"Series 2"] - smoothedseries[n-w-1,"Series 2"]
  diff1 <- smoothedseries[n-w,"Series 1"] - smoothedseries[n-w-1,"Series 1"]
  for (i in seq(n-w+1, n)) {
    smoothedseries[i,"Series 1"] <- smoothedseries[i-1,"Series 1"] - diff1
    smoothedseries[i,"Series 2"] <- smoothedseries[i-1,"Series 2"] - diff2
    smoothedseries[i,"Series 3"] <- smoothedseries[i-1,"Series 3"] - diff3
  }
  
  Year.Month <- as.yearmon(as.matrix(time(smoothedseries)))
  Sales <- as.vector(smoothedseries[,'Series 1'])
  Quantity <- as.vector(smoothedseries[,'Series 2'])
  Profit <- as.vector(smoothedseries[,'Series 3'])
  smoothed.TS <- as.data.frame(as.table(as.yearmon(Year.Month),Sales,Quantity))
  smoothedTS <- as.ts(read.zoo(smoothed.TS))
  
  lmfitSales <- lm(smoothedseries[,"Sales"] ~ sin(25*(time(smoothedseries))) * poly(time(smoothedseries),2,raw=TRUE) + cos(25*(time(smoothedseries))) * poly(time(smoothedseries),2,raw=TRUE) + poly(time(smoothedseries),2,raw=TRUE) + time(smoothedseries) )
  lmfitQuantity <- lm(smoothedseries[,"Quantity"] ~ sin(25*(time(smoothedseries))) * poly(time(smoothedseries),2,raw=TRUE) + cos(25*(time(smoothedseries))) * poly(time(smoothedseries),2,raw=TRUE) + poly(time(smoothedseries),2,raw=TRUE) + time(smoothedseries) )
  smoothedseries
  acf(smoothedseries, main = paste0(plotlabel, " Before ACF Plot "))
  pacf(smoothedseries, main = paste0(plotlabel," Before PACF Plot"))
  
  trend <- predict(lmfit, data.frame(x=fts))
  onlynoise <- fts-trend
  onlynoise <- fts-trend
  acf(onlynoise, main = paste0(plotlabel, " onlynoise ACF Plot"))
  pacf(onlynoise, main = paste0(plotlabel," onlynoise PACF Plot"))
  
  #--Check the residual series for White noise.
  
  sales <- auto.arima(fts, ic="bic" )
  
}

func.WhiteNoise(fts = EU.Consumer.ts, plotlabel="EU Consumer ")
fts <- EU.Consumer.ts[,"Sales"]
sales <- auto.arima(EU.Consumer.ts[,"Sales"], ic="bic" )
