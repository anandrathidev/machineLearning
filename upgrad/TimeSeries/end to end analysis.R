#Loading the required libraries

library(forecast)
require(graphics)

filename <- c("sales-data.csv")
ylab <- c("Sales Registered")
xlab <- c("Months from Jan 1965")
title <- c("Sales of company X, Jan. 1965 to May 1971")
xcol <- c(1)
ycol <- c(2)

example <- 1
rawdata <- read.csv(filename[example])
timeser <- ts(rawdata[,ycol[example]])
plot(timeser)

w <-1
smoothedseries <- filter(timeser, 
                         filter=rep(1/(2*w+1),(2*w+1)), 
                         method='convolution', sides=2)

diff <- smoothedseries[w+2] - smoothedseries[w+1]
for (i in seq(w,1,-1)) {
  smoothedseries[i] <- smoothedseries[i+1] - diff
}
n <- length(timeser)
timevals <- rawdata[[xcol[example]]]
diff <- smoothedseries[n-w] - smoothedseries[n-w-1]
for (i in seq(n-w+1, n)) {
  smoothedseries[i] <- smoothedseries[i-1] + diff
}
lines(smoothedseries, col="blue", lwd=2)
smootheddf <- as.data.frame(cbind(timevals, as.vector(smoothedseries)))
colnames(smootheddf) <- c('month', 'sales')

lmfit <- lm(rawdata$Sales ~ sin(0.5*rawdata$Month) * poly(rawdata$Month,2) + cos(0.5*rawdata$Month) * poly(rawdata$Month,2) + rawdata$Month, data=rawdata)
trend <- predict(lmfit, data.frame(x=timevals))
lines(timevals, trend, col='red', lwd=2)

resi <- timeser - trend
plot(resi, col='red')
acf(resi)
acf(resi, type="partial")
armafit <- auto.arima(resi)

tsdiag(armafit)
armafit

autoarima <- auto.arima(timeser)
autoarima
tsdiag(autoarima)
plot(autoarima$x, col="black")
lines(fitted(autoarima), col="red")
