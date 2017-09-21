

library(car) 
data <- read.table("http://www.stat.ufl.edu/~winner/data/pgalpga2008.dat")
colnames(data) <- c("AvgDriveDistance","Accuracy", "Gender")
datF <- subset(data, Gender==1, select=1:2)
datM <- subset(data, Gender==2, select=1:2)

summary(datF$AvgDriveDistance)
summary(datF$Accuracy)

scatterplot(AvgDriveDistance~Accuracy, data=datF, xlab="percent accuracy", ylab="average drive distance", main="Enhanced Scatter Plot", labels=row.names(datF))
scatterplot(AvgDriveDistance~Accuracy, data=datM, xlab="percent accuracy", ylab="average drive distance", main="Enhanced Scatter Plot", labels=row.names(datF))


driveaccuF.lm=lm(Accuracy~AvgDriveDistance, data=datF)
summary(driveaccuF.lm)
plot(datF$Accuracy,datF$AvgDriveDistance)
lines(datF$Accuracy,fitted(driveaccuF.lm))       

# 95% posterior interval for the slope
-0.25649 - 0.04424*qt(.975,155)
-0.25649 + 0.04424*qt(.975,155)

coef(driveaccuF.lm)

#AvgDriveDistance=260
#y = 130.8933146 -0.2564907 * AvgDriveDistance
# posterior prediction interval (same as frequentist)
predict(driveaccuF.lm, data.frame(AvgDriveDistance=260),interval="predict")  
130.8933146 - 0.2564907*qt(.975,155)*sqrt(1+1/23+((260 - mean(datF$AvgDriveDistance))^2/22/var(datF$AvgDriveDistance)))


10.82052-2.102*qt(.975,21)*sqrt(1+1/23+((31-mean(T))^2/22/var(T)))
