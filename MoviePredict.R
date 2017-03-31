library(recommenderlab)
library(dplyr)
library(ggplot2)
library(corrplot)
library(Hmisc)
#install.packages("corrgram")
library(corrgram)

movieFile <-  "C:/NewWork/BMS_analyse/Normalised.csv"
movieDF1  <- na.omit(read.csv(movieFile, ",", header = T, stringsAsFactors = F))

EventFile <-  "C:/NewWork/BMS_analyse/Event_Venue_data_new.csv"
EventDF1  <- na.omit(read.csv(EventFile, ",", header = T, stringsAsFactors = F))

rownames(movieDF1) <- movieDF1$CUSTOMER_MAILID_nan_nan
movieDF1$CUSTOMER_MAILID_nan_nan <- NULL
str(movieDF1)
rownames(movieDF1)
m1 <- as.matrix(movieDF1)
tofm <-  t(m1)
M <- cor(tofm)
corrplot(M, method="circle")
corrgram( tofm )

rownames(EventDF1) <- EventDF1$EVENT_NAME_nan_nan
EventDF1$EVENT_NAME_nan_nan <- NULL
Eventmatrix <- t(as.matrix(EventDF1))
EventmatrixOrig <- as.matrix(EventDF1)
t(apply(EventmatrixOrig, 1, function(x)(x-min(x))/(max(x)-min(x))))


finalCor <- read.table(text = "", col.names = colnames(movieDF1))
for( j in 1:nrow(m1)) {
  MEcor <- list() 
  for( i in 1:nrow(EventmatrixOrig)) {
  #print(length(Eventmatrix[,i]))
  MEcor[i] <- cor( as.vector(m1[j,]), as.vector(EventmatrixOrig[i,]))
  #colnames(finalCor)[i+1] <- colnames(EventmatrixOrig)[i]
  #corrplot(MEcor, method="circle")  
  }
  finalCor[nrow(finalCor)+1,] <- MEcor
}

rownames(finalCor) <- rownames(movieDF1)
finalCor[[1]] <- NULL

#movieDF1['002akhilulrocks@gmail.com',]
#movieDF1['000anjinigowtham@gmail.com',]
#movieDF1['001.surya@gmail.com']
#corrplot(movieDF1['002akhilulrocks@gmail.com',],movieDF1['000anjinigowtham@gmail.com',])

####-----------------------------
#### recommend  
####-----------------------------
movieColnames <- as.list(colnames(movieDF1))
eventColnames <- as.list(colnames(EventDF1))

matchCols <- match( movieColnames , names(EventDF1))


