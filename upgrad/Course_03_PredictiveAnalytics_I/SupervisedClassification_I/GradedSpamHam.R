library(e1071)
library(caret)

# Setting user directory
setwd("C:\\Users\\anandrathi\\Documents\\ramiyampersonal\\Personal\\Upgrad\\Course_03\\SupervisedClassification_I\\")
# Importing data
spam_ham<-read.csv("spam_ham.csv", header = TRUE, sep = ',', stringsAsFactors= T)
summary(spam_ham)

train<-spam_ham
test<-spam_ham
# Removing the label column (type of mushroom) from the test data
test1<- c("free", "report", "buy", "click")
# Now we will run Naive Bayes algorithm on this data: Using the e1071 package
model <- naiveBayes(Class~. , data = train)

Freq.1<-c("free")
Freq.2<-c("data")
Freq.3<-c("weekend")
Freq.4<-c("click")
test2<- data.frame(Freq.1,Freq.2,Freq.3,Freq.4)  



pred <- predict(model, test1)
#table(pred, mush_test$Type.of.mushroom)
pred <- predict(model, test2)
