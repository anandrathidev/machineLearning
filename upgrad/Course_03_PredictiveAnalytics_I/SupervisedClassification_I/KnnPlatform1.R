#Importing Data set
data<-read.csv("knndata.csv", header = TRUE, sep = ',', stringsAsFactors= FALSE)

#Removing the first row containing serial numbers
data <- data[, -1]               
               
#Check For Missing Values in the Data set
sum(is.na(data))
         
#You can check for outliers yourself
               
               
#Correcting the Data Type of the target Variable
data$Credit.Rating <- as.factor(data$Credit.Rating)
               
               
#Converting Categorical Variables to Numeric by Dummy encoding
dummy <- as.data.frame(model.matrix( ~ Balance.of.Current.Account -1,
                                              data = data)) 
dummy1 <- as.data.frame(model.matrix( ~ Payment.of.Previous.Credits -1,
                                              data = data)) 
dummy2 <- as.data.frame(model.matrix( ~ Purpose.of.Credit -1,
                                              data = data))
dummy3 <- as.data.frame(model.matrix( ~ Value.of.Savings -1,
                                              data = data))
dummy4 <- as.data.frame(model.matrix( ~ Employed.by.Current.Employer.for -1,
                                              data = data))
dummy5 <- as.data.frame(model.matrix( ~ Installment.in...of.Available.Income -1,
                                             data = data))
dummy6 <- as.data.frame(model.matrix( ~ Marital.Status -1,
                                             data = data))
             
# Gender has 2 categories. So no dummy variables needed
data$Gender <- as.numeric(as.factor(data$Gender))
               
#Dummy Variable Creation continues
dummy7 <- as.data.frame(model.matrix( ~ Living.in.Current.Household.for -1,
                                                     data = data))
dummy8 <- as.data.frame(model.matrix( ~ Most.Valuable.Assets -1,
                                                     data = data))
dummy9 <- as.data.frame(model.matrix( ~ Further.running.credits -1,
                                                     data = data))
dummy10 <- as.data.frame(model.matrix( ~ Type.of.Apartment -1,
                                                      data = data))
dummy11 <- as.data.frame(model.matrix( ~ Number.of.previous.credits.at.this.bank -1,
                                                      data = data))
dummy12 <- as.data.frame(model.matrix( ~ Occupation -1,
                                                      data = data))
               
#Collating the data together. 
#Keeping only the original numeric variabes and the dummy numeric variables
data1 <- cbind(data[, c(1, 3, 6, 11, 14)], dummy[, -1], dummy1[,-1],
                  dummy2[,-1], dummy3[,-1], dummy4[,-1], dummy5[, -1],
                  dummy6[, -1], dummy7[, -1], dummy8[, -1], dummy9[, -1],
                  dummy10[, -1], dummy11[, -1], dummy12[, -1])
               
               
# Scaling the numeric variables 
data1$Credit.Rating <- as.factor(data1$Credit.Rating)
data1$Duration.of.Credit <- scale(data1$Duration.of.Credit)
data1$Amount.of.Credit <- scale(data1$Amount.of.Credit)
data1$Age <- scale(data1$Age)
               
               
               
# Splitting into training and testing
set.seed(2)
s=sample(1:nrow(data1),0.7*nrow(data1))
           
# training data contains 70% of the data
data_train=data1[s,]
             
#testing data contains 30% of the data
data_test=data1[-s,]
               
# True class labels of training data
cl <- data_train[, 1]
               
#Training and testing data without the true labels
data_train <- data_train[,-1]
data_test1 <- data_test[, -1]
           
               
#KNn with 1NN
impknn1 <- knn(data_train, data_test1, cl, k = 1,
                              prob = TRUE)
table(impknn1,data_test[,1])
confusionMatrix(impknn1, data_test[,1], positive ="good" )
               
               
#KNN - 3 Nearest neighbours
impknn3 <- knn(data_train,data_test1, cl, k = 3,
                          prob = TRUE)
table(impknn3,data_test[,1])
confusionMatrix(impknn3, data_test[,1], positive ="good")
               
#KNN - 5 Nearest Neighbours
impknn5 <- knn(data_train,data_test1, cl, k = 5,
                            prob = TRUE)
table(impknn5,data_test[,1])
confusionMatrix(impknn5, data_test[,1], positive ="good")
               
#KNN - 7 Nearest Neighbours
impknn7 <- knn(data_train,data_test1, cl, k = 7,
                            prob = TRUE)
table(impknn7,data_test[,1])
confusionMatrix(impknn7, data_test[,1], positive = "good")
               

               
#calculating the values for ROC curve
pred <- prediction(attr(impknn1,"prob"), data_test[,"Credit.Rating"])
perf <- performance(pred,"tpr","fpr")
               
# changing params for the ROC plot - width, etc
par(mar=c(5,5,2,2),xaxs = "i",yaxs = "i",cex.axis=1.3,cex.lab=1.4)
               
# plotting the ROC curve
plot(perf,col="black",lty=3, lwd=3)
               
# calculating AUC
auc <- performance(pred,"auc")