# support vector classifier 

library(caret)
library(e1071)

# read the file

marine.data = read.csv("dolphin_svm.csv", stringsAsFactors = F)

na_vals = sum(is.na(marine.data))

# convert target variable to factor
marine.data$class = factor(marine.data$class)

# generate train and test data 
set.seed(1)
train.indices = sample(1:nrow(marine.data), 0.7*nrow(marine.data))

train.data = marine.data[train.indices, ]
test.data = marine.data[-train.indices, ]

# model 0 with cost = 0.1
model.svm.0 = svm(class ~., data = train.data, kernel = "linear", cost = 0.1, scale = F)  
plot(model.svm.0, train.data)
summary(model.svm.0)


# model 1 with cost = 100
model.svm.1 = svm(class ~., data = train.data, kernel = "linear", cost = 100, scale = F)  
plot(model.svm.1, train.data)
summary(model.svm.1)

# model 2 with cost = 10000
model.svm.2 = svm(class ~., data = train.data, kernel = "linear", cost = 10000, scale = F)  
plot(model.svm.2, marine.data)
summary(model.svm.2)


# finding the optimal value of cost using cross-validation
# svm cross-validation is done using the tune function
# result: higher costs yield lower error rates

tune.svm = tune(svm, class ~., data = train.data, kernel = "linear", ranges = list(cost = c(0.001, 0.01, 0.1, 0.5, 1, 10, 100)))
summary(tune.svm)

# the tune function also stores the best model obtained
# cost = 0.5 is the best model

best.mod = tune.svm$best.model
best.mod




# predicting test classes using the best model and analyzing the table
# best.model is the one with cost = 0.5

ypred = predict(best.mod, test.data)
table(predicted = ypred, truth = test.data$class)
confusionMatrix(ypred, test.data$class)

plot(best.mod, test.data)

 