data(iris)
library(ggplot2)
library(e1071)
library(caret)

#store iris dataset in a variable
first_df <- iris
first_df$Sepal_ratio <- first_df$Sepal.Length/first_df$Sepal.Width
first_df$Petal_ratio <- first_df$Petal.Length/first_df$Petal.Width

#create a variable to split into train and test datasets
smp_size <- floor(0.70 * nrow(first_df))

## set the seed to make your partition reproductible
set.seed(100)
train_ind <- sample(seq_len(nrow(first_df)), size = smp_size)


#split into train and test
train_data <- first_df[train_ind,]
test_data <- first_df[-train_ind,]
train_data$Species <- as.factor(train_data$Species)

#create two variables for a model that only takes in length
train <- subset(train_data, select = -c(Sepal.Width, Petal.Width))
test <- subset(test_data, select = -c(Sepal.Width, Petal.Width))
train$Species <- as.factor(train$Species)


#model
svm_model <- svm(Species ~ Sepal.Length + Petal.Length , train, cost = 1, scale = F, kernel = "linear")
svm_model_1000 <- svm(Species ~ Sepal.Length + Petal.Length , train, cost = 1000, scale = F, kernel = "linear")

svm_model_2 <- svm(Species ~ Sepal.Length + Petal.Length + Sepal.Width + Petal.Width , train_data, cost = 1, scale = F, kernel = "linear")

plot(svm_model,train)

#fitted results
fitted.results <- predict(svm_model, subset(test,select=1:ncol(test)-1))
fitted.results_w <- predict(svm_model_2, subset(test_data,select=1:ncol(test_data)-1))

#error
misClasificError <- mean(fitted.results != test$Species)
misClasificError_w <- mean(fitted.results_w != test$Species)

#summary and details of model
summary(svm_model)
summary(svm_model_1000)
summary(svm_model_2)
#Accuracy of model
print(paste('Accuracy',1-misClasificError))
print(paste('Accuracy 2',1-misClasificError_w))

table(fitted.results,test$Species)
