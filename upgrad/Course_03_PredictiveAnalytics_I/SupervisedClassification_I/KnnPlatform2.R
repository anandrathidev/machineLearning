#Finding an optimal K
#We will use cross validation to do this.
# Splitting into training and testing
set.seed(2)
s=sample(1:nrow(data1),0.7*nrow(data1))
data_train=data1[s,]
data_test=data1[-s,]


#Using the train() command to find the best K.
model <- train(
  Credit.Rating~., 
  data=data_train,
  method='knn',
  tuneGrid=expand.grid(.k=1:50),
  metric='Accuracy',
  trControl=trainControl(
    method='repeatedcv', 
    number=10, 
    repeats=15))

#Generating the plot of the model
model
plot(model)
